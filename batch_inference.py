import torch
from torchvision import transforms
from lavis.processors import transforms_video
from lavis.datasets.data_utils import load_video_demo
from lavis.processors.blip_processors import ToUint8, ToTHWC
from lavis.models.sevila_models.sevila import SeViLA
import numpy as np
from PIL import Image
import glob
import time
import os
import json
from tqdm import tqdm
import gc
import argparse

def load_existing_results(output_path):
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing {output_path}. Starting fresh.")
            return []
    return []

def save_results(results, output_path):
    # Sort results by qid before saving
    sorted_results = sorted(results, key=lambda x: (x['vid'], x['qid']))
    with open(output_path, 'w') as f:
        json.dump(sorted_results, f, indent=4)

def process_video(sevila, video_name, video_path, question, device, transform, img_size, video_frame_num, keyframe_num, LOC_prompt):
    try:
        # Load and process video
        raw_clip, indice, fps, vlen = load_video_demo(
            video_path=video_path,
            n_frms=video_frame_num,
            height=img_size,
            width=img_size,
            sampling="uniform",
            clip_proposal=None
        )

        clip = transform(raw_clip.permute(1, 0, 2, 3))
        clip = clip.float().to(device)
        clip = clip.unsqueeze(0)

        # Prepare input text
        text_input_qa = ""
        text_input_loc = f"Question: {question} {LOC_prompt}"

        # Run inference
        out = sevila.generate_frame_indices(clip, text_input_qa, text_input_loc, keyframe_num)
        select_index = out['frame_idx'][0]

        # Calculate frame paths
        video_len = len(glob.glob(f'/playpen-nas-ssd4/awang/scannet_scenes/2d_color_images/{video_name}/color/*.jpg'))
        frame_paths = []
        for i in select_index:
            select_i = indice[i]
            real_frame_idx = int((select_i / vlen) * video_len)
            frame_path = f"/playpen-nas-ssd4/awang/scannet_scenes/2d_color_images/{video_name}/color/{real_frame_idx}.jpg"
            frame_paths.append(frame_path)

        # Clean up GPU memory
        del raw_clip, clip, out
        torch.cuda.empty_cache()
        gc.collect()

        return frame_paths
    except Exception as e:
        # Ensure cleanup even if there's an error
        torch.cuda.empty_cache()
        gc.collect()
        raise e

def load_eval_queries(eval_file):
    queries = []
    with open(eval_file, 'r') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run SeViLA batch inference')
    parser.add_argument('--key-frames', type=int, default=8,
                      help='Number of key frames to select per video (default: 8)')
    args = parser.parse_args()

    # Update output path to include frame numbers
    output_path = f"selected_images_sevila_k{args.key_frames}.json"
    eval_file = "/playpen-nas-ssd4/awang/UniVTG/eval.jsonl"
    
    # Load any existing results
    results = load_existing_results(output_path)
    print(f"Loaded {len(results)} existing results from {output_path}")

    # Create a set of processed query-video pairs for faster lookup
    processed_pairs = {(r['vid'], r['qid']) for r in results}

    # Load queries from eval.jsonl
    queries = load_eval_queries(eval_file)
    print(f"Loaded {len(queries)} queries from {eval_file}")

    # Model configuration
    img_size = 224
    num_query_token = 32
    t5_model = 'google/flan-t5-xl'
    drop_path_rate = 0
    use_grad_checkpoint = False
    vit_precision = "fp16"
    freeze_vit = True
    prompt = ''
    max_txt_len = 77
    answer_num = 5
    apply_lemmatizer = False
    task = 'freeze_loc_freeze_qa_vid'
    video_frame_num = 200
    keyframe_num = args.key_frames

    # Prompts
    LOC_prompt = 'Does the information within the frame provide the necessary details to accurately answer the given question?'

    # Video processing configuration
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([
        ToUint8(),
        ToTHWC(),
        transforms_video.ToTensorVideo(),
        normalize
    ])

    print('Loading SeViLA model...')
    model_load_start = time.time()
    sevila = SeViLA(
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        use_grad_checkpoint=use_grad_checkpoint,
        vit_precision=vit_precision,
        freeze_vit=freeze_vit,
        num_query_token=num_query_token,
        t5_model=t5_model,
        prompt=prompt,
        max_txt_len=max_txt_len,
        apply_lemmatizer=apply_lemmatizer,
        frame_num=4,
        answer_num=answer_num,
        task=task,
    )

    sevila.load_checkpoint(url_or_filename='https://huggingface.co/Shoubin/SeViLA/resolve/main/sevila_pretrained.pth')
    model_load_time = time.time() - model_load_start
    print(f'Model loaded successfully! Time taken: {model_load_time:.2f} seconds')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        sevila = sevila.float()
    sevila = sevila.to(device)

    # Process each query
    for query_entry in tqdm(queries, desc="Processing queries", unit="query"):
        video_name = query_entry['vid']
        question = query_entry['query']
        query_id = query_entry['qid']
        
        # Skip if already processed
        if (video_name, query_id) in processed_pairs:
            tqdm.write(f"Skipping already processed query {query_id} for video {video_name}")
            continue

        video_path = f"/playpen-nas-ssd4/awang/scannet_scenes/videos/{video_name}.mp4"
        if not os.path.exists(video_path):
            tqdm.write(f"Video file not found for {video_name}, skipping...")
            continue

        tqdm.write(f"\nProcessing video: {video_name}, Query: {question}")
        
        try:
            frame_paths = process_video(
                sevila=sevila,
                video_name=video_name,
                video_path=video_path,
                question=question,
                device=device,
                transform=transform,
                img_size=img_size,
                video_frame_num=video_frame_num,
                keyframe_num=keyframe_num,
                LOC_prompt=LOC_prompt
            )
            
            # Add result in the desired format
            result = {
                "qid": query_id,
                "vid": video_name,
                "frames": frame_paths
            }
            results.append(result)
            processed_pairs.add((video_name, query_id))
            
            tqdm.write(f"Successfully processed query {query_id} for {video_name}")
            
            # Save results after each successful processing
            save_results(results, output_path)
            tqdm.write(f"Saved progress to {output_path}")
            
            # Periodically clear GPU memory
            if len(results) % 5 == 0:  # Clear every 5 videos
                torch.cuda.empty_cache()
                gc.collect()
            
        except Exception as e:
            tqdm.write(f"Error processing {video_name} with query {query_id}: {str(e)}")
            continue

    print(f"\nAll processing complete. Final results saved in {output_path}")

if __name__ == "__main__":
    main() 