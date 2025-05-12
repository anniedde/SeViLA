import torch
from torchvision import transforms
from PIL import Image
from lavis.processors import transforms_video
from lavis.processors.blip_processors import ToUint8, ToTHWC
from lavis.models.sevila_models.sevila import SeViLA
import numpy as np
import time
import os
import json
from tqdm import tqdm
import gc
import argparse

def load_frame_selection_data(frames_json_path):
    with open(frames_json_path, 'r') as f:
        return json.load(f)

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

def load_frames_as_tensor(frame_paths, img_size):
    """Load multiple frames and stack them into a video tensor"""
    frames = []
    for frame_path in frame_paths:
        # Convert relative path to absolute path
        abs_frame_path = os.path.join("/playpen-nas-ssd4/awang", frame_path)
        if not os.path.exists(abs_frame_path):
            raise FileNotFoundError(f"Frame not found: {abs_frame_path}")
            
        # Load and resize image
        img = Image.open(abs_frame_path).convert('RGB')
        if img_size != -1:
            img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        # Convert to tensor
        img = transforms.ToTensor()(img)
        img = img * 255  # Scale to 0-255 range
        frames.append(img)
    
    # Stack frames into video tensor (C, T, H, W)
    video_tensor = torch.stack(frames, dim=1)
    return video_tensor

def process_frames(sevila, video_name, frame_paths, question, device, transform, img_size, keyframe_num, LOC_prompt):
    try:
        # Load frames and create video tensor
        raw_clip = load_frames_as_tensor(frame_paths, img_size)
        
        clip = transform(raw_clip.permute(1, 0, 2, 3))  # Convert to (T, C, H, W)
        clip = clip.float().to(device)
        clip = clip.unsqueeze(0)  # Add batch dimension

        # Prepare input text
        text_input_qa = ""
        text_input_loc = f"Question: {question} {LOC_prompt}"

        # Run inference
        out = sevila.generate_frame_indices(clip, text_input_qa, text_input_loc, keyframe_num)
        select_index = out['frame_idx'][0]

        # Get selected frame paths
        selected_frames = [frame_paths[i] for i in select_index]

        # Clean up GPU memory
        del raw_clip, clip, out
        torch.cuda.empty_cache()
        gc.collect()

        return selected_frames
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
    parser = argparse.ArgumentParser(description='Run SeViLA batch inference on pre-selected frames')
    parser.add_argument('--key-frames', type=int, default=8,
                      help='Number of key frames to select per video (default: 8)')
    parser.add_argument('--frames-json', type=str, default='Video-3D-LLM/data/metadata/scannet_select_frames_complete.json',
                      help='Path to JSON file containing pre-selected frames')
    args = parser.parse_args()

    # Update output path to include frame numbers
    output_path = f"selected_images_sevila_voxel_chain_k{args.key_frames}.json"
    eval_file = "/playpen-nas-ssd4/awang/UniVTG/eval.jsonl"
    
    # Load frame selection data
    frame_selection_data = load_frame_selection_data(args.frames_json)
    frame_selection_dict = {data['vid']: data['frames'] for data in frame_selection_data}
    
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

        # Get pre-selected frames for this video
        if video_name not in frame_selection_dict:
            tqdm.write(f"No pre-selected frames found for video {video_name}, skipping...")
            continue

        frame_paths = frame_selection_dict[video_name]
        tqdm.write(f"\nProcessing video: {video_name}, Query: {question}")
        
        try:
            selected_frames = process_frames(
                sevila=sevila,
                video_name=video_name,
                frame_paths=frame_paths,
                question=question,
                device=device,
                transform=transform,
                img_size=img_size,
                keyframe_num=keyframe_num,
                LOC_prompt=LOC_prompt
            )
            
            # Add result in the desired format
            result = {
                "qid": query_id,
                "vid": video_name,
                "selected_frames": selected_frames
            }
            
            results.append(result)
            
            # Save periodically (every 10 queries)
            if len(results) % 10 == 0:
                save_results(results, output_path)
                
        except Exception as e:
            tqdm.write(f"Error processing video {video_name}: {str(e)}")
            continue

    # Save final results
    save_results(results, output_path)
    print(f"\nProcessing complete! Results saved to {output_path}")

if __name__ == "__main__":
    main() 