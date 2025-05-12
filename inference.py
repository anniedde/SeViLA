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

def main():
    total_start_time = time.time()
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

    # Prompts
    LOC_propmpt = 'Does the information within the frame provide the necessary details to accurately answer the given question?'

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
    
    video = 'scene0000_00'
    video_path = "/playpen-nas-ssd4/awang/scannet_scenes/videos/" + video + '.mp4'
    question = "What color is the chair?"
    video_frame_num = 200
    keyframe_num = 8

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        sevila = sevila.float()
    sevila = sevila.to(device)

    # Load and process video
    print(f"Loading video: {video_path}")
    video_load_start = time.time()
    raw_clip, indice, fps, vlen = load_video_demo(
        video_path=video_path,
        n_frms=video_frame_num,
        height=img_size,
        width=img_size,
        sampling="uniform",
        clip_proposal=None
    )
    video_load_time = time.time() - video_load_start
    #print(f"Raw clip shape: {raw_clip.shape}")
    #print(f"Indices length: {len(indice)}")
    #print(f"Indices content: {indice}")

    clip = transform(raw_clip.permute(1, 0, 2, 3))
    clip = clip.float().to(device)
    clip = clip.unsqueeze(0)

    # Prepare input text
    text_input_qa = ""
    text_input_loc = f"Question: {question} {LOC_propmpt}"

    # Run inference
    print("\nRunning inference...")
    inference_start = time.time()
    out = sevila.generate_frame_indices(clip, text_input_qa, text_input_loc, keyframe_num)
    inference_time = time.time() - inference_start
    print(f"Inference time: {inference_time:.2f} seconds")

    # Process results
    select_index = out['frame_idx'][0]
    
    # print(f"Selected frame indices: {select_index}")

    # Calculate timestamps
    video_len = len(glob.glob(f'/playpen-nas-ssd4/awang/scannet_scenes/2d_color_images/{video}/color/*.jpg'))
    real_frame_indices = []
    for i in select_index:
        select_i = indice[i]
        real_frame_idx = (int)((select_i / vlen) * video_len)
        real_frame_indices.append(real_frame_idx)

    """
    # Save keyframes as PNG files
    # print("\nSaving keyframe images...")
    for i, frame_idx in enumerate(select_index):
        #print(raw_clip.dtype, raw_clip.min().item(), raw_clip.max().item())
        
        # Get frame from raw_clip - properly handling the dimensions
        frame = raw_clip[:, frame_idx, :, :].permute(1, 2, 0).byte().numpy()
        
        # Create filename with frame index
        output_path = f"keyframe_{i}_frame{frame_idx}.png"
        
        # Save image using PIL
        Image.fromarray(frame).save(output_path)
        # print(f"Saved {output_path}")
    """

    # Print results
    print("\nResults:")
    print(f"Question: {question}")
    print("Real frame indices:", real_frame_indices)

    total_time = time.time() - total_start_time
    print("\nTiming Summary:")
    print(f"Model loading time: {model_load_time:.2f} seconds")
    print(f"Video loading time: {video_load_time:.2f} seconds")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 