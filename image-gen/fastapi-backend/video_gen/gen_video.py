import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import fnmatch
import os
import numpy as np
import PIL.Image
import cv2

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V",
        torch_dtype=torch.bfloat16,cache_dir='/data/hf_cache'
    )

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

OUTPUT_DIR = "./static/outputs/"

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def count_files(directory, pattern):
    file_count = 0
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Filter files matching the pattern
        matched_files = fnmatch.filter(files, pattern)
        file_count += len(matched_files)
    
    return file_count

def gen_video(prompt, image_url, prefix = "GenVideo_"):
    try :
        
        image = load_image(image=image_url)
        
        video = pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        
        file_count = count_files(OUTPUT_DIR, f"{prefix}*.mp4")
        export_to_video(video, f"{OUTPUT_DIR}{prefix}{file_count}.mp4", fps=8)
        
        if isinstance(video[0], np.ndarray):
            video = [(frame * 255).astype(np.uint8) for frame in video]

        elif isinstance(video[0], PIL.Image.Image):
            video = [np.array(frame) for frame in video]
            
       
        last_frame = video[-1]
        # save last_frame as png
        counter = 0
        for frame in video:
            cv2.imwrite(f"{OUTPUT_DIR}frame{prefix}_{file_count}_{counter}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            counter += 1
        cv2.imwrite(f"{OUTPUT_DIR}last_frame{prefix}{file_count}.png", cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))

        return [f"{OUTPUT_DIR}{prefix}{file_count}.mp4", f"{OUTPUT_DIR}last_frame{prefix}{file_count}.png"]
    except Exception as e:
        print(e)
        raise e