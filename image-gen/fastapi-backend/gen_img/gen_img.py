import torch
from diffusers import FluxPipeline
import fnmatch
import os

def count_files(directory, pattern):
    file_count = 0
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Filter files matching the pattern
        matched_files = fnmatch.filter(files, pattern)
        file_count += len(matched_files)
    
    return file_count

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float32,use_safetensors=True,cache_dir='/data/hf_cache')
pipe.load_lora_weights("/data/video-gen/fastapi-backend/gen_img/lora/noctua_fingerprint_001.safetensors")
pipe.enable_sequential_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

OUTPUT_DIR = "./static/outputs/"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
def generate_image(prompt, prefix = "GenImg_"):
    # Generate image from prompt
    # pipe.to(device)
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    # pipe.to("cpu") # move back to CPU to save vram
    # finf file count for prefix*.png and then add fileCount+1 to the prefix
    file_count = count_files(OUTPUT_DIR, f"{prefix}*.png")
    filePath = f"{OUTPUT_DIR}{prefix}{file_count}.png"
    print(f"Saving image to {filePath}")
    tmp_file_name = f"file{prefix}{file_count}.png"
    image.save(tmp_file_name)
    os.replace(tmp_file_name, filePath)
    print(f"Image saved to {filePath}")
    return filePath