
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True, device=device,cache_dir='/data/hf_cache')
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True,cache_dir='/data/hf_cache')


def generate_image_caption(image_url, prompt="<MORE_DETAILED_CAPTION>"):
    model.to(device)
    image = Image.open(requests.get(image_url, stream=True).raw)

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>", image_size=(image.width, image.height))
    # move back to CPU to save vram
    model.to("cpu")

    return parsed_answer
