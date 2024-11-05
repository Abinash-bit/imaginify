from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import json


# Load Qwen2.5 Instruct model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "microsoft/Phi-3-mini-128k-instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir='/data/hf_cache')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",
    device_map="auto", trust_remote_code=True,cache_dir='/data/hf_cache')

# Define request body for generating script
class ScriptRequest(BaseModel):
    text: str
    product_name: str
    
def extract_json_array(text):
    # Define the regex pattern to match a JSON array in the text
    pattern = r'\[[\s\S]*?\]'
    
    # Search for the first match in the text
    match = re.search(pattern, text)
    
    if match:
        # Extract the matched portion
        json_array_str = match.group(0)
        
        try:
            # Parse the extracted string as JSON
            json_array = json.loads(json_array_str)
            return json_array
        except json.JSONDecodeError:
            print("Error: Extracted string is not a valid JSON array.")
            return None
    else:
        print("No JSON array found in the text.")
        return None

# Endpoint to generate script using Qwen 2.5
def generate_script(request: ScriptRequest):
    try:
        # Prepare the prompt based on the text provided
        prompt_1 = f"""
        Give a JSON as output for a video script based on the Product User Manual where the script is a JSON array with an image prompt that defines the prompt to be used to create the image for the scene, the text script that is to be converted to audio for the scene, and a video prompt to create a video from the image generated for the image prompt. The format should be:
        [{{"image_prompt": "image prompt here", "script": "The text to be used to create audio here", "video_prompt": "the prompt for video generation here"}}].
        We just need around 5 scenes, so explain it in detail. One important thing to consider is not to use the product name mentioned in the user manual; use {request.product_name} instead. Give very detailed image prompts and video prompts. The video prompt should explain the camera movements, interactions, and things to be done in that scene.
        """

        # Combine prompt_1 and the user manual text, limiting the text to 2048 characters if necessary
        prompt = f"{prompt_1}\n{request.text}\n"
        # model.to(device)
        # Tokenize the input
        messages = [
            {"role": "system", "content": "You are part of a Agent system that generated video for a given product based on its user manual."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        # extract the JSON text from generated_text using regex \[[\s\S]*?\]
        
        scenes = extract_json_array(response)
        # move back to CPU to save vram
        # model.to("cpu")
        # del inputs
        # torch.cuda.empty_cache()
        # Return the generated scenes as JSON
        return {"scenes": scenes}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

