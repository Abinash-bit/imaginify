from fastapi import FastAPI, Request, File, UploadFile, Response, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import re
import json
from llm_i.qwen_script_gen import generate_script
# from img_caption.caption_img import generate_image_caption
from gen_img.gen_img import generate_image
from video_gen.gen_video import gen_video
from combine_video.combine_video import combine_videos
import os
import io
import PyPDF2

os.environ['HF_HOME'] = '/data/hf_cache'
# Initialize FastAPI app
app = FastAPI()
UPLOAD_DIR = "static/uploads"

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



def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Endpoint to generate script using Qwen 2.5
@app.post("/generate_script")
async def generate_script_api(
    file: UploadFile = File(...),
    product_name: str = Form(...),
):
    try:
        # Read and process the uploaded PDF file
        file_content = await file.read()
        extracted_text = extract_text_from_pdf(file_content)
        print("Extracted Text: ", extracted_text)
        
        # Create a ScriptRequest object with the extracted text
        request = ScriptRequest(text=extracted_text, product_name=product_name)
        print("Request: ", request)
        
        generated_script = generate_script(request)
        scenes = generated_script['scenes'] # arrayof scene
        
        # for each scene in the scenes, extract the text and image
        counter = 0
        for scene in scenes:
            print("Scene: ", scene)
            image_prompt = scene["image_prompt"]
            video_prompt = scene["video_prompt"]
            script = scene["script"]
            print("Image Prompt: ", image_prompt)
            # print("Video Prompt: ", video_prompt)
            # print("Script: ", script)
            # generate image for the image prompt
            image_url = generate_image(image_prompt)
            scene["first_frame_url"] = image_url
            # # Generate caption for the image
            # caption = generate_image_caption(image_url)
            # # generate video for the video prompt
            video_result = gen_video(video_prompt, image_url)
            scene["video_url"] = video_result[0]
            scene["last_frame_url"] = video_result[1]
            # scene.caption = caption
            counter += 1
        
        # combine the scenes into a single script by combining the videos
        for scene in scenes:
            print(scene)
        # return the generated script
        generated_script['final_vidoe_url'] = combine_videos(scenes, product_name)
        return Response(generated_script, status_code=200)
        

    except Exception as e:
        return {"error": str(e)}

# Endpoint to generate Image using trained flux model
@app.post("/generate_image")
async def generate_image_api(
    prompt: str = Form(...),
):
    try:

        print("Image Prompt: ", prompt)
        image_url = generate_image(prompt)
        # Check if the image file exists
        if not os.path.isfile(image_url):
            raise HTTPException(status_code=404, detail="Image not found.")

        # Option 1: Using FileResponse to send the image file directly
        return FileResponse(
            path=image_url,
            media_type="image/png",
            filename=os.path.basename(image_url)
        )

        # Option 2: Read the image into a buffer and send it
        # with open(image_url, "rb") as image_file:
        #     image_data = image_file.read()
        # return StreamingResponse(
        #     io.BytesIO(image_data),
        #     media_type="image/png"
        # )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8188)