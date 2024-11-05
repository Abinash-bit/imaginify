import subprocess
import tempfile
import os

OUTPUT_DIR = "./static/outputs/"

def count_files(directory, pattern):
    file_count = 0
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Filter files matching the pattern
        matched_files = fnmatch.filter(files, pattern)
        file_count += len(matched_files)
    
    return file_count

def combine_videos(scenes, product_name):

    prefix = "final_video_"
    file_count = count_files(OUTPUT_DIR, f"{prefix}{product_name}*.mp4")
    file_name = f"{prefix}{product_name}_{file_count}.mp4"
    output_path = f"{OUTPUT_DIR}{file_name}"

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as list_file:
        for scene in scenes:
            list_file.write(f"file '{scene["video_url"]}'\n")
        list_file_name = list_file.name

    command = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file_name, '-c', 'copy', output_path]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during video concatenation: {e}")
        raise e
    finally:
        os.remove(list_file_name)
        return output_path