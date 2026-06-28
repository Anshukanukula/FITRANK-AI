import os
import requests
import time

url = 'https://huggingface.co/BAAI/bge-m3/resolve/main/pytorch_model.bin'
dest_dir = './models/bge-m3'
dest_file = os.path.join(dest_dir, 'pytorch_model.bin')

os.makedirs(dest_dir, exist_ok=True)

print("Starting direct download of pytorch_model.bin (2.2GB) to", dest_file)
start_time = time.time()

# Check if we can resume or start fresh
resume_header = {}
existing_size = 0
if os.path.exists(dest_file):
    existing_size = os.path.getsize(dest_file)
    # If the file is already 2.2GB, it is likely fully downloaded
    if existing_size > 2200 * 1024 * 1024:
         print("File already downloaded successfully!")
         exit(0)
    print(f"Resuming download from byte position {existing_size}...")
    resume_header = {'Range': f'bytes={existing_size}-'}

try:
    res = requests.get(url, headers=resume_header, stream=True, timeout=15)
    
    # If server doesn't support range requests, start fresh
    if res.status_code == 200:
        total_size = int(res.headers.get('content-length', 0))
        write_mode = 'wb'
        print(f"Starting fresh download. Total size: {total_size / (1024*1024):.2f} MB")
    elif res.status_code == 206:
        total_size = int(res.headers.get('content-range', '').split('/')[-1])
        write_mode = 'ab'
        print(f"Resuming download. Total size: {total_size / (1024*1024):.2f} MB")
    else:
        print(f"Server responded with status {res.status_code}. Starting fresh...")
        res = requests.get(url, stream=True, timeout=15)
        total_size = int(res.headers.get('content-length', 0))
        write_mode = 'wb'
        existing_size = 0
        
    downloaded = existing_size
    last_print_time = time.time()
    last_print_bytes = downloaded
    
    with open(dest_file, write_mode) as f:
        for chunk in res.iter_content(chunk_size=1024 * 1024): # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                # Print progress every 5 seconds
                current_time = time.time()
                if current_time - last_print_time > 5.0:
                    speed = (downloaded - last_print_bytes) / (1024 * 1024 * (current_time - last_print_time))
                    percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                    print(f"Downloaded: {downloaded / (1024*1024):.2f}/{total_size / (1024*1024):.2f} MB | {percent:.1f}% | Speed: {speed:.2f} MB/s")
                    last_print_time = current_time
                    last_print_bytes = downloaded
                    
    print(f"Download complete! Total time: {time.time() - start_time:.2f} seconds.")
except Exception as e:
    print(f"Download error: {e}")
