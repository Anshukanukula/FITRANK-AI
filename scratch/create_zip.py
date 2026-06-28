import os
import zipfile
import sys

def create_submission_zip():
    zip_name = "fitrank_ai_submission.zip"
    exclude_dirs = {".git", "__pycache__", "models/bge-m3", "models\\bge-m3"}
    exclude_files = {zip_name}
    
    print(f"Creating compressed ZIP archive: {zip_name}...")
    
    total_uncompressed_size = 0
    file_count = 0
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            # Exclude directories
            dirs[:] = [d for d in dirs if not any(os.path.join(root, d).endswith(ex) or d == ex for ex in exclude_dirs)]
            
            # Check if current root path is under excluded directories
            path_parts = os.path.normpath(root).split(os.sep)
            if any(part in exclude_dirs for part in path_parts):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                # Check if file name matches exclude_files
                if file in exclude_files:
                    continue
                
                # Check if file path is under excluded directories
                relative_path = os.path.relpath(file_path, ".")
                if any(relative_path.startswith(ex) for ex in exclude_dirs):
                    continue
                
                size = os.path.getsize(file_path)
                total_uncompressed_size += size
                file_count += 1
                
                print(f"Adding: {relative_path} ({size / 1024 / 1024:.2f} MB)")
                zipf.write(file_path, relative_path)
                
    zip_size = os.path.getsize(zip_name)
    print("\n" + "="*50)
    print("ZIP ARCHIVE COMPLETED!")
    print(f"Total Files Packed: {file_count}")
    print(f"Total Uncompressed Size: {total_uncompressed_size / 1024 / 1024:.2f} MB")
    print(f"Final Zipped Archive Size: {zip_size / 1024 / 1024:.2f} MB")
    print("="*50)

if __name__ == "__main__":
    create_submission_zip()
