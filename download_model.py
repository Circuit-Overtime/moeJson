from huggingface_hub import hf_hub_download
import os

# Target directory
target_dir = "models"
os.makedirs(target_dir, exist_ok=True)

# Download the GGUF model
file_path = hf_hub_download(
    repo_id="professorf/phi-1-GGUF",
    filename="phi-1.Q8_0.gguf",
    local_dir=target_dir,
    local_dir_use_symlinks=False
)

print("Downloaded to:", file_path)
