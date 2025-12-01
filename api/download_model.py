import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_phi1_model():
    """Download Phi-1 model from HuggingFace"""
    
    MODEL_ID = "microsoft/phi-1"
    CACHE_DIR = "models"
    MODEL_PATH = os.path.join(CACHE_DIR, "phi1")
    
    print(f"🚀 Starting Phi-1 model download...")
    print(f"Model ID: {MODEL_ID}")
    print(f"Save path: {MODEL_PATH}")
    
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    try:
        # Download the model and tokenizer
        print("\n📥 Downloading model files...")
        local_dir = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=CACHE_DIR,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=False,
        )
        
        print(f"\n✅ Download complete!")
        print(f"Model saved to: {local_dir}")
        
        # Verify model files
        required_files = ["config.json", "pytorch_model.bin", "model.safetensors", "generation_config.json"]
        found_files = []
        
        print("\n🔍 Verifying model files...")
        for file in os.listdir(local_dir):
            if file in required_files or file.startswith("pytorch_model") or file.startswith("model.safetensors"):
                found_files.append(file)
                print(f"  ✓ {file}")
        
        if found_files:
            print(f"\n✨ Model is ready! Found {len(found_files)} model files")
            return True
        else:
            print("\n⚠️  Warning: Some expected files not found")
            print(f"Files in directory: {os.listdir(local_dir)}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = download_phi1_model()
    