import os
from huggingface_hub import snapshot_download

def download_phi35_mini():
    MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
    CACHE_DIR = "models"
    MODEL_PATH = os.path.join(CACHE_DIR, "phi-3.5-mini")

    print(f"🚀 Downloading: {MODEL_ID}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        local_dir = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=CACHE_DIR,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print("\n✅ Download finished!")
        print("Saved to:", local_dir)
        print("\nFiles:")
        for f in os.listdir(local_dir):
            print(" •", f)
        return True

    except Exception as e:
        print("\n❌ Error:", e)
        return False


if __name__ == "__main__":
    download_phi35_mini()
