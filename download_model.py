# download_model.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

local_dir = "./models/bge-large-zh-v1.5"
if not os.path.exists(local_dir):
    print("Downloading model...")
    snapshot_download(
        repo_id="BAAI/bge-large-zh-v1.5",
        local_dir=local_dir,
        local_dir_use_symlinks=False  # 避免符号链接问题（Windows）
    )
print("Model ready at:", local_dir)