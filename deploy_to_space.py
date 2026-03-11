import os
from huggingface_hub import HfApi, login

# Configuration
# This is the name of the space you created (sm89/health_score)
SPACE_ID = "sm89/health_score" 

def deploy():
    print(f"--- Deploying to Hugging Face Space: {SPACE_ID} ---")
    
    # 1. Login
    login()
    
    api = HfApi()

    # 2. Prepare the README.md with Metadata for Streamlit
    # The SDK must be set to 'streamlit' for it to work.
    readme_content = """---
title: Cardiovascular Health Risk Predictor
emoji: ❤️
colorFrom: red
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# Cardiovascular Health Risk Predictor

This is a Streamlit application that uses Machine Learning to estimate cardiovascular health risks.
The model is loaded dynamically from the Hugging Face Model Hub.
"""

    with open("temp_README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    try:
        # 3. Upload app_hf.py as app.py (the entry point for Spaces)
        print("Uploading app.py...")
        api.upload_file(
            path_or_fileobj="app_hf.py",
            path_in_repo="app.py",
            repo_id=SPACE_ID,
            repo_type="space"
        )

        # 4. Upload requirements_space.txt as requirements.txt
        print("Uploading requirements.txt...")
        api.upload_file(
            path_or_fileobj="requirements_space.txt",
            path_in_repo="requirements.txt",
            repo_id=SPACE_ID,
            repo_type="space"
        )

        # 5. Upload README.md (Metadata)
        print("Updating Space metadata (README.md)...")
        api.upload_file(
            path_or_fileobj="temp_README.md",
            path_in_repo="README.md",
            repo_id=SPACE_ID,
            repo_type="space"
        )

        print(f"\n🚀 Success! Your app is being deployed at: https://huggingface.co/spaces/{SPACE_ID}")
        print("It may take a few minutes for the container to build.")

    except Exception as e:
        print(f"Deployment failed: {e}")
    finally:
        if os.path.exists("temp_README.md"):
            os.remove("temp_README.md")

if __name__ == "__main__":
    deploy()
