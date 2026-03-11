import os
from huggingface_hub import HfApi, login

def upload_to_hf():
    """
    Uploads the model and scaler files to Hugging Face.
    """
    print("--- Hugging Face Upload Tool ---")
    
    # 1. Login to Hugging Face
    # If you have the HF_TOKEN environment variable set, this will use it.
    # Otherwise, it will prompt you for your Write token.
    login()

    api = HfApi()

    # 2. Configuration
    repo_id = "sm89/health_score"  # Replace with your desired repo ID
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_folder = os.path.join(project_root, "Model")

    if not os.path.exists(model_folder):
        print(f"Error: Model folder not found at {model_folder}")
        return

    print(f"Checking repository '{repo_id}'...")
    
    try:
        # 3. Create repository if it doesn't exist
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        
        # 4. Upload the contents of the Model folder
        print(f"Uploading files from {model_folder} to HF Hub...")
        api.upload_folder(
            folder_path=model_folder,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial upload of model and scaler"
        )
        
        print(f"\nSuccess! Your model is now available at: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Make sure your token has 'Write' permissions.")

if __name__ == "__main__":
    upload_to_hf()
