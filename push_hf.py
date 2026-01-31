from huggingface_hub import HfApi, create_repo

api = HfApi()

# Create the repo first
create_repo("Prahaladha/telugu-diffusion-lm", exist_ok=True)

# Then upload
api.upload_folder(
    folder_path="./telugu-diffusion-lm-reconstructed",
    repo_id="Prahaladha/telugu-diffusion-lm",
    commit_message="Upload Telugu Diffusion LM with model card"
)