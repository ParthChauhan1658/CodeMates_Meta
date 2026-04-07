from huggingface_hub import HfApi
import os

api = HfApi()

skip = {'.venv', '__pycache__', '.pytest_cache', '.git', 'upload_to_hf.py'}

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in skip]
    for file in files:
        local_path = os.path.join(root, file)
        path_in_repo = local_path.lstrip('./').lstrip('.\\').replace('\\', '/')
        print(f'Uploading {path_in_repo}')
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id='ParthChauhan3/customer-service-env',
            repo_type='space',
        )

print('All files uploaded!')
