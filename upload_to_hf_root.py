from huggingface_hub import HfApi
import os

api = HfApi()

# Create the space if it doesn't exist
repo_id = 'ParthChauhan3/customer-service-env'
try:
    api.create_repo(repo_id=repo_id, repo_type='space', space_sdk='docker', exist_ok=True)
    print(f'Space {repo_id} is ready')
except Exception as e:
    print(f'Note: {e}')

# Upload files from customer-service-env directory
files_to_upload = [
    ('customer-service-env/requirements.txt', 'requirements.txt'),
    ('customer-service-env/models.py', 'models.py'),
    ('customer-service-env/client.py', 'client.py'),
    ('customer-service-env/baseline.py', 'baseline.py'),
    ('customer-service-env/README.md', 'README.md'),
    ('Dockerfile', 'Dockerfile'),
    ('openenv.yaml', 'openenv.yaml'),
]

# Upload server directory
server_files = [
    ('customer-service-env/server/__init__.py', 'server/__init__.py'),
    ('customer-service-env/server/app.py', 'server/app.py'),
    ('customer-service-env/server/environment.py', 'server/environment.py'),
    ('customer-service-env/server/fixtures.py', 'server/fixtures.py'),
    ('customer-service-env/server/graders.py', 'server/graders.py'),
    ('customer-service-env/server/tasks.py', 'server/tasks.py'),
    ('customer-service-env/server/tools.py', 'server/tools.py'),
]

# Upload tests directory
test_files = [
    ('customer-service-env/tests/__init__.py', 'tests/__init__.py'),
    ('customer-service-env/tests/test_api.py', 'tests/test_api.py'),
    ('customer-service-env/tests/test_environment.py', 'tests/test_environment.py'),
    ('customer-service-env/tests/test_graders.py', 'tests/test_graders.py'),
    ('customer-service-env/tests/test_models.py', 'tests/test_models.py'),
    ('customer-service-env/tests/test_reward.py', 'tests/test_reward.py'),
    ('customer-service-env/tests/test_tools.py', 'tests/test_tools.py'),
]

all_files = files_to_upload + server_files + test_files

for local_path, repo_path in all_files:
    if os.path.exists(local_path):
        print(f'Uploading {repo_path}')
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type='space',
            )
        except Exception as e:
            print(f'Error uploading {repo_path}: {e}')
    else:
        print(f'Skipping {local_path} (not found)')

print('All files uploaded!')
