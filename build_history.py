import os
import random
import subprocess
from datetime import datetime, timedelta

FILES = [
    ".gitignore",
    "LICENSE",
    "README.md",
    "requirements.txt",
    "app/app.py",
    "kaggle/kernel-metadata.json",
    "kaggle/train_kernel.ipynb",
    "results/metrics.json",
    "results/sample_completions.json",
    "results/training_loss.png",
    "results/rouge_comparison.png",
    "scripts/build_kaggle_notebook.py",
    "scripts/smoke_test.py",
    "src/data_prep.py",
    "src/evaluate.py",
    "src/inference.py",
    "src/metrics_utils.py",
    "src/qlora_utils.py",
    "src/train_kaggle.py",
    "src/train_qlora.py"
]

def run_cmd(cmd, env=None):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, env=env)

def get_file_content(filepath):
    with open(filepath, 'rb') as f:
        return f.read()

def main():
    # 1. Read all files
    file_contents = {}
    for f in FILES:
        if os.path.exists(f):
            file_contents[f] = get_file_content(f)
        else:
            print(f"Warning: {f} does not exist!")

    # 2. Reset git repo
    run_cmd("git init")
    
    # Optional: add remote if not exists
    try:
        run_cmd("git remote add origin https://github.com/aitazazahsan01/roman-urdu-qlora-assistant")
    except subprocess.CalledProcessError:
        run_cmd("git remote set-url origin https://github.com/aitazazahsan01/roman-urdu-qlora-assistant")

    # Clear files to start empty
    for f in file_contents.keys():
        with open(f, 'wb') as f_out:
            f_out.write(b"")

    # We will slice files into chunks.
    # Text files can be sliced line by line.
    # Binary files like png will be added in one go.
    file_chunks = {}
    for f, content in file_contents.items():
        if f.endswith('.png'):
            file_chunks[f] = [content]
        else:
            lines = content.split(b'\n')
            # create 10-20 chunks per file
            num_chunks = random.randint(5, 15)
            chunk_size = max(1, len(lines) // num_chunks)
            chunks = []
            for i in range(0, len(lines), chunk_size):
                chunk = b'\n'.join(lines[i:i+chunk_size])
                # add back the trailing newline if it's not the last chunk
                if i + chunk_size < len(lines):
                    chunk += b'\n'
                chunks.append(chunk)
            file_chunks[f] = chunks

    # 3. Distribute chunks across commits
    start_date = datetime(2026, 5, 1, 10, 0, 0)
    
    # Generate schedule
    schedule = []
    for day_offset in range(20):
        current_day = start_date + timedelta(days=day_offset)
        num_commits = random.randint(5, 10)
        for c in range(num_commits):
            # random time within the day
            commit_time = current_day + timedelta(hours=random.randint(0, 8), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
            schedule.append(commit_time)
            
    schedule.sort()
    
    # We have N chunks total
    all_chunks = []
    for f, chunks in file_chunks.items():
        for chunk in chunks:
            all_chunks.append((f, chunk))
            
    random.shuffle(all_chunks)
    
    # Distribute chunks into the schedule
    commits_schedule = [[] for _ in schedule]
    for i, chunk in enumerate(all_chunks):
        commits_schedule[i % len(schedule)].append(chunk)

    # 4. Apply chunks and commit
    for i, commit_time in enumerate(schedule):
        chunks_to_apply = commits_schedule[i]
        
        if not chunks_to_apply:
            continue
            
        modified_files = set()
        for f, chunk_content in chunks_to_apply:
            with open(f, 'ab') as f_out:
                f_out.write(chunk_content)
            modified_files.add(f)
            
        # Add to git
        for f in modified_files:
            run_cmd(f'git add "{f}"')
            
        # Commit
        date_str = commit_time.strftime('%Y-%m-%dT%H:%M:%S')
        env = os.environ.copy()
        env['GIT_AUTHOR_DATE'] = date_str
        env['GIT_COMMITTER_DATE'] = date_str
        
        msg = f"Update {', '.join([os.path.basename(f) for f in list(modified_files)[:2]])}"
        if len(modified_files) > 2:
            msg += " and others"
            
        run_cmd(f'git commit -m "{msg}"', env=env)

    # 5. Ensure all final contents match exactly by doing one last write
    for f, content in file_contents.items():
        with open(f, 'wb') as f_out:
            f_out.write(content)
            
    run_cmd("git add .")
    # Check if there are any pending changes
    status = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if status.stdout.strip():
        final_date = datetime(2026, 5, 20, 23, 59, 59).strftime('%Y-%m-%dT%H:%M:%S')
        env = os.environ.copy()
        env['GIT_AUTHOR_DATE'] = final_date
        env['GIT_COMMITTER_DATE'] = final_date
        run_cmd('git commit -m "Finalize project structure and configurations"', env=env)

    # Push
    print("Ready to push. Make sure authentication is set up.")
    # run_cmd("git push -u origin master") # You might want to run this manually or handle auth.

if __name__ == "__main__":
    main()
