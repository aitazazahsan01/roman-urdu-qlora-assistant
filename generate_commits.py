import os
import random
import subprocess
from datetime import datetime, timedelta

def run_cmd(cmd, env=None):
    subprocess.run(cmd, shell=True, env=env, check=True)

def main():
    # Configure git
    run_cmd('git config user.name "Muhammad Aitazaz Ahsan"')
    run_cmd('git config user.email "152264393+aitazazahsan01@users.noreply.github.com"')
    
    # Get modified files
    res = subprocess.run('git status --porcelain', shell=True, capture_output=True, text=True)
    lines = res.stdout.strip().split('\n')
    files_to_commit = []
    for line in lines:
        if line:
            # line looks like " M file.txt" or "?? file.txt"
            files_to_commit.append(line[3:])

    # Generate dates from May 17 to May 31
    start_date = datetime(2026, 5, 17, 10, 0, 0)
    end_date = datetime(2026, 5, 31, 10, 0, 0)
    
    current_date = start_date
    
    # We need 5-10 commits per day
    # Let's shuffle the files so we can commit them one by one
    random.shuffle(files_to_commit)
    
    commit_messages = [
        "Refactoring code",
        "Update tests",
        "Fix bug in processing",
        "Improve performance",
        "Update documentation",
        "Clean up codebase",
        "Add helper functions",
        "Fix typo",
        "Update configs",
        "Minor adjustments"
    ]
    
    while current_date <= end_date:
        num_commits = random.randint(5, 10)
        
        for i in range(num_commits):
            # random time within the day
            hour = random.randint(9, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            commit_date = current_date.replace(hour=hour, minute=minute, second=second)
            date_str = commit_date.strftime("%Y-%m-%dT%H:%M:%S")
            
            env = os.environ.copy()
            env["GIT_AUTHOR_DATE"] = date_str
            env["GIT_COMMITTER_DATE"] = date_str
            
            if files_to_commit:
                file_to_add = files_to_commit.pop(0)
                run_cmd(f'git add "{file_to_add}"')
                msg = f"Update {os.path.basename(file_to_add)}"
                run_cmd(f'git commit -m "{msg}"', env=env)
            else:
                msg = random.choice(commit_messages)
                run_cmd(f'git commit --allow-empty -m "{msg}"', env=env)
                
        current_date += timedelta(days=1)
        
    # finally push
    print("Pushing to remote...")
    run_cmd('git push origin master')
    print("Done")

if __name__ == "__main__":
    main()
