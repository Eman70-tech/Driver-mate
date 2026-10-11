import subprocess
import sys
import time
import os

def run_script(script_path, working_dir):

    script_name = os.path.basename(script_path)
    print(f"\nSTARTING: {script_name}\n")
    start_time = time.time()

    result = subprocess.run([sys.executable, script_path], cwd=working_dir)
    
    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed! Stopping pipeline.")
        sys.exit(1)
        
    elapsed = time.time() - start_time
    print(f"FINISHED: {script_name} (Took {elapsed:.1f}s)")

def main():
   
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    pipeline_scripts = [
        "main_prep.py",          
        "main_part2.py",         
        "main_finish.py"    
    ]
    
    print("PREPROCESSING MASTER PIPELINE")
    print(f"Project Directory: {project_dir}")
    total_start = time.time()
    
    for script in pipeline_scripts:

        script_full_path = os.path.join(project_dir, script)

        if not os.path.exists(script_full_path):
            print(f"\nERROR: Missing file! Could not find '{script}' inside '{project_dir}'")
            print("Please make sure all files are in the same folder.")
            sys.exit(1)
            
        run_script(script_full_path, working_dir=project_dir)
        
    total_elapsed = time.time() - total_start

    print(f"ALL DONE! Dataset is DL-Ready in {total_elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()