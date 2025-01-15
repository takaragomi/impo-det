import subprocess
import time

for i in range(10):
    print(f"Running main.py - Attempt {i+1}")
    result = subprocess.run(["python", "src/main.py"], capture_output=True, text=True)
    time.sleep(2)