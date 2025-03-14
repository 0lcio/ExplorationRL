import subprocess
import sys

# Percorso dell'interprete Python
python_executable = sys.executable
inputs = ["policy1_entropy", "policy1_loss", "policy1_random", "policy2_ig_reward"]
# inputs = ["policy2_ig_reward"]

for input_str in inputs:
    subprocess.run([python_executable, "test_agent.py", input_str])
