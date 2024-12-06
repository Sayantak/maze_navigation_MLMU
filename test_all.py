"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import subprocess

test_modifications = [
  "use_wandb=False",
  "trainer.max_epochs=1",
  "trainer.check_val_every_n_epoch=1",
  "mode=local",
  "hydra.job.env_set.DATA_ROOT=$(mktemp -d)",
  "logs_dir=$(mktemp -d)"
]

prefix="HYDRA_FULL_ERROR=1"

commands = [
    "python main.py -m dataset=stargraph datamodule.n_train=500 datamodule.n_val=50",
    "python main.py -m exp=past-3m-10x10-astarmaze datamodule.n_mazes=100",
    "python main.py -m exp=past-3m-3x3-maze datamodule.n_mazes=100 model.train_mode=ar",
    "python main.py -m exp=past-3m-3x3-maze datamodule.n_mazes=100 model.train_mode=absorbing",
    "python main.py -m exp=xlnet-3m-3x3-maze datamodule.n_mazes=100",
    "python main.py -m exp=xlnet-3m-3x3-maze datamodule.n_mazes=100 model.train_mode=permutations",
    "python main.py -m exp=gpt-3m-3x3-maze datamodule.n_mazes=100",
]


# Function to execute a shell command
def run_command(command):
    try:
        # Run the command and capture the output
        cmd = prefix + " " + command + " " + " ".join(test_modifications)
        print(cmd)
        result = subprocess.run(cmd, shell=True, check=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Output:\n{result.stdout}")
        print("WORKED")
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(f"Error:\n{e.stderr}")
        exit()

# Main function to loop through and execute commands
def main():
    for command in commands:
        run_command(command)
    print("ALL SUCCESS")

if __name__ == "__main__":
    main()
