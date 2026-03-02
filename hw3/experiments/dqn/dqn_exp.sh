#!/bin/bash

# Define the list of commands
commands=(
    "python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole.yaml"
    "python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole_biglr.yaml"
    "python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1"
    "python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 2"
    "python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 3"
    "python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 1"
    "python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 2"
    "python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 3"
)

# Loop through and execute
for cmd in "${commands[@]}"; do
    echo "------------------------------------------------"
    echo "Running: $cmd"
    echo "------------------------------------------------"
    eval $cmd
done

echo "All experiments completed!"