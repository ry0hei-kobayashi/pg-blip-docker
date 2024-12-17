#!/bin/bash
#SBATCH --gres=gpu:a100:1

docker compose up
