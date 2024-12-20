#!/bin/bash
#SBATCH --gres=gpu:a100_3g.40gb:1

docker compose up
