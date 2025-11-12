# Project 2 – Containerization (Starter)

This repo demonstrates a reproducible ML training workflow in Docker and GitHub Codespaces.


## TL;DR


```bash
# Local Docker build
docker build -t ml-trainer .


# Local run (overrides defaults from Dockerfile CMD)
docker run --rm -it -v $(pwd):/workspace ml-trainer \
--checkpoint_dir models --lr 1e-3 --epochs 5