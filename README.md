# DL43D

## Name

DL43D Perception Praktikum

## Description

This repo contains all related work for the DL43D perception course.

## Visuals

- Any interesting visuals such as overview figure & training curves ..etc

## Installation

##### Steps neededed to run the project on the 3DML cluster provided by TUM

##### Note: This is an old configuaration & needs to be revised and re-tested when access to the cluster is obtained

- `ssh <user>@ml3d.vc.in.tum.de` - ssh into the login node
  - You will be prompted to enter your password
- `salloc --gpus=1`
- `mkdir /cluster/54/<user>`
- `cd /cluster/54/<user>`
- `git clone https://<username>:<personal_token>@gitlab.com/gitlab-org/gitlab.git`
- `cd dl43d`
- `poetry install`
- `poetry shell`
- `pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
- `poetry run jupyter notebook --no-browser --ip=0.0.0.0 --port=8888`
- From your local machine: `ssh -NL 3000:TUINI15-<connected_node>.vc.in.tum.de:8888 <user>@ml3d.vc.in.tum.de`
- Run the `initial_setup.ipynb` notebook to obtain the dataset

## Usage

Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Roadmap

If you have ideas for releases in the future, it is a good idea to list them in the README.

## General Links

##### This section contains any useful links found

##### This section contains any useful links found

## Authors and acknowledgment

- Mino Erstella - `email here`
- Youssef Youssef - youssef@youssef.tum.de

## Project status

The project is currently in the early development phase.
