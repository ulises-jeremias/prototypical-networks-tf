# Prototypical Networks for Few-shot Learning

Tensorflow v2 implementation of NIPS 2017 Paper _Prototypical Networks for Few-shot Learning_.

## Repository Structure
The repository organized as follows. 

- **data** directory contains scripts for dataset downloading and used as a default directory for datasets.

- **protonet** is the library containing the model itself (`protonet/models`) and logic for datasets loading and processing (`protonet/data`). 

- **scripts** directory contains scripts for launching the training. `train/run_train.py` and `eval/run_eval.py` launch training and evaluation respectively. tests folder contains basic training procedure on small-valued parameters to check general correctness. results folder contains .md file with current configuration and details of conducted experiments.

## Training

Training and evaluation configurations are specified through config files, each config describes single train+eval evnironment.

Run the following command to run training on Omniglot with default parameters.

```sh
$ python scripts/train/run_train.py --config scripts/config_omniglot.conf 
```

## Evaluating

To run evaluation on Omniglot

```sh
$ python scripts/eval/run_eval.py --config scripts/config_omniglot.conf
```

# Environment

## Quickstart

```sh
$ ./bin/start
```

## Setup and use docker

Build the docker image,

```sh
$ docker build --rm -f dockerfiles/tf-py3-jupyter.Dockerfile -t <name>:latest .
```

and now run the image

```sh
$ docker run --rm -u $(id -u):$(id -g) -p 6006:6006 -p 8888:8888 <name>:latest
```

Visit that link, hey look your jupyter notebooks are ready to be created.

If you want, you can attach a shell to the running container

```sh
$ docker exec -it <container-id> /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
```
