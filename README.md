# Prototypical Networks for Few-shot Learning

Tensorflow v2 implementation of NIPS 2017 Paper _Prototypical Networks for Few-shot Learning_.

## Repository Structure
The repository organized as follows. 

- **data** directory contains scripts for dataset downloading and used as a default directory for datasets.

- **protonet** is the library containing the model itself (`protonet/models`) and logic for datasets loading and processing (`protonet/data`). 

- **scripts** directory contains scripts for launching the training. `train/run_train.py` and `eval/run_eval.py` launch training and evaluation respectively. tests folder contains basic training procedure on small-valued parameters to check general correctness. results folder contains .md file with current configuration and details of conducted experiments.

## Training

Training and evaluation configurations are specified through config files, each config describes single train+eval evnironment.

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/protonet --mode train --config <config>
```

`<config> = omniglot | ...`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/protonet --mode eval --config <config>
```

`<config> = omniglot | ...`

#### Results

In the `results/<ds>` directory you can find the following results of training processes on a specific dataset `<ds>`:

-  `results/<ds>/models/`, there are trained models.

-  `results/<ds>/results/`, there are debug output on different `.csv` files.

-  `results/<ds>/summaries/`, tensorboard summaries.

To run TensorBoard, use the following command 

```sh
$ tensorboard --logdir=./results/<ds>/summaries/
```

# Environment

## Quickstart

```sh
$ ./bin/start [-t <tag-name>] -[-sudo <bool>] [--use-official <bool>]
```

```
<tag-name> = cpu | devel-cpu | gpu | nightly-gpu-py3
<bool> = false | true
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

And then you can find the entire source code in `/develop`.

```sh
$ cd /develop
```

To run TensorBoard, use the following command (alternatively python -m tensorboard.main)

```sh
$ tensorboard --logdir=/path/to/summaries
```