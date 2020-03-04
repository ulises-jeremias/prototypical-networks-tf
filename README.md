# Prototypical Networks for Few-shot Learning

<img width="896" alt="Screenshot 2019-04-02 at 9 53 06 AM" src="https://user-images.githubusercontent.com/23639048/55438102-5d9e4c00-55a9-11e9-86e2-b4f79f880b83.png">

Implementation of Prototypical Networks for Few-shot Learning paper (https://arxiv.org/abs/1703.05175) in TensorFlow 2.0. Model has been tested on Omniglot and MiniImagenet datasets with the same splitting as in the paper.

<details><summary>Dependencies and Installation</summary>

* The code has been tested on Ubuntu 18.04 with Python 3.6.8 and TensorFflow 2.0.0-alpha0
* The two main dependencies are TensorFlow and Pillow package (Pillow is included in dependencies)
* To install `protonet` lib run `python setup.py install`
* Run `./bin/download_omniglot.sh` from repo's root directory to download Omniglot dataset
* MiniImagenet was downloaded from brilliant [repo from `renmengye`](https://github.com/renmengye/few-shot-ssl-public) and placed into `data/mini-imagenet` folder
</details>

## Repository Structure
The repository organized as follows. 

- **data** directory contains scripts for dataset downloading and used as a default directory for datasets.

- **protonet** is the library containing the model itself (`protonet/models`) and logic for datasets loading and processing (`protonet/dataset`). 

- **scripts** directory contains scripts for launching the training. `train/run_train.py` and `eval/run_eval.py` launch training and evaluation respectively. tests folder contains basic training procedure on small-valued parameters to check general correctness. results folder contains .md file with current configuration and details of conducted experiments.

## Training

Training and evaluation configurations are specified through config files, each config describes single train+eval evnironment.

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/protonet --mode train --config <config>
```

`<config> = omniglot | mini-imagenet`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/protonet --mode eval --config <config>
```

`<config> = omniglot | mini-imagenet`

#### Results

In the `results/<ds>` directory you can find the following results of training processes on a specific dataset `<ds>`:

In the `/results` directory you can find the results of a training processes using a `<model>` on a specific `<dataset>`:

```
.
├─ . . .
├─ results
│  ├─ <dataset>                            # results for an specific dataset.
│  │  ├─ <model>                           # results training a <model> on a <dataset>.
│  │  │  ├─ models                         # ".h5" files for trained models.
│  │  │  ├─ results                        # ".csv" files with the different metrics for each training period.
│  │  │  ├─ summaries                      # tensorboard summaries.
│  │  │  ├─ config                         # optional configuration files.
│  │  └─ └─ <dataset>_<model>_results.csv  # ".csv" file in which the relationships between configurations, models, results and summaries are listed by date.
│  └─ summary.csv                          # contains the summary of all the training
└─ . . .
```

where

```
<dataset> = omniglot | mini-imagenet
<model> = protonet
```

To run TensorBoard, use the following command 

```sh
$ tensorboard --logdir=./results/<ds>/summaries/
```

# Environment

## Quickstart

```sh
$ ./bin/start [-t <tag-name>] [--sudo] [--build]
```

```
<tag-name> = cpu | devel-cpu | gpu | nightly-gpu-py3
```

<details><summary>or setup and use docker on your own</summary>

Build the docker image,

```sh
$ docker build --rm -f dockerfiles/tf-py3-jupyter.Dockerfile -t <name>:latest .
```

and now run the image

```sh
$ docker run --rm -u $(id -u):$(id -g) -p 6006:6006 -p 8888:8888 <name>:latest
```

</details>

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
