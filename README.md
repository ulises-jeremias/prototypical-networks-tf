# Prototypical Networks for Few-shot Learning

Tensorflow v2 implementation of NIPS 2017 Paper _Prototypical Networks for Few-shot Learning_.

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
