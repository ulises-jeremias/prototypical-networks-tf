ARG DOCKER_ENV=cpu

FROM ulisesjeremias/tf-docker:${DOCKER_ENV}-jupyter

ADD . /develop

RUN apt-get update -q
RUN apt-get install -y git wget nano

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 --version

WORKDIR /develop
RUN bin/download_omniglot -d /develop/data/omniglot/data
RUN python3 setup.py install

RUN chmod -R a+rwx /develop

WORKDIR /develop
