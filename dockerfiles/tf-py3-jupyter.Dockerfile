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

RUN git clone --branch=develop https://github.com/midusi/handshape_datasets.git /develop/lib/handshape_datasets
RUN pip3 install -e /develop/lib/handshape_datasets

RUN mkdir -p /.handshape_datasets
RUN chmod -R a+rwx /.handshape_datasets

RUN chmod -R a+rwx /develop

WORKDIR /develop
