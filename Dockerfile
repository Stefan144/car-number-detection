FROM  ubuntu:18.04

ENV LANG C.UTF-8

RUN apt-get update && yes|apt-get upgrade &&\
    apt-get -y install build-essential yasm nasm cmake unzip git wget curl \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool flex bison \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender1 libssl-dev libx264-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip3 install albumentations==0.4.3
RUN pip3 install opencv-python==4.2.0.32
RUN pip3 install torch==1.4.0
RUN pip3 install trafaret==2.0.2
RUN pip3 install pytorch_argus==0.0.9
RUN pip3 install tabulate==0.8.6
RUN pip3 install deep-pipe==0.0.7.post1
RUN pip3 install numpy==1.18.1
RUN pip3 install torchvision==0.5.0
RUN pip3 install trafaret-config==2.0.2
RUN pip3 install pytorch-argus==0.0.9
RUN pip3 install python-Levenshtein==0.12.0
RUN pip3 install scikit-learn==0.22.1


ENV PYTHONPATH $PYTHONPATH:/workdir/src


COPY . /workdir

WORKDIR /workdir
