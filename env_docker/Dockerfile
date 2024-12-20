# Base image with CUDA 11.3.1 and Ubuntu 20.04
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

#-----------------------------
# Environment Variables
#-----------------------------
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# bash-based shell environment
SHELL ["/bin/bash", "-c"]

# XDG_RUNTIME_DIR setup
RUN mkdir -p /run/user/1000 && \
    chmod 0700 /run/user/1000
ENV XDG_RUNTIME_DIR=/run/user/1000

# Use Japanese server for apt (faster download speeds in Japan)
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

#-----------------------------
# Install common packages
#-----------------------------
RUN apt-get -y update && \
    apt-get -y install lsb-release software-properties-common git ssh \
    python-is-python3 vim nano wget curl make build-essential cmake \
    tmux libopencv-dev python3.9 python3-pip \
    xsel xclip

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip, setuptools, and wheel
RUN python -m pip install --upgrade pip setuptools wheel

#-----------------------------
# Install salesforce-lavis and dependencies
#-----------------------------
RUN git clone https://github.com/salesforce/LAVIS.git /LAVIS

# Install Open3D with compatible version
RUN python -m pip install open3d==0.18.0

# Install LAVIS dependencies
RUN sed -i 's/open3d==0.13.0/open3d==0.18.0/g' /LAVIS/requirements.txt
RUN cd /LAVIS && pip install .

# for avoiding error about numpy.core.multiarray failed to import
RUN pip uninstall opencv-python opencv-python-headless -y
RUN pip install \
    numpy==1.23.5 \
    moviepy==1.0.3 \
    opencv-python-headless==4.5.5.64 \
    huggingface-hub==0.25.0 \
    peft==0.13.0 \
    salesforce-lavis==1.0.1

#-----------------------------
# Setup pg-blip workspace
#-----------------------------
RUN mkdir /pg-blip
COPY ./pg-blip /pg-blip
RUN mkdir /models

#-----------------------------
# Download model
#-----------------------------
#RUN mkdir /models
#RUN cd /models && wget https://huggingface.co/bidiptas/PG-InstructBLIP/resolve/main/pgvlm_weights.bin

WORKDIR /pg-blip

#-----------------------------
# Tmux configuration
#-----------------------------
RUN git clone https://github.com/ry0hei-kobayashi/tmux-conf.git && mv tmux-conf/.tmux.conf ~/

#-----------------------------
# Finalization
#-----------------------------

