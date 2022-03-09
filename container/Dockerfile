# Build an image that can do training and inference in SageMaker

FROM ubuntu:18.04

MAINTAINER Tung Dao <tungdao17@gmail.com>

RUN apt-get update
RUN apt-get install -y python3.8 python3.8-dev python3-pip python3.8-venv python3-setuptools
RUN python3.8 -m pip install pip --upgrade

RUN apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y apt-utils 2>&1

RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN ln -s /usr/bin/pip3.8 /usr/bin/pip

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY code /opt/program
WORKDIR /opt/program

RUN pip install -r requirements.txt

RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve
