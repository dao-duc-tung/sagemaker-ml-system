ARG DEBIAN_FRONTEND=noninteractive
FROM python:3.8-slim

MAINTAINER Tung Dao <tungdao17@gmail.com>

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    wget \
    nginx \
    ca-certificates \
    apt-utils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get -y install curl
RUN apt-get -y install libgomp1

COPY code/requirements.txt /opt/program/requirements.txt
RUN pip install -r /opt/program/requirements.txt

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

RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve
