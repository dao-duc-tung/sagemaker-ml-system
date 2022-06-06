#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1
dockerfile=$2

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name> <dockerfile>"
    exit 1
fi

if [ "$dockerfile" == "" ]
then
    echo "Usage: $0 <image-name> <dockerfile>"
    exit 1
fi

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -t ${image} -f ${dockerfile} .
