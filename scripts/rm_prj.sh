#!/bin/bash

NAME="$1"

aws sagemaker delete-project --project-name "$NAME"
