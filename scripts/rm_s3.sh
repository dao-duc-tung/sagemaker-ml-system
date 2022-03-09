#!/bin/bash

BUCKET="$1"
REGION=$2

aws s3 rm "s3://$BUCKET" --region "$REGION" --recursive
aws s3 rb "s3://$BUCKET" --region "$REGION" --force
