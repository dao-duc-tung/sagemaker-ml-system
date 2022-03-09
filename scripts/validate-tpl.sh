#!/bin/bash
set -euxo pipefail

NOW=$(date +"%x %r %Z")
echo "Time: $NOW"

PREFIX="smsd" # should match the ProjectPrefix parameter in pipeline.yml and studio.yml additional ARN privileges
BUCKET="s3-bucket"

rm -rf build
mkdir build
rsync -av --progress . build \
    --exclude build \
    --exclude "*.git*" \
    --exclude .pre-commit-config.yaml
cd build
# binding resources of pipeline.yml and studio.yml together with common PREFIX
sed -i -e "s/PROJECT_PREFIX/$PREFIX/g" assets/*.yml pipeline.yml
sed -i -e "s/S3_BUCKET_NAME/$BUCKET/g" pipeline.yml
find . -type f -iname "*.yml-e" -delete

bash scripts/lint.sh || exit 1

aws cloudformation validate-template --template-body file://studio.yml
aws cloudformation validate-template --template-body file://pipeline.yml
