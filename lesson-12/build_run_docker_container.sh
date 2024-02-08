#!/bin/zsh

docker build -t body-segmentation:1.2 --progress=plain .
docker tag body-segmentation:1.2 \
europe-west10.pkg.dev/ai-for-beginners/ai-for-beginners-docker-repo/body-segmentation:1.2
docker run -it --rm \
-e GCLOUD_PROJECT=ai-for-beginners \
-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/application_default_credentials.json \
-v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/keys/application_default_credentials.json:ro \
body-segmentation:1.2 shell_scripts/train_test.sh body-segmentation-bucket