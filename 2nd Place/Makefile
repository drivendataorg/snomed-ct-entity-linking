.PHONY: docker docker-build docker-run

IMAGE_NAME := snomedct
IMAGE_TAG := dev

docker-build:
	docker build \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f docker/Dockerfile \
		.

docker-run:
	docker run --rm -it \
		--shm-size=26Gb \
		--gpus all \
		$(IMAGE_NAME):$(IMAGE_TAG)

docker: docker-build docker-run
