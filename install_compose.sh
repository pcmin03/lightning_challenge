#!/bin/sh

# Utility for installing Docker Compose on Linux systems.
# Visit https://docs.docker.com/compose/install for more information.
# This script is separate from the Makefile because downloads are slow in `make` commands.

COMPOSE_VERSION=v2.10.2
COMPOSE_OS_ARCH=linux-x86_64
COMPOSE_URL=https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-${COMPOSE_OS_ARCH}
COMPOSE_PATH=${HOME}/.docker/cli-plugins
COMPOSE_FILE=${COMPOSE_PATH}/docker-compose

if [ -s "${COMPOSE_FILE}" ]; then
    echo "${COMPOSE_FILE} already exists!";
else
    mkdir -p "${COMPOSE_PATH}"
    curl -SL "${COMPOSE_URL}"	-o "${COMPOSE_FILE}"
    chmod +x "${COMPOSE_FILE}";
fi
