#!/usr/bin/env bash
set -euo pipefail

LOCAL_IMAGE="${1:-im-nmpo:noetic}"
REMOTE_IMAGE="${2:-}"

if [ -z "${REMOTE_IMAGE}" ]; then
  echo "Usage:"
  echo "  ./docker/publish_image.sh im-nmpo:noetic ghcr.io/<OWNER>/im-nmpo:noetic"
  echo "  ./docker/publish_image.sh im-nmpo:noetic docker.io/<USER>/im-nmpo:noetic"
  echo
  echo "Login first with 'docker login ghcr.io' or 'docker login docker.io'."
  exit 1
fi

if ! docker image inspect "${LOCAL_IMAGE}" >/dev/null 2>&1; then
  echo "Image '${LOCAL_IMAGE}' does not exist locally. Build it first, for example:"
  echo "  docker build -t ${LOCAL_IMAGE} -f docker/Dockerfile ."
  exit 1
fi

docker tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"

echo
echo "Published image:"
echo "  ${REMOTE_IMAGE}"
echo
echo "Pull command:"
echo "  docker pull ${REMOTE_IMAGE}"
