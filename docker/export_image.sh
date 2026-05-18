#!/usr/bin/env bash
set -euo pipefail

IMAGE_REF="${1:-im-nmpo:noetic}"
OUTPUT_DIR="${2:-docker_release}"
ARCHIVE_BASENAME="${3:-im-nmpo_noetic}"

mkdir -p "${OUTPUT_DIR}"

if ! docker image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
  echo "Image '${IMAGE_REF}' does not exist locally. Build it first, for example:"
  echo "  docker build -t ${IMAGE_REF} -f docker/Dockerfile ."
  exit 1
fi

STAMP="$(date +%Y%m%d)"
ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_BASENAME}_${STAMP}.tar.gz"
CHECKSUM_PATH="${ARCHIVE_PATH}.sha256"

echo "Exporting ${IMAGE_REF} -> ${ARCHIVE_PATH}"
docker save "${IMAGE_REF}" | gzip -1 > "${ARCHIVE_PATH}"

echo "Writing checksum -> ${CHECKSUM_PATH}"
(
  cd "${OUTPUT_DIR}"
  sha256sum "$(basename "${ARCHIVE_PATH}")" > "$(basename "${CHECKSUM_PATH}")"
)

echo
echo "Done."
echo "Archive:  ${ARCHIVE_PATH}"
echo "Checksum: ${CHECKSUM_PATH}"
echo
echo "Loading command:"
echo "  sha256sum -c $(basename "${CHECKSUM_PATH}")"
echo "  docker load -i $(basename "${ARCHIVE_PATH}")"
