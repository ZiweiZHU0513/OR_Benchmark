#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

DATA_DIR=${1:-data}
OUT_DIR=${2:-release_assets}
STAMP=${3:-$(date +%Y%m%d)}
ARCHIVE_BASENAME="or_benchmark_data_${STAMP}"
ARCHIVE_PATH="$OUT_DIR/${ARCHIVE_BASENAME}.tar.gz"
CHECKSUM_PATH="$ARCHIVE_PATH.sha256"

if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory not found: $DATA_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Packing $DATA_DIR -> $ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" "$DATA_DIR"

echo "Writing checksum -> $CHECKSUM_PATH"
(
    cd "$OUT_DIR"
    shasum -a 256 "${ARCHIVE_BASENAME}.tar.gz" > "${ARCHIVE_BASENAME}.tar.gz.sha256"
)

echo
echo "Created release assets:"
ls -lh "$ARCHIVE_PATH" "$CHECKSUM_PATH"
echo
echo "Next steps:"
echo "1. Go to your GitHub repository -> Releases -> Draft a new release"
echo "2. Upload both files above as release assets"
echo "3. In the release notes, mention that users should extract the archive at the repository root"