#!/bin/sh

set -eu

INVOKE_CWD=$(pwd -P)
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

OWNER=${OWNER:-ZiweiZHU0513}
REPO=${REPO:-OR_Benchmark}
TAG=${TAG:-data-20260526}
ASSET_BASENAME=${ASSET_BASENAME:-or_benchmark_data_20260526.tar.gz}
EXTRACT_MODE=${EXTRACT_MODE:-fail}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT}

usage() {
    cat <<'EOF'
Usage:
  sh scripts/download_data_release.sh [options]

Options:
  --tag TAG                    Release tag (default: data-20260526)
  --asset_basename NAME        Archive basename (default: or_benchmark_data_20260526.tar.gz)
  --owner OWNER                GitHub owner (default: ZiweiZHU0513)
  --repo REPO                  GitHub repo (default: OR_Benchmark)
  --extract_mode MODE          fail|replace (default: fail)
    --output_dir DIR             Where to place archive and extracted data/ (default: repo root)
  -h, --help                   Show this help

Environment variables are also supported:
    OWNER, REPO, TAG, ASSET_BASENAME, EXTRACT_MODE, OUTPUT_DIR

Notes:
  1) This script downloads release assets only (not GitHub source-code archives).
  2) It verifies sha256 and tests archive integrity before extraction.
  3) Extraction is atomic: data is extracted to a temp dir first.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --tag)
            TAG=$2
            shift 2
            ;;
        --asset_basename)
            ASSET_BASENAME=$2
            shift 2
            ;;
        --owner)
            OWNER=$2
            shift 2
            ;;
        --repo)
            REPO=$2
            shift 2
            ;;
        --extract_mode)
            EXTRACT_MODE=$2
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$EXTRACT_MODE" in
    fail|replace)
        ;;
    *)
        echo "Invalid --extract_mode: $EXTRACT_MODE (expected: fail|replace)" >&2
        exit 1
        ;;
esac

case "$OUTPUT_DIR" in
    /*)
        ;;
    *)
        OUTPUT_DIR="$INVOKE_CWD/$OUTPUT_DIR"
        ;;
esac

mkdir -p "$OUTPUT_DIR"

ARCHIVE_URL="https://github.com/${OWNER}/${REPO}/releases/download/${TAG}/${ASSET_BASENAME}"
CHECKSUM_URL="${ARCHIVE_URL}.sha256"
ARCHIVE_PATH="${OUTPUT_DIR}/${ASSET_BASENAME}"
CHECKSUM_PATH="${ARCHIVE_PATH}.sha256"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/or_benchmark_data.XXXXXX")
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT HUP TERM

download_file() {
    url=$1
    out=$2
    tmp="${out}.part"

    curl --fail --location \
        --retry 6 --retry-delay 2 --retry-all-errors \
        --connect-timeout 20 --max-time 1800 \
        --output "$tmp" "$url"

    mv "$tmp" "$out"
}

echo "[1/5] Downloading archive: $ARCHIVE_URL"
download_file "$ARCHIVE_URL" "$ARCHIVE_PATH"

echo "[2/5] Downloading checksum: $CHECKSUM_URL"
download_file "$CHECKSUM_URL" "$CHECKSUM_PATH"

echo "[3/5] Verifying checksum"
(
    cd "$OUTPUT_DIR"
    shasum -a 256 -c "$(basename "$CHECKSUM_PATH")"
)

echo "[4/5] Validating archive format"
tar -tzf "$ARCHIVE_PATH" >/dev/null

echo "[5/5] Extracting archive atomically"
tar -xzf "$ARCHIVE_PATH" -C "$TMP_DIR"

if [ ! -d "$TMP_DIR/data" ]; then
    echo "Archive does not contain top-level data/ directory" >&2
    exit 1
fi

if [ -e "$OUTPUT_DIR/data" ]; then
    if [ "$EXTRACT_MODE" = "replace" ]; then
        rm -rf "$OUTPUT_DIR/data"
    else
        echo "Target $OUTPUT_DIR/data already exists. Use --extract_mode replace to overwrite." >&2
        exit 1
    fi
fi

mv "$TMP_DIR/data" "$OUTPUT_DIR/data"

echo "Done. data/ extracted to: $OUTPUT_DIR/data"
