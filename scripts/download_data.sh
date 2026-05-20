#!/bin/bash
# Download NQ / DPR data for the Koga pipeline
set -euo pipefail

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "Downloading NQ biencoder dev data..."
NQ_URL="https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz"
NQ_FILE="$DATA_DIR/biencoder-nq-dev.json"

if [ ! -f "$NQ_FILE" ]; then
    wget -O "$NQ_FILE.gz" "$NQ_URL"
    gunzip "$NQ_FILE.gz"
    echo "Downloaded and extracted: $NQ_FILE"
else
    echo "NQ data already exists: $NQ_FILE"
fi

echo "Data download complete."
echo "Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
