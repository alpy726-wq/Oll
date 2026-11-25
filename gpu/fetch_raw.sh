#!/bin/sh
# Download the real-world sources used by data/build_real.py (run on the GPU provider; same URLs as the sandbox).
set -e
cd "$(dirname "$0")/../data"
mkdir -p real text_dl
cd real
curl -sSfL -o ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
curl -sSfL -o ETTh2.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv
curl -sSfL -o ETTm1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv
curl -sSfL -o jena.zip https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip
python3 -c "import zipfile; zipfile.ZipFile('jena.zip').extractall('.')"
curl -sSfL -o sp500.csv https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv
curl -sSfL -o vix.csv https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv
curl -sSfL -o country-codes.csv https://raw.githubusercontent.com/datasets/country-codes/main/data/country-codes.csv
[ -d fsdd ] || git clone -q --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset.git fsdd
[ -d arc ] || git clone -q --depth 1 https://github.com/fchollet/ARC-AGI.git arc
cd ../text_dl
for c in gutenberg brown; do curl -sSfL -o $c.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/$c.zip; done
