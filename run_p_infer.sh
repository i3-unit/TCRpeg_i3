#!/bin/bash

# Run tcrpeg infer on a folder

# Display help message from the Python script
# python i3_scripts/tcrpeg_p_infer.py -h

# Extract current directory
script_dir=$(dirname "$0")
echo "Running script from $script_dir"

# Set default values for command-line options
input_dir=""
output_dir=""
device="cpu"
seq_col="sequence"
count_col="count"
id_col="id"
word2vec_epochs=10
epochs=20
word2vec_batch_size=32
word2vec_learning_rate=1e-4
hidden_size=128
num_layers=5
batch_size=32
learning_rate=1e-4
test_size=0.4

# Parse command-line options
while getopts 'i:o:d:s:c:I:w:e:b:l:H:n:B:L:t:h' flag; do
  case "${flag}" in
    i) input_dir="${OPTARG}" ;;
    o) output_dir="${OPTARG}" ;;
    d) device="${OPTARG}" ;;
    s) seq_col="${OPTARG}" ;;
    c) count_col="${OPTARG}" ;;
    I) id_col="${OPTARG}" ;;
    w) word2vec_epochs="${OPTARG}" ;;
    e) epochs="${OPTARG}" ;;
    b) word2vec_batch_size="${OPTARG}" ;;
    l) word2vec_learning_rate="${OPTARG}" ;;
    H) hidden_size="${OPTARG}" ;;
    n) num_layers="${OPTARG}" ;;
    B) batch_size="${OPTARG}" ;;
    L) learning_rate="${OPTARG}" ;;
    t) test_size="${OPTARG}" ;;
    h) echo "Usage: $0 -i input_dir -o output_dir -d device -s seq_col -c count_col -I id_col -w word2vec_epochs -e epochs -b word2vec_batch_size -l word2vec_learning_rate -H hidden_size -n num_layers -B batch_size -L learning_rate -t test_size" && exit 1 ;;
    *) echo "Unexpected option ${flag}" && exit 1 ;;
  esac
done

# Check if input_dir and output_dir are provided
if [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
  echo "Usage: $0 -i input_dir -o output_dir -d device -s seq_col -c count_col -I id_col -w word2vec_epochs -e epochs -b word2vec_batch_size -l word2vec_learning_rate -H hidden_size -n num_layers -B batch_size -L learning_rate -t test_size"
  exit 1
fi

mkdir -p "$output_dir"
mkdir -p "$output_dir"/logs

# Loop through all files in the input directory
for file in "$input_dir"/*.csv; do
  if [ -f "$file" ]; then
    echo "Processing $file"
    # Run tcrpeg infer on the file
    sample_name=$(basename "$file" .csv)
    python3 ${script_dir}/tcrpeg_toolkit/p_infer_calculation.py \
      --input "$file" \
      --output "$output_dir" \
      --device "${device:-cpu}" \
      --seq_col "${seq_col:-sequence}" \
      --count_col "${count_col:-count}" \
      --id "${id_col:-id}" \
      --word2vec_epochs "${word2vec_epochs:-10}" \
      --epochs "${epochs:-20}" \
      --word2vec_batch_size "${word2vec_batch_size:-100}" \
      --word2vec_learning_rate "${word2vec_learning_rate:-1e-4}" \
      --hidden_size "${hidden_size:-128}" \
      --num_layers "${num_layers:-5}" \
      --batch_size "${batch_size:-100}" \
      --learning_rate "${learning_rate:-1e-4}" \
      --test_size "${test_size:-0.2}" \
      --log "$output_dir/logs/${sample_name}_tcrpeg_p_infer.log"
  fi
done
