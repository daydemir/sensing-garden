#!/bin/bash

# Set the resource directory
RESOURCE_DIR="./resources"
mkdir -p "$RESOURCE_DIR"

# Define download function with file existence check and retries
download_model() {
  local url=$1
  local file_name=$(basename "$url")

  # Check if the file is for H8L and rename it accordingly
  if [[ "$url" == *"hailo8l"* && "$url" != *"_h8l.hef" ]]; then
    file_name="${file_name%.hef}_h8l.hef"
  fi

  local file_path="$RESOURCE_DIR/$file_name"

  if [ ! -f "$file_path" ]; then
    echo "Downloading $file_name..."
    wget -q --show-progress "$url" -O "$file_path" || {
      echo "Failed to download $file_name after multiple attempts."
      exit 1
    }
  else
    echo "File $file_name already exists in $RESOURCE_DIR. Skipping download."
  fi
}

# Define all URLs in arrays
H8_HEFS=(
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov8m.hef"
)

# other H8 HEFS available
# "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov6n.hef"
# "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/scdepthv3.hef"
# "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov8m_pose.hef"
# "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov5m_seg.hef"

H8L_HEFS=(
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov8s.hef"
)

# other H8L HEFS available: 
# "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov8s_pose.hef"
# "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov5n_seg.hef"
# "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov6n.hef"
# "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/scdepthv3.hef"

# If --all flag is provided, download everything in parallel
if [ "$1" == "--all" ]; then
  echo "Downloading all models and video resources..."
  for url in "${H8_HEFS[@]}" "${H8L_HEFS[@]}" "${VIDEOS[@]}"; do
    download_model "$url" &
  done
else
  if [ "$DEVICE_ARCHITECTURE" == "HAILO8L" ]; then
    echo "Downloading HAILO8L models..."
    for url in "${H8L_HEFS[@]}"; do
      download_model "$url" &
    done
  elif [ "$DEVICE_ARCHITECTURE" == "HAILO8" ]; then
    echo "Downloading HAILO8 models..."
    for url in "${H8_HEFS[@]}"; do
      download_model "$url" &
    done
  fi
fi


# Wait for all background downloads to complete
wait

echo "All downloads completed successfully!"
