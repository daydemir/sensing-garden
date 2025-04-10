#!/bin/bash

# Path to the Python script you want to run
TL_PYTHON_SCRIPT="./time-lapse/timelapse_video.py"

# Log file for output and errors
TIME_LAPSE_LOG_FILE="./log-files/timelapse.log"

# Source the environment
source "./setup_env.sh"


# Run the Python script with parsed arguments and log output/errors
python3 "$TL_PYTHON_SCRIPT" >> "$TIME_LAPSE_LOG_FILE" 2>&1

#sleep 540 # does not work in shells cript - ADD IN PYTHON SCRIPT
