#!/bin/bash
echo "Running main.py..."
CONDA_PATH="/Users/julie_chung/miniforge3/bin/conda"
source "${CONDA_PATH}/../etc/profile.d/conda.sh"
conda env list
conda activate cs1430
if [ $? -eq 0 ]; then
	echo "Conda environment activated."
else
	echo "Failed to activate Conda environment."
	exit 1
fi
python main.py --task 3
