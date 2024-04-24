#!/bin/bash
echo "Running main.py..."
VENV_PATH="/Users/julie_chung/miniforge3/envs/cs1430/lib/python3.9/venv"
cd $(dirname $VENV_PATH)
if [ $? -eq 0 ]; then
	echo "Changed directory to $(pwd)"
	source "$VENV_PATH/bin/activate"
	if [ $? -eq 0 ]; then
        	echo "Virtual environment activated."
	else
		echo "Failed to activate virtual environment."
		exit 1
	fi
else
	echo "Failed to change directory."
	exit 1
fi
python main.py --task 3
