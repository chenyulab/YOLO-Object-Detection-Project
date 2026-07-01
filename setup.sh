#!/bin/bash

# VENV_DIR="venv"

# # Check if the virtual environment directory exists
# if [ ! -d "$VENV_DIR" ]; then
#     echo "Creating virtual environment..."
#     python -m venv $VENV_DIR
# fi

# # Activate the virtual environment
# source $VENV_DIR/bin/activate

# # Install dependencies from requirements.txt
# if [ -f "requirements.txt" ]; then
#     echo "Installing dependencies..."
#     pip install -r requirements.txt
# else
#     echo "No requirements.txt file found."
# fi

# # Deactivate the virtual environment
# deactivate

#!/bin/bash

# Check if venv directory exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment based on the operating system
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu" ]]; then
    source venv/bin/activate
    echo "venv activated!"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv\\Scripts\\activate
    echo "venv activated!"
else
    echo "Unsupported operating system. Please activate the virtual environment manually."
    exit 1
fi

python -m pip install --upgrade pip
# # # Install requirements
pip install -r requirements.txt

# Deactivate virtual environment
deactivate