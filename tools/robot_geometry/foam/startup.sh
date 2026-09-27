#!/bin/bash

# Build the project
cmake -Bbuild -GNinja .
cmake --build build/
pip install -e .

# Execute the command passed to docker run
exec "$@"