#!/bin/bash

# This is a script to build the project, It shows what install paths I used for the libraries and the project itself.
# To build the project, the following dependencies are required:
# 1. CMake
# 2. A C++ compiler
# 3. MATLAB
# 4. CUDA
# 5. BLAS and LAPACK
#
# A test cpp executable can be built, but it needs opencv io capability to run.
# [OPTIONAL] 6. OpenCV
#

HOME_DIR=$(pwd)

# Move into the cpp directory
cd cpp

# Call cmake to build the project, put the build files in cpp/build, and install the project in the home directory
cmake -B build -DCMAKE_INSTALL_PREFIX=../

# Move into the build directory
cd build

# Build and install the project
make install

# Move back to the parent directory
cd $HOME_DIR

# Call cmake to build the mex file, put the build in $HOME_DIR/mex.
cmake -B mex

# Move into the build directory
cd mex

# Build the mex file
make
# Now, it is up to the user to move the mex file to the desired location, or to add the directory to the MATLAB path.

# Move back to the parent directory
cd $HOME_DIR


