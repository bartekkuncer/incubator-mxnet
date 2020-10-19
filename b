#!/bin/bash
rm -rf build
mkdir build
cd build
cmake -GNinja -DUSE_BLAS=mkl -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)

cd ../python
pip install -e .
