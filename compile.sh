#!/bin/bash

nvcc -G -g \
     -I lib/ -L /usr/local/cuda \
     -lcublas -lcusolver -std=c++11 \
     -O2 -larmadillo \
     -o cuda_PCA src/cuda_PCA.cpp

nvcc -G -g \
     -I lib/ -L /usr/local/cuda \
     -lcublas -lcusolver -std=c++11 \
     -O2 -larmadillo \
     -o cuda_lmFit src/cuda_lmFit.cpp
