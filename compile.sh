#!/bin/bash

nvcc -I lib/ -L /usr/local/cuda \
     -lcublas -lcusolver -std=c++11 \
     -o cuda_PCA src/main.cpp