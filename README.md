# CUDA Bioinformatics Tools

## IN DEVELOPMENT (kind of just for fun)

A collection of commandline tools for common bioinfomatics analyses that utilizes GPU-computing.


### cuda_PCA

```
Description: cuda_PCA is a tool used for Principal Component Analysis utilizing GPU 
             computing through CUDA and its libraries. Current calculations are not batched,
             so GPUs with smaller memory capacities will fail on larger datasets.
             Resulting PC matrix is printed to standard out.

Usage: ./cuda_PCA -f filename > output.txt

Options:
  -f filename       Specifies input file
  -h                Displays help message
```

