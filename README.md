# CUDA Bioinformatics Tools

## IN DEVELOPMENT (kind of just for fun)

A collection of commandline tools for common bioinformatics analyses that utilizes GPU-computing.


### cuda_PCA

```
Description: cuda_PCA is a tool used for Principal Component Analysis utilizing GPU 
             computing through CUDA and its libraries. Current calculations are not batched,
             so GPUs with smaller memory capacities will fail on larger datasets.
             Resulting PC matrix is printed to standard out.

Usage: ./cuda_PCA [ -h help ] filename > output.txt

Options:
  -h                Displays help message
```

### cuda_lmFit
```
Description: cuda_lmFit is a tool used for fitting linear models utilizing GPU
             computing. Currently implements, the least squares (standard or weighted) 
             algorithm. Does not filter or perform normalization. 
             Uses formula Y = Xb. Results are printed to standard out.

Usage: ./cuda_lmFit [ OPTIONS ] Y_file X_file  > output.txt

Options:
  -h                Displays help message.
  -w weights_file   File containings wieghts. If not specified, least sqares is unweighted.
```
