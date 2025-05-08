# CUDA Bioinformatics Tools

A collection of commandline tools for common bioinformatics analyses that utilizes GPU-computing. This exists because I wanted to practice CUDA and linear algebra for fun.

### cuda_PCA
```
Description: cuda_PCA is a tool for Principal Component Analysis utilizing GPU computing
             through CUDA and its libraries. Current calculations are not batched, so
             GPUs with smaller memory capacities will fail to allocate on larger datasets.
             Resulting PC matrix is printed to standard out.

Usage: ./cuda_PCA [ -h help ] filename > output.txt

Options:
  -h                Displays help message
```

### cuda_lmFit
```
Description: cuda_lmFit is a tool for fitting linear models utilizing GPU computing.
             Currently implements the (standard or weighted) least squares 
             algorithm. Does not filter or perform normalization. 
             Future implementations will fit every gene in a table 
             Uses formula Y = Xb. Results are printed to standard out.

Usage: ./cuda_lmFit [ OPTIONS ] Y_file X_file 

Options:
  -h                Displays help message.
  -w weights_file   File containings wieghts. If not specified, least sqares is unweighted.
```
