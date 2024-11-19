# CUDA Bioinformatics Tools

## IN DEVELOPMENT (kind of just for fun)

A collection of commandline tools for common bioinformatics analyses that utilizes GPU-computing.


### cuda_PCA

```
Description: cuda_PCA is a tool used for Principal Component Analysis computed on the GPU
	     through CUDA and its libraries. Current calculations are not batched,
             so GPUs with smaller memory capacities will fail on larger datasets.
             Additionally, singular value decomposition is currently computed on the GPU
	     so it is slowwwww. Resulting PC matrix is printed to standard out.

Usage: ./cuda_PCA [ -h help ] filename > output.txt

Options:
  -h                Displays help message
```

