/////////////////////////////////////////////////////////////////
/*
	cuda_PCA
*/
/////////////////////////////////////////////////////////////////

// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// cuBLAS and cuSolver
#include "cublas_v2.h"
#include "cusolverDn.h"

// Armadillo
//#include <armadillo>

// utils
#include "tsv.h"
#include "utils.h"

// SL
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept> 
#include <cmath>
#include <cassert>
#include <getopt.h>

/////////////////////////////////////////////////////////////////
void display_help(const char* program) {
   std::cerr << "Description: cuda_PCA is a tool used for Principal Component Analysis utilizing GPU \n             computing through CUDA and its libraries. Current calculations are not batched,\n             so GPUs with smaller memory capacities will fail on larger datasets.\n             Resulting PC matrix is printed to standard out.\n\n"
             << "Usage: " << program << " [ -h help ] filename > output.txt\n\n"
             << "Options:\n"
             << "  -h                Displays help message\n"
             << std::endl;
}

void argparse(int argc, char** argv, std::string &filename) {
   int opt;

   while ((opt = getopt(argc, argv, "h")) != -1) {
      switch (opt) {
         case 'h':
            display_help(argv[0]);
            exit(EXIT_SUCCESS);
         default:
            display_help(argv[0]);
            exit(EXIT_FAILURE);
      }
   }

   if (optind >= argc) {
      display_help(argv[0]);
      exit(EXIT_FAILURE);
   }

   filename = argv[optind];
}

void cuda_assert(cudaError_t status) {
   if (status != cudaSuccess) {
      std::cerr << "\n// CUDA_ERROR: "
                << cudaGetErrorString(status) 
                << ".\n";
   exit(EXIT_FAILURE);
   }
}


/////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {

	std::cerr << "// cuda_PCA\n";

   // Arguments
   std::string filename;

   argparse(argc, argv, filename);

   if (filename.empty()) {
      display_help(argv[0]);
      std::cerr << "// ERROR: must include an input file.\n";
      exit(EXIT_FAILURE);
   }

   std::cerr << "// Parsing file..............................";

   /////////////////////////////////////////////////////////////////
   // Read in CSV
   TSV<double> table(filename);
   table.read_delim('\t');
   double *mat = table.flatten('R'); // write transposed so end coords are for samples
 
   std::cerr << "COMPLETE\n";  

   // Get Dimensions and bytes
   //   column major order on the transposed matrix = the same matrix lol  
   const unsigned long long int m = table.cols;
   const unsigned long long int n = table.rows;
   const unsigned long long int lda = m;
   const unsigned long long int size = table.size;
   const unsigned long long int cov_size = n * n;
   const size_t bytes = size * sizeof(double);
   const size_t norm_vec_bytes = n * sizeof(double);
   const size_t one_vec_bytes = m * sizeof(double);    
   const size_t cov_bytes = cov_size * sizeof(double);


   // Print Table Stats
   // std::cerr << "    Rows: " << m << "\n"; 
   // std::cerr << "    Cols: " << n << "\n";  
   // std::cerr << "    Size: " << size << "\n"; 
   // std::cerr << "    Bytes: " << bytes << "\n";  
   // std::cerr << "    Covariance Size: " << cov_size << "\n"; 
   // std::cerr << "    Covariance Bytes: " << cov_bytes << "\n";  

   // Create normalization vector, and one vectors
   double *norm_vec = get_norm_vec(mat, m, n);
   double *one_vec = get_one_vec(m);

   // Create pointer to GPU
   double *d_norm, *d_one;
   double *d_mat, *norm_d_mat, *cent_d_mat, *cov_d_mat;


   ///////////////////////////////////////////////////////////////// 
   // Create and initialize cuBLAS handle object
   cublasHandle_t handle;
   cublasCreate_v2(&handle);
   cublasStatus_t cublas_status;
   cudaError_t cuda_error;

   // Allocate GPU memory for vectors
   std::cerr << "// Allocating GPU memory.....................";  
   cuda_error = cudaMalloc((void**)&d_norm, norm_vec_bytes); cuda_assert(cuda_error);   // Normalization vector
   cuda_error = cudaMalloc((void**)&d_one, one_vec_bytes); cuda_assert(cuda_error);     // One Vector 
   cuda_error = cudaMalloc((void**)&d_mat, bytes); cuda_assert(cuda_error);             // Data Matrix
   cuda_error = cudaMalloc((void**)&norm_d_mat, bytes); cuda_assert(cuda_error);        // Normalization Matrix
   cuda_error = cudaMalloc((void**)&cent_d_mat, bytes); cuda_assert(cuda_error);        // Mean-Centered Matrix
   cuda_error = cudaMalloc((void**)&cov_d_mat, cov_bytes); cuda_assert(cuda_error);     // Covariance Matrix
   std::cerr << "COMPLETE\n";  

   ///////////////////////////////////////////////////////////////// 
   std::cerr << "// Calculating Covariance Matrix.............";  
   // Set Matrix and vectors (I am treating vector as matrices, deal it)
   cublas_status = cublasSetMatrix(1, 1, bytes, mat, 1, d_mat, 1);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n// ERROR: Could not set data matrix.\n");}

   cublas_status = cublasSetMatrix(1, 1, norm_vec_bytes, norm_vec, 1, d_norm, 1); 
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n// ERROR: Could not set normalization vector.\n");}

   cublas_status = cublasSetMatrix(1, 1, one_vec_bytes, one_vec, 1, d_one, 1);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n// ERROR: Could not generate one vector.\n");}

   ///////////////////////////////////////////////////////////////// 
   // Normalized matrix 
   //     - https://stackoverflow.com/questions/45307686/non-square-c-order-matrices-in-cublas-numba
   double alpha = 1.0f;
   double beta = 0.0f;
   cublas_status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            	 m, n, 1, 
                            	 &alpha, d_one, m,
                            	 d_norm, 1, &beta,
                           	 norm_d_mat, m);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n// ERROR: Could not generate normalization matrix.\n");}

   ///////////////////////////////////////////////////////////////// 
   // Normal-Center matrix
   alpha = 1.0f;
   beta = -1.0f;
   cublas_status = cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               m, n, 
                               &alpha, d_mat, lda,
                               &beta, norm_d_mat, lda,
                               cent_d_mat, m);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n// ERROR: Could not center matrix.\n");}

   ///////////////////////////////////////////////////////////////// 
   // Get Coveriance matrix
   alpha = 1.0f;
   beta = 0.0f;
   cublas_status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               n, n, m, 
                               &alpha, cent_d_mat, lda,
                               cent_d_mat, m, &beta,
                               cov_d_mat, n);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n// ERROR: Could not calculate covariance matrix.\n");}

   // Scale Centered Matrix and Coveriance Matrix by sqrt(N)
   alpha = 1 / std::sqrt(cov_size);
   cublasDscal(handle, cov_size, &alpha, cov_d_mat, 1);
   std::cerr << "COMPLETE\n";  

   //////////////////////////////////////////////////////////////////////////////////
   // Free up memory
   cudaDeviceSynchronize();
   cudaFree(d_mat);
   cudaFree(norm_d_mat);
   cudaFree(d_norm);
   cudaFree(d_one);

   //////////////////////////////////////////////////////////////////////////////////
   // Singular Value Decomposition
   //    This implementation is extremely slow on the GPU
   // https://docs.nvidia.com/cuda/cusolver/index.html#dense-eigenvalue-solver-reference-legacy
   // https://docs.nvidia.com/cuda/archive/9.1/cusolver/index.html#svd-example1

   // Create cuSolver Handle
   cusolverDnHandle_t cusolver_handle;
   cusolverDnCreate(&cusolver_handle);
   cusolverStatus_t cusolver_status;

   // vector for eigenvalues
   std::vector<double> W(n, 0);

   // Create device pointers
   double *d_A = cov_d_mat;
   double *d_W, *d_t_mat;
   int *devInfo;

   // Allocate memory on GPU
   cuda_error = cudaMallocManaged((void**)&d_t_mat, size * sizeof(double)); cuda_assert(cuda_error); 
   cuda_error = cudaMalloc((void**)&d_W, sizeof(double) * n);

   ///////////////////////////////////////////////////////////////////////////////////
   std::cerr << "// Performing Eigenvalue Decomposition.......";  


   // Create Buffer for Eigenvalues
   int lwork = 0;
   double* d_work;
   // std::vector<double> h_W(n);
   cusolver_status = cusolverDnDsyevd_bufferSize(cusolver_handle, 
                                                 CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 
                                                 n, d_A, n, d_W, &lwork);
   if (cusolver_status != CUSOLVER_STATUS_SUCCESS) { throw std::runtime_error("\n// ERROR: Could not create buffer for Eigenvalue Decomposition.\n");}


   cuda_error = cudaMalloc(&d_work , sizeof(double) * lwork); cuda_assert(cuda_error);
   cuda_error = cudaMalloc((void**)&devInfo, sizeof(int)); cuda_assert(cuda_error); 


   // Compute Eigenvalues
   cusolver_status = cusolverDnDsyevd(cusolver_handle, 
                                      CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                      n, d_A, n, d_W, 
                                      d_work, lwork, devInfo);

   if (cusolver_status != CUSOLVER_STATUS_SUCCESS) { throw std::runtime_error("\n// ERROR: Could not perform Eigenvalue Decomposition.\n");}
   std::cerr << "COMPLETE\n";  


   ///////////////////////////////////////////////////////////////// 
   std::cerr << "// Transforming Data.........................";  

   // Projecting Centered Data onto PCs
   alpha = 1.0f;
   beta = 0.0f;
   cublas_status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               m, n, n, 
                               &alpha, cent_d_mat, lda,
                               d_A, n, &beta,
                               d_t_mat, m);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n// ERROR: Projecting Centered Data onto PCs.\n");}
   std::cerr << "COMPLETE\n";  

   ///////////////////////////////////////////////////////////////// 
   // Report results
   double *transformed = (double*)malloc(bytes);
   cudaMemcpy(transformed, d_t_mat, sizeof(double) * size, cudaMemcpyDeviceToHost);
   sort_matrix_descending(transformed, n, m);
   output_matrix(transformed, m, n, table.col_names);

   //////////////////////////////////////////////////////////////////////////////////
   // Free up memory
   cudaDeviceSynchronize();
   cudaFree(d_A);
   cudaFree(cent_d_mat);
   cudaFree(d_t_mat);

   cusolverDnDestroy(cusolver_handle);
   cudaDeviceReset();
  
   std::cerr << "// PROGRAM COMPLETED SUCCESSFULLY!\n";  
   return 0;
   }
