/////////////////////////////////////////////////////////////////
/*
	cuda_PCA
*/
/////////////////////////////////////////////////////////////////

// cuBLAS and cuSolver
#include "cublas_v2.h"
#include "cusolverDn.h"

// utils
#include "tsv.h"
#include "utils.h"

// STL
#include <iostream>
#include <cassert>
#include <getopt.h>

/////////////////////////////////////////////////////////////////
void display_help(const char* program) {
   std::cerr << "Description: cuda_PCA is a tool for Principal Component Analysis utilizing GPU computing\n"
             << "             through CUDA and its libraries. Current calculations are not batched, so\n"
             << "             GPUs with smaller memory capacities will fail to allocate on larger datasets.\n"
             << "             Resulting PC matrix is printed to standard out.\n\n"
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


/////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {

   // Arguments
   std::string filename;

   argparse(argc, argv, filename);

   if (filename.empty()) {
      display_help(argv[0]);
      std::cerr << "// ERROR: must include an input file.\n";
      exit(EXIT_FAILURE);
   }

   std::cerr << "// cuda_PCA\n";
   /////////////////////////////////////////////////////////////////
   // Read in CSV
    std::cerr << "// Parsing file..........................";
   TSV<double> table(filename);
   table.read_delim('\t');
   double *mat = table.flatten();  // Write to C array in column major order
   std::cerr << "COMPLETE\n";  

   // Get Dimensions and bytes
   unsigned long long int m = table.rows;   // Columns of original table
   unsigned long long int n = table.cols;   // Rows of original table
   const unsigned long long int lda = m;
   const unsigned long long int ldu = m;
   const unsigned long long int lds = std::min(m, n);
   const unsigned long long int ldv = n;
   const unsigned long long int size = m * n;
   const unsigned long long int u_size = m * m;
   const unsigned long long int v_size = n * n;
   const size_t bytes = size * sizeof(double);
   const size_t u_bytes = m * m * sizeof(double);
   const size_t v_bytes = m * m * sizeof(double);    


   // Create Matrix for Mean-Centering
   double *norm_mat = get_norm_mat(mat, m, n); 

   ///////////////////////////////////////////////////////////////// 
   // Create and initialize cuBLAS handle object
   cublasHandle_t handle;
   cublasCreate_v2(&handle);
   cublasStatus_t cublas_status;
   cudaError_t cuda_error;

   double *d_norm, *d_mat, *cent_d_mat;

   // Allocate GPU memory for vectors
   std::cerr << "// Allocating GPU memory.................";  
   cuda_error = cudaMalloc(&d_mat, bytes); cuda_assert(cuda_error);        // Data Matrix
   cuda_error = cudaMalloc(&d_norm, bytes); cuda_assert(cuda_error);       // Normalization vector
   std::cerr << "COMPLETE\n";  


   ///////////////////////////////////////////////////////////////// 
   // Set Matrix and vectors
   cuda_error = cudaMemcpy(d_mat, mat, bytes, cudaMemcpyHostToDevice); cuda_assert(cuda_error);
   cuda_error = cudaMemcpy(d_norm, norm_mat, bytes, cudaMemcpyHostToDevice); cuda_assert(cuda_error);


   ///////////////////////////////////////////////////////////////// 
   // Mean-Center Data
   std::cerr << "// Mean Centering Matrix.................";  
   double alpha = 1.0f;
   double beta = -1.0f;
   cublas_status = cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               m, n, 
                               &alpha, d_mat, lda,
                               &beta, d_norm, lda,
                               d_mat, lda);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n// ERROR: Could not center matrix.\n");}
   std::cerr << "COMPLETE\n"; 

   //////////////////////////////////////////////////////////////////
   // Free up memory
   cudaFree(d_norm);

   //////////////////////////////////////////////////////////////////
   // Singular Value Decomposition

   // Create cuSolver Handle
   cusolverDnHandle_t cusolver_handle;
   cusolverDnCreate(&cusolver_handle);
   cusolverStatus_t cusolver_status;

   // Create device pointers
   double *d_X;
   double *d_U, *d_S,  *d_VT;
   double *d_work, *d_rwork, *d_T;
   int *devInfo;

   // Allocate memory on GPU
   cuda_error = cudaMalloc(&d_X, bytes); cuda_assert(cuda_error);
   cuda_error = cudaMalloc((void**)&d_U, u_bytes); cuda_assert(cuda_error); 
   cuda_error = cudaMalloc((void**)&d_S, lds * sizeof(double)); cuda_assert(cuda_error); 
   cuda_error = cudaMalloc((void**)&d_VT, v_bytes); cuda_assert(cuda_error); 
   cuda_error = cudaMalloc((void**)&d_T, v_bytes); cuda_assert(cuda_error); 
   cuda_error = cudaMalloc((void**)&devInfo, sizeof(int)); cuda_assert(cuda_error); 

   // Copy Centered Matrix
   cuda_error = cudaMemcpy(d_X, d_mat, bytes, cudaMemcpyDeviceToDevice); cuda_assert(cuda_error);


   ///////////////////////////////////////////////////////////////
   // Create SVD Buffer
   std::cerr << "// Performing SVD........................";  
   int lwork = 0;
   cusolver_status = cusolverDnDgesvd_bufferSize(cusolver_handle, m, n, &lwork);
   if (cusolver_status != CUSOLVER_STATUS_SUCCESS) { throw std::runtime_error("FAILED\n//ERROR: Could not create SVD buffer.\n");}

   // Allocate memory for work size
   cuda_error = cudaMalloc((void**)&d_work, sizeof(double) * lwork); cuda_assert(cuda_error); 

   ///////////////////////////////////////////////////////////////
   // Compute SVD
   signed char jobu = 'A'; // full U
   signed char jobvt = 'A'; // full VT
   cusolver_status = cusolverDnDgesvd(cusolver_handle, jobu, jobvt,
                                      m, n, d_X, lda, 
                                      d_S, d_U, ldu,
                                      d_VT, ldv,
                                      d_work, lwork, d_rwork, devInfo);
   if (cusolver_status != CUSOLVER_STATUS_SUCCESS) { throw std::runtime_error("FAILED\n// ERROR: Could not perform SVD.\n"); }
   std::cerr << "COMPLETE\n";  

   ///////////////////////////////////////////////////////////////// 
   // Projecting Centered Data onto PCs
   std::cerr << "// Projecting Data.......................";  
   alpha = 1.0f;
   beta = 0.0f;
   cublas_status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               n, ldu, m, 
                               &alpha, d_mat, m,
                               d_U, ldu, &beta,
                               d_T, n);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n// ERROR: Projecting Centered Data onto PCs.\n");}
   std::cerr << "COMPLETE\n";  

   /////////////////////////////////////////////////////////////// 
   // Report results
   double *transformed = (double*)malloc(v_bytes);
   cudaMemcpy(transformed, d_T, v_bytes, cudaMemcpyDeviceToHost); cuda_assert(cuda_error);
   output_pca(transformed, n, n, table.col_names);

   ///////////////////////////////////////////////////////////////
   // Free up memory
   cudaFree(d_U);
   cudaFree(d_S);
   cudaFree(d_VT);
   cudaFree(devInfo);
   cudaFree(d_work);
   cudaFree(d_X);
   cudaFree(d_mat);

   cusolverDnDestroy(cusolver_handle);
   cudaDeviceReset();
  
   std::cerr << "// PROGRAM COMPLETED SUCCESSFULLY!\n";  
   return 0;
}
