/////////////////////////////////////////////////////////////////
/*
	cuda_lmFit
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
   std::cerr << "Description: cuda_lmFit is a tool for fitting linear models utilizing GPU computing.\n"
             << "             Currently implements the (standard or weighted) least squares \n"
             << "             algorithm. Does not filter or perform normalization. \n"
             << "             Future implementations will fit every gene in a table \n"
             << "             Uses formula Y = Xb. Results are printed to standard out.\n\n"
             << "Usage: " << program << " [ OPTIONS ] Y_file X_file \n\n"
             << "Options:\n"
             << "  -h                Displays help message.\n"
             << "  -w weights_file   File containings wieghts. If not specified, least sqares is unweighted."
             << std::endl;
}


void argparse(int argc, char** argv, std::string &y_filename, std::string &x_filename, std::string &weights_file) {

   int opt;

   while ((opt = getopt(argc, argv, "w:h")) != -1) {
      switch (opt) {
         case 'w':
            weights_file = optarg;
            break; 
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

   y_filename = argv[optind];
   x_filename = argv[optind + 1];

}


/////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {


   std::string y_filename;
   std::string x_filename;
   std::string weights_file;

   argparse(argc, argv, y_filename, x_filename, weights_file);

   if (y_filename.empty()) {
      display_help(argv[0]);
      std::cerr << "//ERROR: must include an input Y file.\n";
      exit(EXIT_FAILURE);
   }

   if (x_filename.empty()) {
      display_help(argv[0]);
      std::cerr << "//ERROR: must include an input X file.\n";
      exit(EXIT_FAILURE);
   }

   std::cerr << "// cuda_lmFit\n";

   /////////////////////////////////////////////////////////////////
   // Read in X
   std::cerr << "// Parsing files....................";
   TSV<double> table(x_filename);
   table.read_delim('\t');
   double *mat = table.flatten(); // write transposed so end coords are for samples

   // Read in Y
   TSV<double> y_table(y_filename);
   y_table.read_delim('\t');
   double *y_mat = y_table.flatten(); // write transposed so end coords are for samples
   std::cerr << "COMPLETE\n"; 

   // Get Dimensions and bytes
   const unsigned long long int m = table.rows;
   const unsigned long long int n = table.cols;
   const unsigned long long int y_m = y_table.rows;
   const unsigned long long int y_n = y_table.cols;
   const unsigned long long int size = table.size;
   const size_t bytes = size * sizeof(double);
   const size_t id_bytes = n * n * sizeof(double);
   const size_t weight_bytes = m * sizeof(double);
   const size_t y_bytes = y_m * y_n * sizeof(double);


   /////////////////////////////////////////////////////////////////
   // Create pointers to GPU
   double *d_mat, *d_w_mat;
   double *d_y_mat, *d_w_y_mat;
   double *d_id_mat, *d_t_mat, *d_w_vec;
   double *d_res_mat;

   /////////////////////////////////////////////////////////////////
   // Get weight vector
   double *weight_vec;

   if (!weights_file.empty()) {
      TSV<double> weights(weights_file);
      weights.read_delim('\t');
      weight_vec = weights.flatten();
   
   } else {
      weight_vec = get_one_vec(m);
   }

   ///////////////////////////////////////////////////////////////// 
   // Create and initialize cuBLAS handler objects
   cublasHandle_t handle;
   cublasCreate_v2(&handle);
   cublasStatus_t cublas_status;
   cudaError_t cuda_error;

   // Allocate GPU memory for vectors
   std::cerr << "// Allocating GPU memory............";   
   cuda_error = cudaMalloc((void**)&d_mat, bytes); cuda_assert(cuda_error);                            // Data Matrix
   cuda_error = cudaMalloc((void**)&d_y_mat, y_bytes); cuda_assert(cuda_error);                        // Data Y Matrix
   cuda_error = cudaMalloc((void**)&d_w_mat, bytes); cuda_assert(cuda_error);                          // Weighted Data Matrix
   cuda_error = cudaMalloc((void**)&d_w_y_mat, y_bytes); cuda_assert(cuda_error);                      // Weighted Y Matrix
   cuda_error = cudaMalloc((void**)&d_w_vec, weight_bytes); cuda_assert(cuda_error);                   // Identity Matrix
   cuda_error = cudaMalloc((void**)&d_id_mat, id_bytes); cuda_assert(cuda_error);                      // Identity Matrix
   cuda_error = cudaMalloc((void**)&d_t_mat, n * y_m * sizeof(double)); cuda_assert(cuda_error);       // Temporary Matrix
   cuda_error = cudaMalloc((void**)&d_res_mat, n * y_m * sizeof(double)); cuda_assert(cuda_error);     // Result Matrix
   std::cerr << "COMPLETE\n";  


   // Set Matrix and vectors (I am treating vector as matrices, deal it)
   std::cerr << "// Setting Matrices.................";  
   cublas_status = cublasSetMatrix(1, 1, bytes, mat, 1, d_mat, 1);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n//ERROR: Could not set data matrix.\n");}

   cublas_status = cublasSetMatrix(1, 1, weight_bytes, weight_vec, 1, d_w_vec, 1); 
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n//ERROR: Could not set normalization vector.\n");}

   cublas_status = cublasSetMatrix(1, 1, y_bytes, y_mat, 1, d_y_mat, 1); 
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n//ERROR: Could not set normalization vector.\n");}
   std::cerr << "COMPLETE\n";  


   ///////////////////////////////////////////////////////////////// 
   std::cerr << "// Weighting Matrices...............";  

   // Weight X matrix 
   cublas_status = cublasDdgmm(handle, CUBLAS_SIDE_LEFT,
                               m, n, d_mat, m,
                               d_w_vec, 1,
                               d_w_mat, m);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n//ERROR: Could not weight matrix.\n");}

   // Weight Y matrix
   cublas_status = cublasDdgmm(handle, CUBLAS_SIDE_LEFT,
                               y_n, y_m, d_y_mat, y_n,
                               d_w_vec, 1,
                               d_w_y_mat, y_n);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n//ERROR: Could not weight matrix.\n");}
   std::cerr << "COMPLETE\n";  


   ///////////////////////////////////////////////////////////////// 
   std::cerr << "// Calculating X^t * X..............";  

   // Calculate "Identity" matrix 
   double alpha = 1.0f;
   double beta = 0.0f;
   cublas_status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               n, n, m, 
                               &alpha, d_mat, m,
                               d_w_mat, m, &beta,
                               d_id_mat, n);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n//ERROR: Could not calculate covariance matrix.\n");}
   std::cerr << "COMPLETE\n";  


   ///////////////////////////////////////////////////////////////// 
   std::cerr << "// Calculating Inverse..............";  

   // Set LU Decomposition Pointers
   double *d_inv, *d_work;
   int *d_pivot, *d_info;
   int lwork;

   cuda_error = cudaMalloc((void**)&d_inv, n * n * sizeof(double)); cuda_assert(cuda_error);       // Inverse Matrix
   cuda_error = cudaMalloc((void**)&d_pivot, n * sizeof(int)); cuda_assert(cuda_error);            // Pivot Indices Matrix 
   cuda_error = cudaMalloc((void**)&d_info, sizeof(int)); cuda_assert(cuda_error);                 // Device Info

   // Create cuSolver Handle
   cusolverDnHandle_t cusolver_handle;
   cusolverDnCreate(&cusolver_handle);
   cusolverStatus_t cusolver_status;

   // Create Buffer
   cusolver_status = cusolverDnDgetrf_bufferSize(cusolver_handle, n, n, d_id_mat, n, &lwork);
   if (cusolver_status != CUSOLVER_STATUS_SUCCESS) { throw std::runtime_error("\n//ERROR: Could not create LU buffer.\n");}

   cuda_error = cudaMalloc((void**)&d_work, lwork * sizeof(double)); cuda_assert(cuda_error); 

   // Compute LU Decomposition
   cusolver_status = cusolverDnDgetrf(cusolver_handle,
                                      n, n, d_id_mat, n,
                                      d_work, d_pivot,
                                      d_info);
   if (cusolver_status != CUSOLVER_STATUS_SUCCESS) { throw std::runtime_error("\n//ERROR: Could not perform LU decomposition.\n");}

   // Create identity matrix
   double *ones = get_one_vec(n);
   for (int i = 0; i < n; i++) {
      cuda_assert(cudaMemset(d_inv + i * n, 0, n * sizeof(double)));
      cuda_assert(cudaMemcpy(d_inv + i * n + i, ones, sizeof(double), cudaMemcpyHostToDevice));
   }

   // Solve for inverse
   cusolver_status = cusolverDnDgetrs(cusolver_handle, CUBLAS_OP_N,
                                      n, n, d_id_mat, n,
                                      d_pivot, d_inv,
                                      n, d_info);
   if (cusolver_status != CUSOLVER_STATUS_SUCCESS) { throw std::runtime_error("\n//ERROR: Could not compute inverse.\n");}
   std::cerr << "COMPLETE\n";  


   ///////////////////////////////////////////////////////////////// 
   // Transpose X matrix * Weighted Y
   std::cerr << "// Calculating X^t * Y..............";    
   cublas_status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               n, y_n, m, 
                               &alpha, d_mat, m,
                               d_w_y_mat, y_n, &beta,
                               d_t_mat, n);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n//ERROR: Could not calculate product of X^t and Y.\n");}
   std::cerr << "COMPLETE\n";  


   ///////////////////////////////////////////////////////////////// 
   // Inverse * (Transpose X matrix * Weighted Y)
   std::cerr << "// Calculating Coefficients.........";
   cublas_status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               n, y_m, n, 
                               &alpha, d_inv, n,
                               d_t_mat, y_n, &beta,
                               d_res_mat, n);
   if (cublas_status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error("\n//ERROR: Could not calculate final weights.\n");}
   std::cerr << "COMPLETE\n";  

   ///////////////////////////////////////////////////////////////// 
   // Report results

   double *result_matrix = (double*)malloc(n * y_m * sizeof(double));
   cudaMemcpy(result_matrix, d_res_mat, n * y_m * sizeof(double), cudaMemcpyDeviceToHost);
   output_fit(result_matrix, y_m, n, table.col_names);


   //////////////////////////////////////////////////////////////////////////////////
   // Free up memory
   cudaDeviceSynchronize();
   cudaFree(d_mat);
   cudaFree(d_w_mat);
   cudaFree(d_y_mat);
   cudaFree(d_w_y_mat);
   cudaFree(d_id_mat);
   cudaFree(d_t_mat);
   cudaFree(d_res_mat);
  
   std::cerr << "//PROGRAM COMPLETED SUCCESSFULLY!\n";  
   return 0;
   }
