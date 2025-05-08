// Basic helper functions
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// CUDA Error Handling
void cuda_assert(cudaError_t status) {
   if (status != cudaSuccess) {
      std::cerr << "\n// CUDA_ERROR: "
                << cudaGetErrorString(status) 
                << ".\n";
   exit(EXIT_FAILURE);
   }
}

// Create Vector of Ones
double* get_one_vec(const int &size) {
    double *one_vec = (double*)malloc(size * sizeof(double));
    std::fill_n(one_vec, size, 1);
    return one_vec;
}

// Get Mean Matrix for Mean Centering
double* get_norm_mat(double *mat, const int &rows, const int &cols) {

    double tmp;
    double *norm_vec = new double[rows];
    double *norm_mat = new double[rows * cols];

    // Calculate Col Means
    for (int i = 0; i < rows; i++) {
        tmp = 0;
        for (int j = 0; j < cols; j++) {
            tmp += mat[j * rows + i];
        }
        norm_vec[i] = tmp / cols;
    }

    // Populate Norm Matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            norm_mat[j * rows + i] = norm_vec[i];
        }
    }

    return norm_mat;
}

// Print lmFit Output (will generalize in future)
void output_fit(double *mat, const int &rows, const int &cols, const std::vector<std::string> &names) {
    // Print Header
    for (int i = 0; i < cols; i++) { std::cout << names.at(i) << "\t"; }
    std::cout << "\n";

    // Print Data
    int tmp = 0;
    std::cout << std::setprecision(5);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat[j * rows + i] << "\t";
        }
        std::cout << "\n";
    }
}

// Print PCA Output (will generalize in future)
void output_pca(double *mat, const int &rows, const int &cols, const std::vector<std::string> &rownames) {
    // Print Header
    for (int i = 1; i < cols + 1; i++) { std::cout << "PC" << i << "\t"; }
    std::cout << "\n";

    // Print Data
    for (int i = 0; i < rows; i++) {
        std::cout << rownames.at(i) << "\t";
        std::cout << std::setprecision(5);
        for (int j = 0; j < cols; j++) {
            std::cout << mat[j * rows + i] << "\t";
        }
        std::cout << "\n";
    }
}
