// Basic helper functions
#include <iostream>
#include <iomanip>
#include <algorithm>

double* get_one_vec(const int &size) {
    double *one_vec = (double*)malloc(size * sizeof(double));
    std::fill_n(one_vec, size, 1);
    return one_vec;
}

double* get_norm_vec(double *mat, const int &rows, const int &cols) {

    double *norm_vec = new double[cols];
    double tmp;

    for (int j = 0; j < cols; j++) {
        tmp = 0;
        for (int k = 0; k < rows; k++) {
            tmp += mat[j * rows + k];
        }
        norm_vec[j] = tmp / rows;
    }

    return norm_vec;
}


void sort_matrix_descending(double *mat,  const int &m, const int &n) {
    // Reverse the columns (in-place)
    for (int col = 0; col < m / 2; ++col) {
        for (int row = 0; row < n; ++row) {
            std::swap(mat[row + col * n], mat[row + (m - col - 1) * n]);
        }
    }
}

void print_matrix(double *mat, const int &rows, const int &cols) {
    int tmp = 0;
    std::cout << std::setprecision(5);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat[j * rows + i] << "\t";
        }
        std::cout << "\n";
    }
}


void output_matrix(double *mat, const int &rows, const int &cols, const std::vector<std::string> &rownames) {
     for (int i = 1; i < cols + 1; i++) {
        std::cout << "PC" << i << "\t";
     }
     std::cout << "\n";

     for (int i = 0; i < rows; i++) {
        std::cout << rownames.at(i) << "\t";
        std::cout << std::setprecision(5);
        for (int j = 0; j < cols; j++) {
            std::cout << mat[j * rows + i] << "\t";
        }
        std::cout << "\n";
    }

}
