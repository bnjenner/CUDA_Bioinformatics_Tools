// Basic helper functions

#include <iostream>
#include <iomanip>

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

double print_matrix(double *mat, const int &rows, const int &cols) {
    int tmp = 0;
    std::cout << std::setprecision(5);
    // for (int i = 0; i < size; i++) {
    //     std::cout << mat[i] << " ";
    // }
    // std::cout << std::endl;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat[j * rows + i] << "\t";
        }
        std::cout << "\n";
    }
}
