#include <fixed_point_constraints.h>
#include <algorithm>
#include <iostream>

//Input:
//  q_size - total number of scalar generalized coordinates (3 times number of vertices in the mesh)
//  indices - indices (row ids in V) for fixed vertices
//Output:
//  P - 3*mx3*n sparse matrix which projects out fixed vertices
void fixed_point_constraints(Eigen::SparseMatrixd &P, unsigned int q_size, const std::vector<unsigned int> indices) {
    P.resize(q_size - 3 * indices.size(), q_size);
    P.setZero();

    int j = 0, k = 0;
    for (int i = 0; i < q_size / 3; i++) {
        if (i == indices[j]) { 
            j++;
        } else {
            P.coeffRef(3 * k, 3 * i) = 1;
            P.coeffRef(3 * k + 1, 3 * i + 1) = 1;
            P.coeffRef(3 * k + 2, 3 * i + 2) = 1;
            k++;
        }
    } 
}