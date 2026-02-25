#include <mass_matrix_mesh.h>

//Input: 
//  q - generalized coordinates for the FEM system
//  V - the nx3 matrix of undeformed vertex positions
//  F - the mx3 matrix of triangle-vertex indices
//  density - the density of the cloth material
//  areas - the mx1 vector of undeformed triangle areas
//Output:
//  M - sparse mass matrix for the entire mesh
void mass_matrix_mesh(Eigen::SparseMatrixd &M, Eigen::Ref<const Eigen::VectorXd> q, 
                         Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::MatrixXi> F,
                         double density, Eigen::Ref<const Eigen::VectorXd> areas) {
    M.resize(q.size(), q.size());
    M.setZero();

    // Assemble M0 to the sparse mass matrix M for the entire mesh
    for (int i = 0; i < areas.size(); i++) {
        double mass = density * areas(i) / 12.0;
        for (int j = 0; j < F.cols(); j++) {
            int i1 = F(i, j);
            for (int k = 0; k < F.cols(); k++) {
                int i2 = F(i, k);
                if (i1 == i2) {
                    M.coeffRef(3 * i1    , 3 * i2    ) += 2 * mass;
                    M.coeffRef(3 * i1 + 1, 3 * i2 + 1) += 2 * mass;
                    M.coeffRef(3 * i1 + 2, 3 * i2 + 2) += 2 * mass;
                } else {
                    M.coeffRef(3 * i1    , 3 * i2    ) += mass;
                    M.coeffRef(3 * i1 + 1, 3 * i2 + 1) += mass;
                    M.coeffRef(3 * i1 + 2, 3 * i2 + 2) += mass;
                }
            }
        }
    }
}
 