#include <assemble_stiffness.h>
#include <iostream>

//  q - generalized coordinates for the FEM system
//  qdot - generalized velocity for the FEM system
//  dX - an mx9 matrix which stores the flattened dphi/dX matrices for each tetrahedron.
//       Convert this values back to 3x3 matrices using the following code (NOTE YOU MUST USE THE TEMPORARY VARIABLE tmp_row):
//       Eigen::Matrix<double, 1,9> tmp_row
//       tmp_row = dX.row(ei); //ei is the triangle index. 
//       Eigen::Map<const Eigen::Matrix3d>(tmp_row.data())
//  V - the nx3 matrix of undeformed vertex positions. Each row is a single undeformed vertex position.
//  F - the mx3 triangle connectivity matrix. Each row contains to indices into V that indicate a spring between those vertices.
//  a0 - the mx1 vector of undeformed triangle areas
//  mu,lambda - material parameters for the cloth material model
//Output:
//  K- the 3n by 3n sparse stiffness matrix. 
void assemble_stiffness(Eigen::SparseMatrixd &K, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::VectorXd> qdot, Eigen::Ref<const Eigen::MatrixXd> dX,
                     Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::MatrixXi> F, Eigen::Ref<const Eigen::VectorXd> a0, 
                     double mu, double lambda) { 
        K.resize(q.size(), q.size());
        K.setZero();

        for (int i = 0; i < F.rows(); i++) {

            Eigen::Matrix<double, 1, 9> tmp_row;
            tmp_row = dX.row(i); //ei is the triangle index. 
            Eigen::Matrix3d dX_i = Eigen::Map<const Eigen::Matrix3d>(tmp_row.data());

            Eigen::Matrix99d H = Eigen::Matrix99d::Zero();
            d2V_membrane_corotational_dq2(H, q, dX_i, V, F.row(i), a0(i), mu, lambda);

            for (int j = 0; j < F.row(i).size(); j++) {
                for (int k = 0; k < F.row(i).size(); k++) {
                    int i1 = F(i, j), i2 = F(i, k);

                    K.coeffRef(3 * i1    , 3 * i2    ) -= H(3 * j    , 3 * k    );
                    K.coeffRef(3 * i1    , 3 * i2 + 1) -= H(3 * j    , 3 * k + 1);
                    K.coeffRef(3 * i1    , 3 * i2 + 2) -= H(3 * j    , 3 * k + 2);
                    K.coeffRef(3 * i1 + 1, 3 * i2    ) -= H(3 * j + 1, 3 * k    );
                    K.coeffRef(3 * i1 + 1, 3 * i2 + 1) -= H(3 * j + 1, 3 * k + 1);
                    K.coeffRef(3 * i1 + 1, 3 * i2 + 2) -= H(3 * j + 1, 3 * k + 2);
                    K.coeffRef(3 * i1 + 2, 3 * i2    ) -= H(3 * j + 2, 3 * k    );
                    K.coeffRef(3 * i1 + 2, 3 * i2 + 1) -= H(3 * j + 2, 3 * k + 1);
                    K.coeffRef(3 * i1 + 2, 3 * i2 + 2) -= H(3 * j + 2, 3 * k + 2);
                    
                }
            }
        }
    };
