#include <assemble_forces.h>
#include <iostream>

//Input:
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
//  f - the vector 3xn vector of forces acting on each node of the mass-spring system
void assemble_forces(Eigen::VectorXd &f, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::MatrixXd> qdot, Eigen::Ref<const Eigen::MatrixXd> dX,
                     Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::MatrixXi> F, Eigen::Ref<const Eigen::VectorXd> a0,
                     double mu, double lambda) { 
        
        f.resize(q.size());
        f.setZero();

        for (int i = 0; i < F.rows(); i++) {

            Eigen::Matrix<double, 1, 9> tmp_row;
            tmp_row = dX.row(i); //ei is the triangle index. 
            Eigen::Matrix3d dX_i = Eigen::Map<const Eigen::Matrix3d>(tmp_row.data());

            Eigen::Vector9d dV = Eigen::Vector9d::Zero();
            dV_membrane_corotational_dq(dV, q, dX_i, V, F.row(i), a0(i), mu, lambda);

            for (int j = 0; j < F.row(i).size(); j++) {
                f(3 * F(i, j)    ) -= dV(3 * j    );
                f(3 * F(i, j) + 1) -= dV(3 * j + 1);
                f(3 * F(i, j) + 2) -= dV(3 * j + 2);
            }
        }
    };
