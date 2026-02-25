#include <dphi_cloth_triangle_dX.h>

//Input:
//  V - the nx3 matrix of undeformed vertex positions. Each row is a single undeformed vertex position.
//  element - the 1x3 vertex indices for this tetrahedron
//  X - the 3D position in the underformed space at which to compute the gradient
//Output:
//  dphi - the 3x3 gradient of the the basis functions wrt to X. The i'th row stores d phi_i/dX
//compute 3x3 deformation gradient 
void dphi_cloth_triangle_dX(Eigen::Matrix3d &dphi, Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::RowVectorXi> element, Eigen::Ref<const Eigen::Vector3d> X) {

    Eigen::MatrixXd T;
    T.resize(V.cols(), element.size()); // 3x2
    T.setZero();
    for (int i = 0; i < T.rows(); i++) {
        for (int j = 0; j < T.cols() - 1; j++) {
            T(i, j) = V(element(j+1), i) - V(element(0), i);
        }
    }

    Eigen::Matrix2d T2 = T.transpose() * T;
    Eigen::Matrix2d T2_inv = T2.inverse();

    Eigen::MatrixXd phi12;
    phi12.resize(2, 3);  // 2x3
    phi12.setZero();
    phi12 = T2_inv * T.transpose();

    Eigen::Vector2d one = Eigen::Vector2d::Zero();
    one << 1, 1;

    Eigen::RowVector3d phi0 = Eigen::RowVector3d::Zero();
    phi0 = -one.transpose() * phi12;

    dphi << phi0,   // 1x3
            phi12;  // 2x3
    

}