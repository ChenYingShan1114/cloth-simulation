#include <dV_membrane_corotational_dq.h>
#include <iostream>

//  q - generalized coordinates for the FEM system
//  dX - the 3x3 matrix containing dphi/dX
//  V - the nx3 matrix of undeformed vertex positions. Each row is a single undeformed vertex position.
//  element - the vertex indices of this triangle
//  area - the area of this triangle
//  mu,lambda - material parameters for the cloth material model
//Output:
//  f - the per-triangle gradient of the membrane potential energy (the linear model described in the README).
void dV_membrane_corotational_dq(Eigen::Vector9d &dV, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::Matrix3d> dX, 
                          Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::RowVectorXi> element, double area, 
                          double mu, double lambda) {

    //Deformation Gradient
    Eigen::Matrix3d dx; //deformed tangent matrix 
    Eigen::Matrix3d U;
    Eigen::Vector3d S; 
    Eigen::Matrix3d W; 
    Eigen::Matrix3d F;

    //TODO: SVD Here
    // reference coordinate normal vector
    Eigen::Vector3d dX1 = V.row(element(1)) - V.row(element(0));
    Eigen::Vector3d dX2 = V.row(element(2)) - V.row(element(0));
    Eigen::Vector3d N = dX1.cross(dX2);
    N.normalize();

    Eigen::Matrix<double, 4, 3> dphi_43;
    dphi_43 << dX, N.transpose();

    // world coordinate normal vector
    Eigen::Vector3d dx1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d dx2 = Eigen::Vector3d::Zero();
    for (int i = 0; i < element.size(); i++) {
        dx1(i) = q(3 * element(1) + i) - q(3 * element(0) + i);
        dx2(i) = q(3 * element(2) + i) - q(3 * element(0) + i);
    }
    Eigen::Vector3d n = dx1.cross(dx2);
    Eigen::Vector3d n_nor = n.normalized();

    Eigen::Matrix<double, 3, 4> q_tri;
    q_tri.setZero();
    for (int i = 0; i < q_tri.rows(); i++) {
        for (int j = 0; j < q_tri.cols() - 1; j++) {
            q_tri(i, j) = q(3 * element(j) + i);
        }
        q_tri(i, q_tri.cols() - 1) = n_nor(i);
    }
    
    F = q_tri * dphi_43;
    // std::cout << "Deformation Gradient F:\n" << F << "\n\n";

    // Perform SVD using JacobiSVD.
    // Eigen::ComputeThinU | Eigen::ComputeThinV computes the thin U and V matrices.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Access the results
    U = svd.matrixU();
    S = svd.singularValues();
    W = svd.matrixV();

    // std::cout << "U matrix:\n" << U << "\n\n";
    // std::cout << "Singular values vector S:\n" << S << "\n\n";
    // std::cout << "V matrix:\n" << W << "\n\n";

    //Fix for inverted elements (thanks to Danny Kaufman)
    double det = S[0]*S[1];
    
    if(det <= -1e-10)
    {
        if(S[0] < 0) S[0] *= -1;
        if(S[1] < 0) S[1] *= -1;
        if(S[2] < 0) S[2] *= -1;
    }
    
    if(U.determinant() <= 0)
    {
        U(0, 2) *= -1;
        U(1, 2) *= -1;
        U(2, 2) *= -1;
    }
    
    if(W.determinant() <= 0)
    {
        W(0, 2) *= -1;
        W(1, 2) *= -1;
        W(2, 2) *= -1;
    }

    //TODO: energy model gradient 
    
    // dpsi/ds
    Eigen::Matrix3d dpsi_ds = Eigen::Matrix3d::Zero();
    for (int i = 0; i < dpsi_ds.rows(); i++) {
        dpsi_ds(i, i) = 2 * mu * (S(i) - 1) + lambda * (S.sum() - 3);
    }
    // dpsi/dF: 3x3 -> 1x9
    Eigen::Matrix3d dpsi_33 = U * dpsi_ds * W.transpose();
    Eigen::Vector9d dpsi_9 = Eigen::Vector9d::Zero(); // 9x1
    for (int i = 0; i < dpsi_33.rows(); i++) {
        for (int j = 0; j < dpsi_33.cols(); j++) {
            dpsi_9(3 * i + j) = dpsi_33(i, j);
        }
    }

    // dn/dq: 3x9
    Eigen::Matrix3d I;
    I.setIdentity();
    Eigen::Matrix3d Zeros = Eigen::Matrix3d::Zero();

    Eigen::Matrix3d dx1_cross = Eigen::Matrix3d::Zero();
    dx1_cross <<      0, -dx1(2),  dx1(1), 
                 dx1(2),       0, -dx1(0), 
                -dx1(1),  dx1(0),       0;
    Eigen::Matrix3d dx2_cross = Eigen::Matrix3d::Zero();
    dx2_cross <<      0, -dx2(2),  dx2(1), 
                 dx2(2),       0, -dx2(0), 
                -dx2(1),  dx2(0),       0;
    Eigen::Matrix<double, 3, 9> I1;
    I1.setZero();
    I1 << -I, I, Zeros;
    Eigen::Matrix<double, 3, 9> I2;
    I2.setZero();
    I2 << -I, Zeros, I;

    Eigen::Matrix<double, 3, 9> dn;
    dn.setZero();
    dn = (I - n_nor * n_nor.transpose()) / n.norm() * (dx1_cross * I2 - dx2_cross * I1);

    // put dphi/dX and N to big matrix B and N_93
    Eigen::Matrix99d B = Eigen::Matrix99d::Zero();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 9; j++) {
            B(j % 3 + 3 * i, 3 * (j / 3) + i) = dX(j / 3, j % 3);
        }
    }

    Eigen::Matrix<double, 9, 3> N_93;
    N_93.setZero();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < N.size(); j++) {
            N_93(3 * i + j, i) = N(j);
        }
    }
    
    // dF/dq
    B = B + N_93 * dn;

    dV = area * B.transpose() * dpsi_9;

}
