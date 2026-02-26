#include <V_membrane_corotational.h>
#include <iostream>
//  q - generalized coordinates for the FEM system
//  dX - the 3x3 matrix containing dphi/dX
//  V - the nx3 matrix of undeformed vertex positions. Each row is a single undeformed vertex position.
//  element - the vertex indices of this triangle
//  area - the area of this triangle
//  mu,lambda - material parameters for the cloth material model
//Output:
//  energy- the per-triangle potential energy (the linear model described in the README).
//Allowed to use libigl SVD or Eigen SVD for this part
void V_membrane_corotational(double &energy, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::Matrix3d> dX, 
                          Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::RowVectorXi> element, double area, 
                          double mu, double lambda) {

    //Deformation Gradient
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

    energy = mu * (pow(S(0) - 1, 2) + pow(S(1) - 1, 2) + pow(S(2) - 1, 2)) + 0.5 * lambda * pow(S.sum() - 3, 2);
}
