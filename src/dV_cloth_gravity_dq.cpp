#include <dV_cloth_gravity_dq.h>

//  M - sparse mass matrix for the entire mesh
//  g - the acceleration due to gravity
//Output:
//  fg - the gradient of the gravitational potential for the entire mesh
void dV_cloth_gravity_dq(Eigen::VectorXd &fg, Eigen::SparseMatrixd &M, Eigen::Ref<const Eigen::Vector3d> g) {
    
    fg.resize(M.rows());
    fg.setZero();

    Eigen::VectorXd g_vec;
    g_vec.resize(M.rows());
    g_vec.setZero();
    for (int i = 0; i < g_vec.size(); i++) {
        g_vec(i) = g(i % 3);
    }

    fg = -M * g_vec;
}
