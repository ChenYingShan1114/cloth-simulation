#include <velocity_filter_cloth_sphere.h>
#include <algorithm>

void velocity_filter_cloth_sphere(Eigen::VectorXd &qdot, const std::vector<unsigned int> &indices, 
                                  const std::vector<Eigen::Vector3d> &normals) {

    double alpha = 0;
    for (int i = 0; i < indices.size(); i++) {
        Eigen::Vector3d qdot_tmp = qdot.segment<3>(3 * indices[i]);
        double alpha_tmp = normals[i].transpose() * qdot_tmp;
        double alpha = -std::min(0.0, alpha_tmp);
        qdot.segment<3>(3 * indices[i]) = qdot_tmp + alpha * normals[i];
    }

}