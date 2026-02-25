#include <collision_detection_cloth_sphere.h>
#include <iostream>
void collision_detection_cloth_sphere(std::vector<unsigned int> &cloth_index, std::vector<Eigen::Vector3d> &normals, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::Vector3d> center, double radius) {

    cloth_index.clear();
    normals.clear();

    for (int i = 0; i < q.size() / 3; i++) {
        Eigen::Vector3d qi = q.segment<3>(3 * i);
        Eigen::Vector3d n = qi - center;
        double dist = (qi - center).norm();
        if (dist < radius) {
            n.normalize();
            normals.push_back(n);
            cloth_index.push_back(i);
        }
    }
}