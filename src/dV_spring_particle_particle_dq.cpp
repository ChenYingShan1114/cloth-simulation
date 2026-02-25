#include <dV_spring_particle_particle_dq.h>
#include <iostream>

//Input:
//  q0 - the generalized coordinates of the first node of the spring
//  q1 - the generalized coordinates of the second node of the spring
//  l0 - the undeformed length of the spring
//  stiffness - the stiffness constant for this spring
//Output:
//  f - the 6x1 per spring energy gradient
void dV_spring_particle_particle_dq(Eigen::Ref<Eigen::Vector6d> f, Eigen::Ref<const Eigen::Vector3d> q0,  Eigen::Ref<const Eigen::Vector3d>     q1, double l0, double stiffness) {
    f.setZero();

    double dist = (q1 - q0).norm();
    double force = -stiffness * (dist - l0) / dist;
    f << force * (q1 - q0), force * (q0 - q1);
    
}