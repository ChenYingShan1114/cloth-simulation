#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <EigenTypes.h>

//Input:
//  q - generalized coordinates for the FEM system
//  qdot - generalized velocity for the FEM system
//  dt - the time step in seconds
//  mass - the mass matrix
//  force(f, q, qdot) - a function that computes the force acting on the FEM system. This takes q and qdot as parameters, returns the force in f.
//  stiffness(K, q, qdot) - a function that computes the stiffness (negative second derivative of the potential energy). This takes q and qdot as parameters, returns the stiffness matrix in K.  
//  tmp_force - scratch space to collect forces
//  tmp_stiffness - scratch space to collect stiffness matrix
//Output:
//  q - set q to the updated generalized coordinate using linearly implicit time integration
//  qdot - set qdot to the updated generalized velocity using linearly implicit time integration
template<typename FORCE, typename STIFFNESS> 
inline void linearly_implicit_euler(Eigen::VectorXd &q, Eigen::VectorXd &qdot, double dt, 
                            const Eigen::SparseMatrixd &mass,  FORCE &force, STIFFNESS &stiffness, 
                            Eigen::VectorXd &tmp_force, Eigen::SparseMatrixd &tmp_stiffness) {
    
    force(tmp_force, q, qdot);
    stiffness(tmp_stiffness, q, qdot);

    Eigen::SparseMatrixd A;
    A = mass - dt * dt * tmp_stiffness;
    A.makeCompressed(); // Recommended after filling

    // Declare the SimplicialLDLT solver class
    Eigen::SimplicialLDLT<Eigen::SparseMatrixd> solver;

    // Compute the LDLT factorization of A
    solver.compute(A);
    
    // Check if the decomposition was successful
    if (solver.info() != Eigen::Success) {
        std::cerr << "LDLT decomposition failed. The matrix might not be symmetric positive definite." << std::endl;
    }

    // Create an identity matrix I to represent the right-hand side of AX = I
    // Note: The "inverse" matrix A_inv will be densem so a dense MatrixXd is used
    Eigen::MatrixXd I(A.rows(), A.cols());
    I.setIdentity();

    // Solve the system AX = I for X, which results in the inverse matrix A_inv
    Eigen::MatrixXd A_inv = solver.solve(I);

    qdot = A_inv * (mass * qdot + dt * tmp_force); 
    q = q + dt * qdot;

}