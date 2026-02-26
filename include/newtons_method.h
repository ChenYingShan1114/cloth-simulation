#include <Eigen/Dense>
#include <EigenTypes.h>

//Input:
//  x0 - initial point for newtons search
//  f(x) - function that evaluates and returns the cost function at x
//  g(dx, x) - function that evaluates and returns the gradient of the cost function in dx
//  H(dH, x) - function that evaluates and returns the Hessian in dH (as a sparse matrix).
//  max steps - the maximum newton iterations to take
//  tmp_g and tmp_H are scratch space to store gradients and hessians
//Output: 
//  x0 - update x0 to new value
template<typename Objective, typename Jacobian, typename Hessian>
double newtons_method(Eigen::VectorXd &x0, Objective &f, Jacobian &g, Hessian &H, unsigned int maxSteps, Eigen::VectorXd &tmp_g, Eigen::SparseMatrixd &tmp_H) {
   
   tmp_g.resize(x0.size());
   tmp_H.resize(x0.size(), x0.size());
   tmp_g.setZero();
   tmp_H.setZero();

   Eigen::VectorXd d;
   d.resize(x0.size());
   d.setZero();

   double alpha = 1, p = 0.5, c = 1e-8, tol = 1e-8;
   
   for (int i = 0; i < maxSteps; i++) {
      tmp_g.setZero();
      g(tmp_g, x0);

      if (tmp_g.norm() < tol) {
         return 0.0;
      } else {
         // Gradient descent
         // d = -tmp_g;
         
         // Newton's Method
         H(tmp_H, x0);
         tmp_H.makeCompressed(); // Recommended after filling

         // Declare the SimplicialLDLT solver class
         Eigen::SimplicialLDLT<Eigen::SparseMatrixd> solver;

         // Compute the LDLT factorization of tmp_H
         solver.compute(tmp_H);
         
         // Check if the decomposition was successful
         if (solver.info() != Eigen::Success) {
            std::cerr << "LDLT decomposition failed. The matrix might not be symmetric positive definite." << std::endl;
         }

         // Create an identity matrix I to represent the right-hand side of tmp_HX = I
         // Note: The "inverse" matrix tmp_H_inv will be densem so a dense MatrixXd is used
         Eigen::MatrixXd I(tmp_H.rows(), tmp_H.cols());
         I.setIdentity();

         // Solve the system tmp_HX = I for X, which results in the inverse matrix tmp_H_inv
         Eigen::MatrixXd tmp_H_inv = solver.solve(I);
         
         d = -tmp_H_inv * tmp_g;

         while (f(x0 + alpha * d) > f(x0) + c * d.transpose() * tmp_g && alpha > tol) {
            alpha = p * alpha;
         }

         x0 = x0 + alpha * d;

      }
   }

   return 0.0;
}
