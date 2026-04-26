#ifndef CGMRES_SOLVER_H
#define CGMRES_SOLVER_H

#include "mpc_model.h"

#define KMAX 10 // Maximum number of internal Krylov iterations for GMRES
#define ZETA 100.0 // Stabilization parameter

// Replace u̇ with plain English u_dot
void cgmres_solve(const double x_current[], const double U[], double u_dot[]);

#endif
