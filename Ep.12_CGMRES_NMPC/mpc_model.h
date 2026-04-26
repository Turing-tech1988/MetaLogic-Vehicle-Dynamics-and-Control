#ifndef MPC_MODEL_H
#define MPC_MODEL_H

#include <math.h>

#define N_HORIZON 100      // prediction steps
#define DT 0.01            // control period 0.01s

#define DIM_X 4 // Fpy, θF, Fpx, V
#define DIM_U 2 // delta, a

// KBM (kinematic bicycle model)
#define LF 1.04
#define LR 1.56
#define RHO_REF 0.0 

void kbm_forward(const double x[], const double u[], double dx[]);
void calculate_costates(const double X[], const double U[], double lambda[]);
void calculate_Hu(const double X[], const double U[], const double lambda[], double Hu[]);

#endif
