#include "mpc_model.h"
// Cost matrix weights
const double Q_weight[4]  = {1.0, 1.0, 0.0, 1.0};
const double Sf_weight[4] = {10.0, 10.0, 0.0, 10.0};
const double R_weight[2]  = {1.0, 1.0};

// KBM state equations (Equation 2)
void kbm_forward(const double x[], const double u[], double dx[]) {
    double Fpy = x[0];
    double theta_F  = x[1];
    double Fpx = x[2];
    double V   = x[3];
    double delta = u[0];
    double a = u[1];
    
    double D = 1.0 - RHO_REF * Fpy; 
    double tan_delta = tan(delta);
    double beta = atan((LR / (LF + LR)) * tan_delta);
    
    dx[0] = V * sin(theta_F + beta);
    dx[1] = (V / LR) * sin(beta) + (V * RHO_REF * cos(theta_F + beta)) / D;
    dx[2] = (V * cos(theta_F + beta)) / D;
    dx[3] = a;
}

// Backward integration of costate equations and terminal conditions (Equations 14~16)
void calculate_costates(const double X[], const double U[], double lambda[]) {
    double dx_pert = 1e-6; // Forward difference step size
    
    // Terminal conditions λ_N = ∂Φ/∂x
    for (int i = 0; i < DIM_X; i++) {
        double term_error = X[N_HORIZON * DIM_X + i]; // Correction: The terminal target is also 10.0m/s
        if (i == 3) {
            term_error = term_error - 10.0;
        }
        lambda[N_HORIZON * DIM_X + i] = Sf_weight[i] * term_error;
    }
    
    // Backward integration λ_k = λ_{k+1} + Δt * ∂H/∂x_k
    for (int k = N_HORIZON - 1; k >= 0; k--) {
        const double* x_k = &X[k * DIM_X];
        const double* u_k = &U[k * DIM_U];
        const double* lambda_next = &lambda[(k + 1) * DIM_X];
        
        double dx_orig[DIM_X], dx_perturbed[DIM_X];
        kbm_forward(x_k, u_k, dx_orig);
        
        for (int i = 0; i < DIM_X; i++) {
            double x_pert_arr[DIM_X];
            for (int j = 0; j < DIM_X; j++) x_pert_arr[j] = x_k[j];
            x_pert_arr[i] += dx_pert;
            
            kbm_forward(x_pert_arr, u_k, dx_perturbed);
            
            double sum_lambda_fx = 0.0;
            for (int j = 0; j < DIM_X; j++) {
                sum_lambda_fx += lambda_next[j] * ((dx_perturbed[j] - dx_orig[j]) / dx_pert);
            }
            
            // double dL_dx = Q_weight[i] * x_k[i]; // ∂L/∂x
            double error = x_k[i]; // If it is the velocity dimension (i == 3), calculate the error with the target velocity 10.0
            if (i == 3) {
                error = x_k[i] - 10.0; 
            }
            double dL_dx = Q_weight[i] * error;
            // Euler backward difference
            lambda[k * DIM_X + i] = lambda_next[i] + DT * (dL_dx + sum_lambda_fx);
        }
    }
}

// Calculate the partial derivatives of the Hamiltonian with respect to the control variables ∂H/∂u
void calculate_Hu(const double X[], const double U[], const double lambda[], double Hu[]) {
    double du_pert = 1e-6;
    for (int k = 0; k < N_HORIZON; k++) {
        const double* x_k = &X[k * DIM_X];
        const double* u_k = &U[k * DIM_U];
        const double* lambda_next = &lambda[(k + 1) * DIM_X];
        
        double dx_orig[DIM_X], dx_perturbed[DIM_X];
        kbm_forward(x_k, u_k, dx_orig);
        
        for (int i = 0; i < DIM_U; i++) {
            double u_pert_arr[DIM_U];
            for (int j = 0; j < DIM_U; j++) u_pert_arr[j] = u_k[j];
            u_pert_arr[i] += du_pert;
            
            kbm_forward(x_k, u_pert_arr, dx_perturbed);
            
            double sum_lambda_fu = 0.0;
            for (int j = 0; j < DIM_X; j++) {
                sum_lambda_fu += lambda_next[j] * ((dx_perturbed[j] - dx_orig[j]) / du_pert);
            }
            
            double dL_du = R_weight[i] * u_k[i]; // ∂L/∂u
            Hu[k * DIM_U + i] = dL_du + sum_lambda_fu;
        }
    }
}
