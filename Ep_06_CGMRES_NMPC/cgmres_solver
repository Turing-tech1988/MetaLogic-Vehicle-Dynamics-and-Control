#include "cgmres_solver.h"

#define TOTAL_U (N_HORIZON * DIM_U)

static void compute_F(const double x_curr[], const double U[], double F_vec[]) {
    double X[DIM_X * (N_HORIZON + 1)];
    double lambda_vec[DIM_X * (N_HORIZON + 1)]; // Replace λ
    
    for (int i = 0; i < DIM_X; i++) X[i] = x_curr[i];
    
    // Forward integration to predict the state trajectory
    for (int k = 0; k < N_HORIZON; k++) {
        double dx[DIM_X]; // Replace ẋ
        kbm_forward(&X[k * DIM_X], &U[k * DIM_U], dx);
        for (int i = 0; i < DIM_X; i++) {
            X[(k + 1) * DIM_X + i] = X[k * DIM_X + i] + dx[i] * DT;
        }
    }
    
    calculate_costates(X, U, lambda_vec);
    calculate_Hu(X, U, lambda_vec, F_vec);
}

// Forward difference to calculate the directional derivative
static void forward_difference_F(const double x_curr[], const double U[], const double u_dot[], const double F_orig[], double dF[]) {
    double h = 1e-5;
    double U_pert[TOTAL_U];
    double F_pert[TOTAL_U];
    
    for (int i = 0; i < TOTAL_U; i++) {
        U_pert[i] = U[i] + h * u_dot[i];
    }
    compute_F(x_curr, U_pert, F_pert);
    for (int i = 0; i < TOTAL_U; i++) {
        dF[i] = (F_pert[i] - F_orig[i]) / h;
    }
}

// Arnoldi orthogonalization and GMRES core
void cgmres_solve(const double x_current[], const double U[], double u_dot[]) {
    double F_orig[TOTAL_U];
    compute_F(x_current, U, F_orig);
    
    double b[TOTAL_U];
    for (int i = 0; i < TOTAL_U; i++) {
        b[i] = -ZETA * F_orig[i];
    }
    
    // FdGMRES initialization
    double r[TOTAL_U];
    double dF[TOTAL_U];
    forward_difference_F(x_current, U, u_dot, F_orig, dF);
    
    double norm_r = 0.0;
    for (int i = 0; i < TOTAL_U; i++) {
        r[i] = b[i] - dF[i];
        norm_r += r[i] * r[i];
    }
    norm_r = sqrt(norm_r);
    if (norm_r < 1e-6) return;
    
    double V[KMAX + 1][TOTAL_U];
    double H_mat[KMAX + 1][KMAX] = {0};
    double g[KMAX + 1] = {0};
    g[0] = norm_r;
    
    for (int i = 0; i < TOTAL_U; i++) {
        V[0][i] = r[i] / norm_r;
    }
    
    int k_iters = 0;
    for (int k = 0; k < KMAX; k++) {
        k_iters++;
        forward_difference_F(x_current, U, V[k], F_orig, dF);
        
        for (int i = 0; i < TOTAL_U; i++) {
            V[k+1][i] = dF[i];
        }
        
        // Arnoldi orthogonalization
        for (int j = 0; j <= k; j++) {
            double dot = 0.0;
            for (int i = 0; i < TOTAL_U; i++) dot += V[j][i] * V[k+1][i];
            H_mat[j][k] = dot;
            for (int i = 0; i < TOTAL_U; i++) V[k+1][i] -= dot * V[j][i];
        }
        
        double norm_v = 0.0;
        for (int i = 0; i < TOTAL_U; i++) norm_v += V[k+1][i] * V[k+1][i];
        norm_v = sqrt(norm_v);
        H_mat[k+1][k] = norm_v;
        for (int i = 0; i < TOTAL_U; i++) V[k+1][i] /= norm_v;
    }
    
    // Least squares solution
    double y[KMAX] = {0};
    for (int k = k_iters - 1; k >= 0; k--) {
        y[k] = g[k];
        for (int j = k + 1; j < k_iters; j++) {
            y[k] -= H_mat[k][j] * y[j];
        }
        y[k] /= H_mat[k][k];
    }
    
    // Update the control variable derivative u̇
    for (int k = 0; k < k_iters; k++) {
        for (int i = 0; i < TOTAL_U; i++) {
            u_dot[i] += V[k][i] * y[k];
        }
    }
}
