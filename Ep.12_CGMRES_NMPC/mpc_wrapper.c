#define S_FUNCTION_NAME  mpc_wrapper
#define S_FUNCTION_LEVEL 2

#include "simstruc.h"
#include "cgmres_solver.h"

// State inputs: Fpy, θF, Fpx, V
#define NUM_INPUTS  4

// Control outputs: δ, a
#define NUM_OUTPUTS 2

static double U_guess[N_HORIZON * DIM_U] = {0};

static void mdlInitializeSizes(SimStruct *S) {
    ssSetNumSFcnParams(S, 0);
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) return;
    
    ssSetNumContStates(S, 0);
    ssSetNumDiscStates(S, 0);
    
    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, NUM_INPUTS);
    ssSetInputPortDirectFeedThrough(S, 0, 1);
    
    if (!ssSetNumOutputPorts(S, 1)) return;
    ssSetOutputPortWidth(S, 0, NUM_OUTPUTS);
    
    ssSetNumSampleTimes(S, 1);
    ssSetNumRWork(S, 0);
    ssSetNumIWork(S, 0);
    ssSetNumPWork(S, 0);
    ssSetNumModes(S, 0);
    ssSetNumNonsampledZCs(S, 0);
}

static void mdlInitializeSampleTimes(SimStruct *S) {
    ssSetSampleTime(S, 0, DT); // 0.01s
    ssSetOffsetTime(S, 0, 0.0);
}

static void mdlOutputs(SimStruct *S, int_T tid) {
    InputRealPtrsType u_in = ssGetInputPortRealSignalPtrs(S, 0);
    real_T *y_out = ssGetOutputPortRealSignal(S, 0);
    
    double x_current[DIM_X];
    for (int i = 0; i < DIM_X; i++) {
        x_current[i] = *u_in[i];
    }
    
    // Call the GMRES solver engine to compute u̇
    double u_dot[N_HORIZON * DIM_U]={0.0};
    cgmres_solve(x_current, U_guess, u_dot);
    
    // Euler integration to update the control guess U(t + Δt) = U(t) + U̇ * Δt
    for (int i = 0; i < N_HORIZON * DIM_U; i++) {
        U_guess[i] += u_dot[i] * DT;
    }
    
    // Output the current optimal control law U₀
    y_out[0] = U_guess[0]; // δ
    y_out[1] = U_guess[1]; // a
}

static void mdlTerminate(SimStruct *S) {}

// [Fix] Standard S-Function ending, requires including simulink.c during MEX compilation
#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
