import numpy as np
import matplotlib.pyplot as plt


def reproduce_fig5_robust():
    # --- 1. Parameter Definitions ---
    m2 = 500.0  # Sprung mass [kg]
    m1 = 80.0  # Unsprung mass [kg]
    ks = 30000.0  # Spring stiffness [N/m]
    cs = 2000.0  # Damping coefficient [N/(m/s)]
    kt = 300000.0  # Tire stiffness [N/m]

    # Control parameters
    csh = 0.4 * cs
    tc = 0.03  # Time constant 30ms

    # Frequency range
    f = np.logspace(np.log10(0.5), np.log10(20), 500)
    omega = 2 * np.pi * f

    # --- 2. Frequency Response Calculation Function ---
    def get_robust_response(control_mode):
        # State vector x = [z2, z2_dot, z1, z1_dot, Fc]
        # Dimension definition
        n_states = 5
        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, 1))

        # --- Equations of Motion (Eq 1, 2) ---
        # 1. z2_dot = x2
        A[0, 1] = 1.0

        # 2. z2_dd
        A[1, 0] = -ks / m2
        A[1, 1] = -cs / m2
        A[1, 2] = ks / m2
        A[1, 3] = cs / m2
        A[1, 4] = 1.0 / m2

        # 3. z1_dot = x4
        A[2, 3] = 1.0

        # 4. z1_dd
        A[3, 0] = ks / m1
        A[3, 1] = cs / m1
        A[3, 2] = -(ks + kt) / m1
        A[3, 3] = -cs / m1
        A[3, 4] = -1.0 / m1
        B[3, 0] = kt / m1  # Road input term

        # --- Control Force Dynamics ---
        if control_mode == 'No control':
            # Fc decays rapidly, does not affect the system
            A[4, 4] = -1.0 / 1e-6

        elif control_mode == 'Conventional skyhook':
            # F_ideal = -csh * z2_dot
            A[4, 1] = -csh / tc
            A[4, 4] = -1.0 / tc

        elif control_mode == 'Unsprung negative skyhook':
            # F_ideal = -csh * z1_dot
            A[4, 3] = -csh / tc
            A[4, 4] = -1.0 / tc

        # --- Output Matrix Definition ---
        # Output 1: Sprung mass acceleration z2_dd (for magnitude)
        C_acc = A[1:2, :]
        D_acc = np.zeros((1, 1))

        # Output 2: Sprung mass displacement z2 (for phase)
        C_disp = np.zeros((1, n_states))
        C_disp[0, 0] = 1.0
        D_disp = np.zeros((1, 1))

        # --- Direct Matrix Calculation of Frequency Response ---
        # H(jw) = C * (jwI - A)^-1 * B + D

        n_freqs = len(omega)
        H_acc = np.zeros(n_freqs, dtype=complex)
        H_disp = np.zeros(n_freqs, dtype=complex)

        I = np.eye(n_states)

        for i, w in enumerate(omega):
            s = 1j * w

            try:
                x_state = np.linalg.solve(s * I - A, B)
            except np.linalg.LinAlgError:
                x_state = np.zeros_like(B)

            # Calculate output
            # y_acc = C_acc * x + D_acc * u (u=1)
            H_acc[i] = (C_acc @ x_state + D_acc)[0, 0]

            # y_disp = C_disp * x + D_disp * u (u=1)
            H_disp[i] = (C_disp @ x_state + D_disp)[0, 0]

        # --- Data Post-processing ---

        # 1. Magnitude
        # Paper requires input amplitude |z0| = 1/f
        # H_acc is z2_dd / z0 (unit input)
        # Actual output = H_acc * (1/f)
        mag_val = np.abs(H_acc) * (1.0 / f)
        mag_db = 20 * np.log10(mag_val + 1e-20)  # Avoid log(0)

        # 2. Phase
        # Use displacement transfer function H_disp = z2 / z0
        phase_rad = np.unwrap(np.angle(H_disp))
        phase_deg = np.degrees(phase_rad)

        return mag_db, phase_deg

    # --- 3. Calculate Three Conditions ---
    print("Calculating responses...")
    mag_none, ph_none = get_robust_response('No control')
    mag_sky, ph_sky = get_robust_response('Conventional skyhook')
    mag_neg, ph_neg = get_robust_response('Unsprung negative skyhook')

    # --- 4. Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Top plot: Magnitude
    ax1.semilogx(f, mag_none, 'k--', linewidth=1.5, label='No control')
    ax1.semilogx(f, mag_sky, 'k-', linewidth=1.0, alpha=0.6, label='Conventional skyhook')
    ax1.semilogx(f, mag_neg, 'k-', linewidth=2.5, label='Unsprung reverse skyhook')

    ax1.set_ylabel('Magnitude of Sprung acc. [dB]')
    ax1.grid(True, which="both", linestyle='-', alpha=0.5)
    ax1.legend(loc='lower left')
    ax1.set_xlim(0.5, 20)
    ax1.set_ylim(25, 45)

    # Annotations
    # ax1.text(1.2, 38, 'Better', ha='center', fontsize=10, fontweight='bold')
    ax1.text(5.0, 32, 'Worse', ha='center', fontsize=9)
    ax1.text(5.0, 27, 'Better', ha='center', fontweight='bold')

    # Bottom plot: Phase
    ax2.semilogx(f, ph_none, 'k--', linewidth=1.5, label='No control')
    ax2.semilogx(f, ph_sky, 'k-', linewidth=1.0, alpha=0.6, label='Conventional skyhook')
    ax2.semilogx(f, ph_neg, 'k-', linewidth=2.5, label='Unsprung reverse skyhook')

    ax2.set_ylabel('Phase [deg]')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.grid(True, which="both", linestyle='-', alpha=0.5)

    # Set phase axis range
    ax2.set_ylim(-270, 0)
    ax2.set_yticks(np.arange(-270, 1, 90))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    reproduce_fig5_robust()