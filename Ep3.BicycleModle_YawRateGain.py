import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Define Parameters
m = 1500          # Mass (kg)
Kf = 55000        # Front tire cornering stiffness (N/rad) - Single tire
Kr = 60000        # Rear tire cornering stiffness (N/rad) - Single tire
lf = 1.1          # Distance from CG to front axle (m)
lr = 1.6          # Distance from CG to rear axle (m)
Iz = 2500         # Yaw moment of inertia (kg m^2)
V_kph = 100       # Vehicle velocity (kph)

# Unit Conversion
V = V_kph / 3.6   # Convert to m/s

# Calculate total axle cornering stiffness (assuming 2 tires per axle)
Cf = 2 * Kf
Cr = 2 * Kr

# 2. Construct State Space Matrices
# State vector x = [beta, r].T (Sideslip angle, Yaw rate)
# Input vector u = [delta] (Front wheel steering angle)
# System equation: x_dot = A * x + B * u

# Derive matrix elements from 2-DOF dynamic equations:
# m*V*(beta_dot + r) = Fyf + Fyr
# Iz*r_dot = lf*Fyf - lr*Fyr
# Fyf = -Cf * (beta + lf*r/V - delta)
# Fyr = -Cr * (beta - lr*r/V)

# Elements of Matrix A
a11 = -(Cf + Cr) / (m * V)
a12 = (Cr * lr - Cf * lf) / (m * V**2) - 1
a21 = (Cr * lr - Cf * lf) / Iz
a22 = -(Cf * lf**2 + Cr * lr**2) / (Iz * V)

A = [[a11, a12],
     [a21, a22]]

# Elements of Matrix B
b1 = Cf / (m * V)
b2 = (lf * Cf) / Iz

B = [[b1],
     [b2]]

# Matrices C and D (Define Outputs)
# Output y1 = r (yaw rate) -> Corresponds to the 2nd element of state vector
# Output y2 = beta (sideslip) -> Corresponds to the 1st element of state vector

C_r = [[0, 1]]      # Output: r
C_beta = [[1, 0]]   # Output: beta
D = [[0]]

# 3. Create System Models
# Define state space systems using scipy.signal
sys_r = signal.StateSpace(A, B, C_r, D)
sys_beta = signal.StateSpace(A, B, C_beta, D)

# 4. Generate Bode Plot Data
# Define frequency range: 0.1 to 100 rad/s
w = np.logspace(-1, 2, 500)
w_r, mag_r, phase_r = signal.bode(sys_r, w)
w_b, mag_b, phase_b = signal.bode(sys_beta, w)

# 5. Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# --- Left Column: Yaw Rate Gain ---
# Magnitude Plot
axs[0, 0].semilogx(w_r, mag_r)
axs[0, 0].set_title(r'Bode Magnitude: Yaw Rate ($r/\delta$)')
axs[0, 0].set_ylabel('Magnitude (dB)')
axs[0, 0].set_xlabel('Frequency (rad/s)')
axs[0, 0].grid(True, which="both", ls="-")

# Phase Plot
axs[1, 0].semilogx(w_r, phase_r)
axs[1, 0].set_title(r'Bode Phase: Yaw Rate ($r/\delta$)')
axs[1, 0].set_ylabel('Phase (degrees)')
axs[1, 0].set_xlabel('Frequency (rad/s)')
axs[1, 0].grid(True, which="both", ls="-")

# --- Right Column: Sideslip Angle Gain ---
# Magnitude Plot
axs[0, 1].semilogx(w_b, mag_b, color='orange')
axs[0, 1].set_title(r'Bode Magnitude: Sideslip ($\beta/\delta$)')
axs[0, 1].set_ylabel('Magnitude (dB)')
axs[0, 1].set_xlabel('Frequency (rad/s)')
axs[0, 1].grid(True, which="both", ls="-")

# Phase Plot
axs[1, 1].semilogx(w_b, phase_b, color='orange')
axs[1, 1].set_title(r'Bode Phase: Sideslip ($\beta/\delta$)')
axs[1, 1].set_ylabel('Phase (degrees)')
axs[1, 1].set_xlabel('Frequency (rad/s)')
axs[1, 1].grid(True, which="both", ls="-")

plt.show()