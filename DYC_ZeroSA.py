import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Page Configuration ---
st.set_page_config(page_title="DYC Response Analysis", layout="wide")

st.title("🚗 DYC with Zero Slip Angle at Any Location - Bode Plot Analysis")
st.markdown("""
This tool simulates the effect of adjusting the control point **x** on the vehicle yaw rate response (r/δ), based on equation (27) from the paper.
- **x = 0**: Zero slip angle at Rear Wheels (Baseline, 1st-order stable)
- **x > 0**: Control point moves forward (Faster response, reduced damping)
- **x = lr**: Zero slip angle at C.G. (Theoretically zero lag, critically stable)
""")

# --- Sidebar: Parameter Settings ---
st.sidebar.header("1. Vehicle Parameters")
m = st.sidebar.number_input("Mass m [kg]", value=1200.0)
V_kmh = st.sidebar.slider("Velocity V [km/h]", 10.0, 120.0, 90.0)
V = V_kmh / 3.6
l = st.sidebar.number_input("Wheelbase l [m]", value=2.5)
lf = st.sidebar.number_input("Distance C.G. to Front Axle lf [m]", value=1.25)
lr = l - lf
Kf = st.sidebar.number_input("Front Cornering Stiffness Kf [N/rad]", value=30000.0)
Kr = st.sidebar.number_input("Rear Cornering Stiffness Kr [N/rad]", value=60000.0)

st.sidebar.markdown("---")
st.sidebar.header("2. Control Tuning")
# Slider for x, ranging from rear wheels (0) to slightly past C.G. (lr * 1.1) to observe instability
x_limit = lr * 1.1
x = st.sidebar.slider(
    "Control Point Position x [m] (0=Rear, Positive=Forward)",
    min_value=0.0,
    max_value=float(x_limit),
    value=0.0,
    step=0.05,
    help="x=0: Zero slip at rear wheels; x=lr: Zero slip at C.G."
)

# --- Core Calculation (Paper Equation 27) ---
# Common denominator term D = 2*Kf*(l-x) - 2*Kr*x + m*V^2
# Note: Using strict mechanical derivation
D = 2 * Kf * (l - x) - 2 * Kr * x + m * V ** 2

# 1. Steady State Gain K
# r/delta(s=0) = (2 * Kf * V) / D
K_steady = (2 * Kf * V) / D

# 2. Time Constant T
# T = (m * V * (lr - x)) / D
T_const = (m * V * (lr - x)) / D

# 3. Build Transfer Function System
# Transfer Function = K_steady / (T_const * s + 1)
if D <= 0:
    st.error("⚠️ Parameter Error: Denominator stiffness term is negative. Statically Unstable!")
    system = None
else:
    num = [K_steady]
    den = [T_const, 1]
    system = signal.TransferFunction(num, den)

# --- Plotting & Analysis ---
col1, col2 = st.columns([2, 1])

with col1:
    if system:
        # Generate Bode Plot
        w = np.logspace(-1, 2, 500)  # 0.1 to 100 rad/s
        w, mag, phase = signal.bode(system, w)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Magnitude Plot
        ax1.semilogx(w, mag, 'b', linewidth=2)
        ax1.set_title(f'Bode Plot (x = {x:.2f} m)', fontsize=14)
        ax1.set_ylabel('Magnitude [dB]', fontsize=12)
        ax1.grid(True, which="both", ls="-", alpha=0.5)

        # Mark Corner Frequency (1/T)
        if T_const > 0:
            corner_freq = 1.0 / T_const
            ax1.axvline(x=corner_freq, color='r', linestyle='--', label=f'Corner Freq: {corner_freq:.2f} rad/s')
            ax1.legend()

        # Phase Plot
        ax2.semilogx(w, phase, 'g', linewidth=2)
        ax2.set_ylabel('Phase [deg]', fontsize=12)
        ax2.set_xlabel('Frequency [rad/s]', fontsize=12)
        ax2.grid(True, which="both", ls="-", alpha=0.5)

        st.pyplot(fig)

with col2:
    st.subheader("📊 Key Metrics")

    # Status Indicator
    status_color = "green"
    status_text = "Stable"

    if T_const < 0:
        status_color = "red"
        status_text = "Unstable! (x too large)"
    elif T_const == 0:
        status_color = "orange"
        status_text = "Critical (Zero Lag)"

    st.markdown(f"**System Status**: :{status_color}[{status_text}]")

    st.metric("Control Point x", f"{x:.2f} m")
    st.metric("Time Constant T", f"{T_const:.4f} s", delta_color="inverse")

    # Bandwidth Frequency
    if T_const > 0:
        bw = 1 / (2 * np.pi * T_const)
        st.metric("Bandwidth", f"{bw:.2f} Hz")

    st.info(f"""
    **Theoretical Insight:**
    * **Current ratio x/lr = {x / lr:.2f}**
    * When x approaches {lr:.2f}m (C.G.), the Time Constant T approaches 0. Response becomes extremely fast, but high-frequency gain maximizes.
    * When x > {lr:.2f}m, T becomes negative. The system physically loses stability (Pole moves to the Right Half Plane).
    """)