# MetaLogic YouTube Channel Resources

This repository contains the simulation code and engineering resources for the **MetaLogic** YouTube channel.

---

## Episode 3: 2-DOF Bicycle Model Frequency Response
**File:** `Ep3.BicycleModle_YawRateGain.py`

### Description
This script simulates the lateral dynamics of a vehicle using a linear **2-Degree-of-Freedom (2-DOF) Bicycle Model**. It utilizes State-Space representation to perform a frequency response analysis.

### Features
- **State-Space Modeling:** Constructs system matrices ($A, B, C, D$) based on vehicle parameters (mass, cornering stiffness, etc.).
- **Frequency Response:** Calculates and plots Bode diagrams using `scipy.signal`.
- **Key Metrics:**
  - **Yaw Rate Gain ($r/\delta$):** Analyzes the vehicle's steering response characteristics.
  - **Sideslip Angle Gain ($\beta/\delta$):** Analyzes the vehicle's stability characteristics.

### Dependencies
- `numpy`
- `matplotlib`
- `scipy`

---

## Episode 2: Reverse Skyhook Simulation
**File:** `MetaLogic_E2_ReverseSkyhook.py`

### Description
Simulates the frequency response of a vehicle suspension system to demonstrate the effectiveness of **Reverse Skyhook** control (academically known as *Unsprung Negative Skyhook*).

## 📄 Reference Paper
* **Title:** Improvement of Ride Comfort by Unsprung Negative Skyhook Damper Control Using In-Wheel Motors
* **Authors:** Etsuo Katsuyama, Ayana Omae (Toyota Motor Corporation)
* **Context:** The paper proposes a control method to mitigate the ride comfort worsening in the mid-frequency range (3-9Hz) caused by the increased unsprung mass in In-Wheel Motor (IWM) vehicles.
* **Note:** In the MetaLogic video, this method is referred to as **"Reverse Skyhook"**.

## 📊 What This Code Does
The script `MetaLogic_E2_ReverseSkyhook.py` performs a frequency response analysis on a Quarter-Car Model (2-DOF) comparing three scenarios:
1.  **No Control:** Passive suspension baseline.
2.  **Conventional Skyhook:** Traditional skyhook control (shows worsening in mid-frequencies due to delay).
3.  **Reverse Skyhook (Proposed):** The method discussed in the video, which applies force proportional to unsprung velocity to suppress vibration.

It generates two plots:
* **Magnitude:** Sprung mass acceleration response (dB).
* **Phase:** Displacement transfer function phase (deg).

## 💡 Key Technical Highlight: Robust State-Space Solver
This is **not** a standard `scipy.signal.bode` implementation.
Standard libraries often fail with vehicle dynamics parameters (e.g., stiff tire stiffness vs. small mass) when converting State-Space models to Transfer Functions, leading to `BadCoefficients` or numerical instability.

**This code implements a robust direct solver:**
```python
# Solves H(jw) = C * (jwI - A)^-1 * B + D directly using matrix inversion
x_state = np.linalg.solve(s * I - A, B)
