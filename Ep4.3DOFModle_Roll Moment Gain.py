import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def plot_mx_bode_full_with_peak():
    # ==========================================
    # 1. 参数定义
    # ==========================================
    m = 2310.0
    V_kmh = 100.0
    V = V_kmh / 3.6

    Iz = 4796.0
    Ix = 889.0

    lf = 1.48
    lr = 1.65

    h_sf = 0.46
    h_sr = 0.41
    h = 0.43

    Kx = 12462.0
    Cx = 8512.0

    K_sf = 65921.0
    K_sr = 129012.0

    # ==========================================
    # 2. 状态空间构建 (A, B 矩阵)
    # ==========================================
    # 辅助系数
    Cf_beta, Cf_r, Cf_p = 1, lf / V, h_sf / V
    Cr_beta, Cr_r, Cr_p = 1, lr / V, h_sr / V

    # --- A 矩阵 ---
    a11 = (2 * (-K_sf * Cf_beta) + 2 * (-K_sr * Cr_beta)) / (m * V)
    a12 = ((2 * (-K_sf * Cf_r) + 2 * (-K_sr * Cr_r)) / (m * V)) - 1
    a13 = (2 * (-K_sf * Cf_p) + 2 * (-K_sr * Cr_p)) / (m * V)
    a14 = 0

    a21 = (2 * lf * (-K_sf * Cf_beta) - 2 * lr * (-K_sr * Cr_beta)) / Iz
    a22 = (2 * lf * (-K_sf * Cf_r) - 2 * lr * (-K_sr * Cr_r)) / Iz
    a23 = (2 * lf * (-K_sf * Cf_p) - 2 * lr * (-K_sr * Cr_p)) / Iz
    a24 = 0

    Mx_beta = 2 * h_sf * (-K_sf * Cf_beta) + 2 * h_sr * (-K_sr * Cr_beta)
    Mx_r = 2 * h_sf * (-K_sf * Cf_r) + 2 * h_sr * (-K_sr * Cr_r)
    Mx_p = 2 * h_sf * (-K_sf * Cf_p) + 2 * h_sr * (-K_sr * Cr_p)

    a31 = Mx_beta / Ix
    a32 = Mx_r / Ix
    a33 = (Mx_p - Cx) / Ix
    a34 = -Kx / Ix

    a41, a42, a43, a44 = 0, 0, 1, 0

    A = np.array([
        [a11, a12, a13, a14],
        [a21, a22, a23, a24],
        [a31, a32, a33, a34],
        [a41, a42, a43, a44]
    ])

    # --- B 矩阵 ---
    Ff_u = K_sf
    b1 = (2 * Ff_u) / (m * V)
    b2 = (2 * lf * Ff_u) / Iz
    b3 = (2 * h_sf * Ff_u) / Ix
    b4 = 0
    B = np.array([[b1], [b2], [b3], [b4]])

    # ==========================================
    # 3. 输出方程构建 (C, D 矩阵 -> Output: Mx)
    # ==========================================
    D_val = 2 * h_sf * K_sf
    D = np.array([D_val])

    C = np.array([Mx_beta, Mx_r, Mx_p, 0.0])

    # ==========================================
    # 4. 计算频域响应
    # ==========================================
    sys = signal.StateSpace(A, B, C, D)

    w = np.logspace(-1, 2.1, 1000)  # rad/s
    w, mag, phase = signal.bode(sys, w)
    freq_hz = w / (2 * np.pi)

    # 找到峰值
    peak_idx = np.argmax(mag)
    peak_freq = freq_hz[peak_idx]
    peak_mag = mag[peak_idx]

    # ==========================================
    # 5. 绘图 (双子图布局)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- 上图：幅值 (Magnitude) ---
    ax1.semilogx(freq_hz, mag, linewidth=2, color='#00008B')  # 深蓝色
    ax1.set_title(r'Roll Moment Gain ($M_x / \delta$) Bode Plot', fontsize=14, pad=15)
    ax1.set_ylabel('Magnitude (dB)', fontsize=12)
    ax1.grid(True, which="both", ls="-", alpha=0.4)

    # 标注直流增益
    dc_gain = mag[0]
    ax1.annotate(f'Low Freq: {dc_gain:.1f} dB',
                 xy=(freq_hz[0], dc_gain), xytext=(freq_hz[0] * 1.5, dc_gain - 10),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # 标注峰值
    ax1.plot(peak_freq, peak_mag, 'ro')  # 红点标记
    ax1.annotate(f'Peak: {peak_mag:.1f} dB @ {peak_freq:.2f} Hz',
                 xy=(peak_freq, peak_mag),
                 xytext=(peak_freq * 0.5, peak_mag - 15),  # 调整位置避免遮挡
                 arrowprops=dict(facecolor='red', shrink=0.05))

    # --- 下图：相位 (Phase) ---
    ax2.semilogx(freq_hz, phase, linewidth=2, color='#8B0000')  # 深红色
    ax2.set_ylabel('Phase (degrees)', fontsize=12)
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.grid(True, which="both", ls="-", alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_mx_bode_full_with_peak()