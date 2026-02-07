import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 页面基础设置 ---
st.set_page_config(page_title="DYC控制位置响应分析", layout="wide")

st.title("🚗 任意位置侧偏角零化 DYC 控制 - 伯德图分析")
st.markdown("""
本工具基于论文公式 (27)，模拟调节控制点 **x** 的位置对车辆横摆角速度响应 (r/δ) 的影响。
- **x = 0**: 后轮侧偏角零化（基准设计，一阶稳定）
- **x > 0**: 控制点前移（响应变快，阻尼减小）
- **x = lr**: 质心侧偏角零化（理论无滞后，临界稳定）
""")

# --- 侧边栏：参数设置 ---
st.sidebar.header("1. 车辆参数 (基于论文)")
m = st.sidebar.number_input("质量 m [kg]", value=1200.0)
V_kmh = st.sidebar.slider("车速 V [km/h]", 10.0, 120.0, 90.0)
V = V_kmh / 3.6
l = st.sidebar.number_input("轴距 l [m]", value=2.5)
lf = st.sidebar.number_input("质心到前轴 lf [m]", value=1.25)
lr = l - lf
Kf = st.sidebar.number_input("前轮侧偏刚度 Kf [N/rad]", value=30000.0)
Kr = st.sidebar.number_input("后轮侧偏刚度 Kr [N/rad]", value=60000.0)

st.sidebar.markdown("---")
st.sidebar.header("2. 控制参数调节")
# x 的滑动条，范围从后轮 (0) 到略超质心 (lr * 1.1) 以观察不稳定现象
x_limit = lr * 1.1
x = st.sidebar.slider(
    "控制点位置 x [m] (0=后轮, 正值=向前)",
    min_value=0.0,
    max_value=float(x_limit),
    value=0.0,
    step=0.05,
    help="x=0时为后轮零化控制；x=lr时为质心零化控制"
)

# --- 核心计算 (论文公式 27) ---
# 分母公共项 D = 2*Kf*(l-x) - 2*Kr*x + m*V^2
# 注意：论文中可能有近似或符号差异，此处采用严格力学推导形式
D = 2 * Kf * (l - x) - 2 * Kr * x + m * V ** 2

# 1. 稳态增益 K (Steady State Gain)
# r/delta(s=0) = (2 * Kf * V) / D
K_steady = (2 * Kf * V) / D

# 2. 时间常数 T (Time Constant)
# T = (m * V * (lr - x)) / D
T_const = (m * V * (lr - x)) / D

# 3. 建立传递函数系统
# Transfer Function = K_steady / (T_const * s + 1)
if D <= 0:
    st.error("⚠️ 系统参数异常：分母刚度项为负，静态不稳定！")
    system = None
else:
    num = [K_steady]
    den = [T_const, 1]
    system = signal.TransferFunction(num, den)

# --- 绘图与分析 ---
col1, col2 = st.columns([2, 1])

with col1:
    if system:
        # 绘制伯德图
        w = np.logspace(-1, 2, 500)  # 0.1 到 100 rad/s
        w, mag, phase = signal.bode(system, w)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 幅频图
        ax1.semilogx(w, mag, 'b', linewidth=2)
        ax1.set_title(f'Bode Plot (x = {x:.2f} m)', fontsize=14)
        ax1.set_ylabel('Magnitude [dB]', fontsize=12)
        ax1.grid(True, which="both", ls="-", alpha=0.5)

        # 标记转折频率 (Corner Frequency = 1/T)
        if T_const > 0:
            corner_freq = 1.0 / T_const
            ax1.axvline(x=corner_freq, color='r', linestyle='--', label=f'Corner Freq: {corner_freq:.2f} rad/s')
            ax1.legend()

        # 相频图
        ax2.semilogx(w, phase, 'g', linewidth=2)
        ax2.set_ylabel('Phase [deg]', fontsize=12)
        ax2.set_xlabel('Frequency [rad/s]', fontsize=12)
        ax2.grid(True, which="both", ls="-", alpha=0.5)

        st.pyplot(fig)

with col2:
    st.subheader("📊 关键指标")

    # 状态指示
    status_color = "green"
    status_text = "稳定 (Stable)"

    if T_const < 0:
        status_color = "red"
        status_text = "不稳定 (Unstable)! x 过大"
    elif T_const == 0:
        status_color = "orange"
        status_text = "临界 (Zero Lag)"

    st.markdown(f"**系统状态**: :{status_color}[{status_text}]")

    st.metric("控制点位置 x", f"{x:.2f} m")
    st.metric("时间常数 T", f"{T_const:.4f} s", delta_color="inverse")

    # 截止频率 (带宽)
    if T_const > 0:
        bw = 1 / (2 * np.pi * T_const)
        st.metric("带宽频率", f"{bw:.2f} Hz")

    st.info(f"""
    **理论解读：**
    * **当前 x/lr = {x / lr:.2f}**
    * 当 x 接近 {lr:.2f}m (质心) 时，时间常数 T 趋近于 0，响应极快但高频增益极大。
    * 当 x > {lr:.2f}m 时，T 变为负值，系统在物理上失去稳定性（极点进入右半平面）。
    """)