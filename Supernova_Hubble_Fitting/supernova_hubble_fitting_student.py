import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def load_supernova_data(file_path):
    """
    从文件加载超新星数据，指定UTF-8编码并跳过注释行
    
    参数:
        file_path (str): 数据文件路径
        
    返回:
        tuple: (红移数据, 距离模数数据, 距离模数误差)
    """
    # 使用UTF-8编码读取文件，跳过所有以#开头的注释行
    data = np.genfromtxt(file_path, comments='#', encoding='utf-8')
    return data[:, 0], data[:, 1], data[:, 2]


def hubble_model(z, H0):
    """基础哈勃模型"""
    c = 299792.458
    return 5 * np.log10(c * z / H0) + 25


def hubble_model_with_deceleration(z, H0, a1):
    """带减速参数的哈勃模型"""
    c = 299792.458
    correction = 1 + 0.5 * (1 - a1) * z
    return 5 * np.log10(c * z * correction / H0) + 25


def hubble_fit(z, mu, mu_err):
    """加权最小二乘拟合哈勃常数"""
    popt, pcov = curve_fit(
        hubble_model, z, mu, sigma=mu_err,
        p0=[70.0], bounds=(50, 100), absolute_sigma=True
    )
    return popt[0], np.sqrt(pcov[0, 0])


def hubble_fit_with_deceleration(z, mu, mu_err):
    """拟合哈勃常数和减速参数"""
    popt, pcov = curve_fit(
        hubble_model_with_deceleration, z, mu, sigma=mu_err,
        p0=[70.0, 1.0], bounds=([50, -5], [100, 5]), absolute_sigma=True
    )
    return popt[0], np.sqrt(pcov[0, 0]), popt[1], np.sqrt(pcov[1, 1])


def plot_hubble_diagram(z, mu, mu_err, H0):
    """绘制哈勃图"""
    plt.figure(figsize=(10, 6), dpi=300)
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', color='#1f77b4', ecolor='gray', 
                 capsize=2, label='观测数据', alpha=0.8)
    
    z_fit = np.linspace(z.min(), z.max(), 500)
    plt.plot(z_fit, hubble_model(z_fit, H0), 'r-', linewidth=2, 
             label=f'拟合曲线 ($H_0$={H0:.2f} km/s/Mpc)')
    
    plt.xlabel('红移 z', fontsize=12)
    plt.ylabel('距离模数 μ', fontsize=12)
    plt.title('哈勃图：距离模数 vs 红移', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1):
    """绘制带减速参数的哈勃图"""
    plt.figure(figsize=(10, 6), dpi=300)
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', color='#ff7f0e', ecolor='gray', 
                 capsize=2, label='观测数据', alpha=0.8)
    
    z_fit = np.linspace(z.min(), z.max(), 500)
    plt.plot(z_fit, hubble_model_with_deceleration(z_fit, H0, a1), 'g-', linewidth=2, 
             label=f'拟合曲线 ($H_0$={H0:.2f}, $a_1$={a1:.2f})')
    
    plt.xlabel('红移 z', fontsize=12)
    plt.ylabel('距离模数 μ', fontsize=12)
    plt.title('哈勃图（含减速参数）', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    # 确保文件路径正确，此处使用用户提供的数据集路径
    data_file = "supernova_data.txt"
    
    try:
        z, mu, mu_err = load_supernova_data(data_file)
    except UnicodeDecodeError:
        # 备用方案：尝试Latin-1编码（极少情况需要）
        data = np.genfromtxt(data_file, comments='#', encoding='latin-1')
        z, mu, mu_err = data[:, 0], data[:, 1], data[:, 2]
    
    # 基础模型拟合
    H0, H0_err = hubble_fit(z, mu, mu_err)
    print(f"基础模型哈勃常数: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    fig = plot_hubble_diagram(z, mu, mu_err, H0)
    plt.savefig("hubble_basic.png", dpi=300)
    
    # 带减速参数模型拟合
    H0_dec, H0_err_dec, a1, a1_err = hubble_fit_with_deceleration(z, mu, mu_err)
    print(f"带减速参数模型: H0 = {H0_dec:.2f} ± {H0_err_dec:.2f}, a1 = {a1:.2f} ± {a1_err:.2f}")
    fig = plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0_dec, a1)
    plt.savefig("hubble_deceleration.png", dpi=300)
    
    plt.show()
