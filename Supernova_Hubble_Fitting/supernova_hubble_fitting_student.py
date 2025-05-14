import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_supernova_data(file_path):
    """
    从文件加载超新星数据，自动处理任意数量的注释行
    
    参数:
        file_path (str): 数据文件路径
        
    返回:
        tuple: (红移数据, 距离模数数据, 距离模数误差)
    """
    # 使用np.genfromtxt处理不规则注释行
    data = np.genfromtxt(file_path, comments='#')
    return data[:, 0], data[:, 1], data[:, 2]


def hubble_model(z, H0):
    """
    基础哈勃模型：距离模数与红移的关系
    
    参数:
        z (array_like): 红移值
        H0 (float): 哈勃常数 (km/s/Mpc)
        
    返回:
        array_like: 计算的距离模数
    """
    # 光速 (km/s)
    c = 299792.458
    return 5 * np.log10(c * z / H0) + 25


def hubble_model_with_deceleration(z, H0, a1):
    """
    包含减速参数的哈勃模型
    
    参数:
        z (array_like): 红移值
        H0 (float): 哈勃常数 (km/s/Mpc)
        a1 (float): 减速参数 (对应q0=1-a1)
        
    返回:
        array_like: 计算的距离模数
    """
    c = 299792.458
    correction_term = 1 + 0.5 * (1 - a1) * z
    return 5 * np.log10(c * z / H0 * correction_term) + 25


def hubble_fit(z, mu, mu_err):
    """
    加权最小二乘法拟合哈勃常数
    
    参数:
        z (array_like): 红移数据
        mu (array_like): 距离模数数据
        mu_err (array_like): 距离模数误差
        
    返回:
        tuple: (哈勃常数, 哈勃常数误差)
    """
    # 设置合理的初始猜测和参数边界
    popt, pcov = curve_fit(
        hubble_model, 
        z, 
        mu, 
        sigma=mu_err,
        p0=[70.0],
        bounds=(50.0, 100.0),
        absolute_sigma=True
    )
    return popt[0], np.sqrt(pcov[0, 0])


def hubble_fit_with_deceleration(z, mu, mu_err):
    """
    加权最小二乘法拟合哈勃常数和减速参数
    
    参数:
        z (array_like): 红移数据
        mu (array_like): 距离模数数据
        mu_err (array_like): 距离模数误差
        
    返回:
        tuple: (哈勃常数, 哈勃常数误差, 减速参数, 减速参数误差)
    """
    popt, pcov = curve_fit(
        hubble_model_with_deceleration,
        z,
        mu,
        sigma=mu_err,
        p0=[70.0, 1.0],
        bounds=([50.0, -5.0], [100.0, 5.0]),
        absolute_sigma=True
    )
    return popt[0], np.sqrt(pcov[0, 0]), popt[1], np.sqrt(pcov[1, 1])


def plot_hubble_diagram(z, mu, mu_err, H0):
    """
    绘制哈勃图：距离模数 vs 红移
    
    参数:
        z (array_like): 红移数据
        mu (array_like): 距离模数数据
        mu_err (array_like): 距离模数误差
        H0 (float): 拟合的哈勃常数
        
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    plt.figure(figsize=(10, 6), dpi=300)
    
    # 绘制数据点和误差棒
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', color='dodgerblue', 
                 ecolor='gray', elinewidth=1, capsize=3, label='观测数据')
    
    # 生成拟合曲线
    z_fit = np.linspace(min(z), max(z), 500)
    mu_fit = hubble_model(z_fit, H0)
    
    # 绘制拟合曲线
    plt.plot(z_fit, mu_fit, 'r-', linewidth=2, 
             label=f'拟合曲线 ($H_0$ = {H0:.2f} km/s/Mpc)')
    
    # 美化图表
    plt.xlabel('红移 z', fontsize=12)
    plt.ylabel('距离模数 μ', fontsize=12)
    plt.title('哈勃图：距离模数 vs 红移', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    return plt.gcf()


def plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1):
    """
    绘制包含减速参数的哈勃图
    
    参数:
        z (array_like): 红移数据
        mu (array_like): 距离模数数据
        mu_err (array_like): 距离模数误差
        H0 (float): 拟合的哈勃常数
        a1 (float): 拟合的减速参数
        
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    plt.figure(figsize=(10, 6), dpi=300)
    
    # 绘制数据点和误差棒
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', color='dodgerblue', 
                 ecolor='gray', elinewidth=1, capsize=3, label='观测数据')
    
    # 生成拟合曲线
    z_fit = np.linspace(min(z), max(z), 500)
    mu_fit = hubble_model_with_deceleration(z_fit, H0, a1)
    
    # 绘制拟合曲线
    plt.plot(z_fit, mu_fit, 'r-', linewidth=2, 
             label=f'拟合曲线 ($H_0$ = {H0:.2f}, $a_1$ = {a1:.3f})')
    
    # 美化图表
    plt.xlabel('红移 z', fontsize=12)
    plt.ylabel('距离模数 μ', fontsize=12)
    plt.title('哈勃图（包含减速参数）', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    return plt.gcf()


if __name__ == "__main__":
    # 加载数据
    z, mu, mu_err = load_supernova_data("supernova_data.txt")
    
    # 基础模型拟合
    H0, H0_err = hubble_fit(z, mu, mu_err)
    print(f"基础模型: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    
    # 绘制基础模型图
    plot_hubble_diagram(z, mu, mu_err, H0)
    plt.savefig("hubble_diagram.png", dpi=300)
    plt.show()
    
    # 带减速参数的模型拟合
    H0, H0_err, a1, a1_err = hubble_fit_with_deceleration(z, mu, mu_err)
    print(f"带减速参数模型: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    print(f"减速参数 a1 = {a1:.3f} ± {a1_err:.3f}")
    
    # 绘制带减速参数的模型图
    plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1)
    plt.savefig("hubble_diagram_with_deceleration.png", dpi=300)
    plt.show()
