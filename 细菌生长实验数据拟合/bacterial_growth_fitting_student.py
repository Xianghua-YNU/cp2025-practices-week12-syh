import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def load_bacterial_data(file_path):
    """从文件加载细菌生长数据，自动检测分隔符"""
    # 尝试常见的分隔符
    delimiters = [',', '\t', ' ', None]
    for delimiter in delimiters:
        try:
            data = np.loadtxt(file_path, delimiter=delimiter)
            return data[:, 0], data[:, 1]  # 返回时间和活性
        except:
            continue
    raise ValueError(f"无法解析文件: {file_path}")

def V_model(t, tau):
    """V(t)模型：描述诱导分子TMG的渗透过程"""
    return 1 - np.exp(-t / tau)

def W_model(t, A, tau):
    """W(t)模型：描述β-半乳糖苷酶的合成过程"""
    return A * (np.exp(-t / tau) - 1 + t / tau)

def fit_model(t, data, model_func, p0, bounds=None):
    """
    使用curve_fit拟合模型并计算参数误差
    
    返回:
        tuple: (拟合参数, 参数误差, 协方差矩阵)
    """
    popt, pcov = curve_fit(model_func, t, data, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))  # 参数的标准误差
    return popt, perr, pcov

def plot_results(t, data, model_func, popt, perr, title):
    """
    绘制实验数据与拟合曲线，包含参数标注和拟合优度
    
    参数:
        perr: 参数误差数组
    """
    plt.figure(figsize=(12, 7))
    
    # 绘制实验数据点
    plt.scatter(t, data, color='dodgerblue', alpha=0.7, 
                edgecolors='k', s=40, label='实验数据')
    
    # 生成密集的拟合曲线
    t_fit = np.linspace(min(t), max(t), 500)
    plt.plot(t_fit, model_func(t_fit, *popt), 'r-', linewidth=2.5,
             label='拟合曲线')
    
    # 计算并显示拟合优度
    residuals = data - model_func(t, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 添加参数标注和拟合优度
    param_text = generate_param_text(model_func, popt, perr, r_squared)
    plt.figtext(0.15, 0.02, param_text, ha='left', 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 美化图表
    plt.title(title, fontsize=15)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('酶活性', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部的文本留出空间
    plt.show()

def generate_param_text(model_func, popt, perr, r_squared):
    """生成包含参数和拟合优度的文本"""
    if model_func == V_model:
        param_names = ['τ']
    elif model_func == W_model:
        param_names = ['A', 'τ']
    else:
        return ""
    
    param_text = "拟合参数:\n"
    for name, value, error in zip(param_names, popt, perr):
        param_text += f"{name} = {value:.3f} ± {error:.3f}\n"
    param_text += f"R² = {r_squared:.4f}"
    return param_text

def main():
    """主函数：加载数据、拟合模型、绘制结果"""
    # 请替换为实际数据目录
    data_dir = "/Users/lixh/Library/CloudStorage/OneDrive-个人/Code/cp2025-InterpolateFit/细菌生长实验数据拟合"
    
    # 加载数据
    print("正在加载实验数据...")
    t_V, V_data = load_bacterial_data(f"{data_dir}/g149novickA.txt")
    t_W, W_data = load_bacterial_data(f"{data_dir}/g149novickB.txt")
    
    # 拟合V(t)模型 (添加边界约束确保tau为正)
    print("\n拟合V(t)模型...")
    popt_V, perr_V, pcov_V = fit_model(t_V, V_data, V_model, p0=[1.0], bounds=(0, np.inf))
    print(f"V(t)模型拟合参数: τ = {popt_V[0]:.3f} ± {perr_V[0]:.3f}")
    
    # 拟合W(t)模型 (确保A和tau为正)
    print("\n拟合W(t)模型...")
    popt_W, perr_W, pcov_W = fit_model(t_W, W_data, W_model, p0=[1.0, 1.0], 
                                     bounds=([0, 0], [np.inf, np.inf]))
    print(f"W(t)模型拟合参数: A = {popt_W[0]:.3f} ± {perr_W[0]:.3f}, τ = {popt_W[1]:.3f} ± {perr_W[1]:.3f}")
    
    # 绘制结果
    plot_results(t_V, V_data, V_model, popt_V, perr_V, 'V(t)模型拟合结果')
    plot_results(t_W, W_data, W_model, popt_W, perr_W, 'W(t)模型拟合结果')

if __name__ == "__main__":
    main()
