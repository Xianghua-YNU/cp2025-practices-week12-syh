import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def load_bacterial_data(file_path):
    """从文件加载细菌生长数据，自动检测分隔符"""
    delimiters = [',', '\t', ' ', None]
    for delimiter in delimiters:
        try:
            data = np.loadtxt(file_path, delimiter=delimiter)
            return data[:, 0], data[:, 1]
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
    使用curve_fit拟合模型
    
    返回:
        tuple: (拟合参数, 协方差矩阵)
    """
    # 只有当提供了边界时才传递bounds参数
    if bounds is not None:
        popt, pcov = curve_fit(model_func, t, data, p0=p0, bounds=bounds)
    else:
        popt, pcov = curve_fit(model_func, t, data, p0=p0)
    
    return popt, pcov  # 只返回popt和pcov，与测试代码兼容

def plot_results(t, data, model_func, popt, title):
    """
    绘制实验数据与拟合曲线
    
    参数:
        t (numpy.ndarray): 时间数据
        data (numpy.ndarray): 实验数据
        model_func (function): 模型函数
        popt (numpy.ndarray): 拟合参数
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(t, data, color='blue', alpha=0.6, label='实验数据')
    
    t_fit = np.linspace(min(t), max(t), 500)
    plt.plot(t_fit, model_func(t_fit, *popt), 'r-', label='拟合曲线')
    
    # 添加参数标注
    param_text = get_param_text(model_func, popt)
    plt.figtext(0.15, 0.02, param_text, ha='left', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('时间')
    plt.ylabel('酶活性')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def get_param_text(model_func, popt):
    """生成参数文本用于图表标注"""
    if model_func == V_model:
        return f'拟合参数: τ = {popt[0]:.3f}'
    elif model_func == W_model:
        return f'拟合参数: A = {popt[0]:.3f}, τ = {popt[1]:.3f}'
    return ''

def calculate_errors(pcov):
    """计算拟合参数的标准误差"""
    return np.sqrt(np.diag(pcov))

def main():
    """主函数：加载数据、拟合模型、绘制结果"""
    data_dir = "/Users/lixh/Library/CloudStorage/OneDrive-个人/Code/cp2025-InterpolateFit/细菌生长实验数据拟合"
    
    # 加载数据
    print("加载实验数据...")
    t_V, V_data = load_bacterial_data(f"{data_dir}/g149novickA.txt")
    t_W, W_data = load_bacterial_data(f"{data_dir}/g149novickB.txt")
    
    # 拟合V(t)模型
    print("拟合V(t)模型...")
    popt_V, pcov_V = fit_model(t_V, V_data, V_model, p0=[1.0], bounds=(0, np.inf))
    perr_V = calculate_errors(pcov_V)
    print(f"V(t)模型拟合参数: τ = {popt_V[0]:.3f} ± {perr_V[0]:.3f}")
    
    # 拟合W(t)模型
    print("拟合W(t)模型...")
    popt_W, pcov_W = fit_model(t_W, W_data, W_model, p0=[1.0, 1.0], 
                             bounds=([0, 0], [np.inf, np.inf]))
    perr_W = calculate_errors(pcov_W)
    print(f"W(t)模型拟合参数: A = {popt_W[0]:.3f} ± {perr_W[0]:.3f}, τ = {popt_W[1]:.3f} ± {perr_W[1]:.3f}")
    
    # 绘制结果
    plot_results(t_V, V_data, V_model, popt_V, 'V(t)模型拟合结果')
    plot_results(t_W, W_data, W_model, popt_W, 'W(t)模型拟合结果')

if __name__ == "__main__":
    main()
