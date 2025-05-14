import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# 设置字体为黑体
plt.rcParams['font.family'] = 'SimHei'

def load_bacterial_data(file_name):
    """从当前目录加载细菌生长数据，自动检测分隔符"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_name}")
    
    # 尝试常见的分隔符
    delimiters = [',', '\t', ' ', None]
    for delimiter in delimiters:
        try:
            data = np.loadtxt(file_path, delimiter=delimiter)
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("数据格式不正确，需要至少两列数据")
            return data[:, 0], data[:, 1]  # 返回时间和活性
        except Exception as e:
            continue  # 尝试下一个分隔符
    
    raise ValueError(f"无法解析文件格式: {file_name}")

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
    print("\n正在从当前目录加载实验数据...")
    
    try:
        # 直接从当前目录加载文件
        t_V, V_data = load_bacterial_data("g149novickA.txt")
        t_W, W_data = load_bacterial_data("g149novickB.txt")
    except FileNotFoundError as e:
        print(f"数据加载失败: {e}")
        print("请确保g149novickA.txt和g149novickB.txt文件与本程序在同一目录下")
        return
    except ValueError as e:
        print(f"数据格式错误: {e}")
        return
    
    # 拟合V(t)模型
    print("\n拟合V(t)模型...")
    popt_V, pcov_V = fit_model(t_V, V_data, V_model, p0=[1.0], bounds=(0, np.inf))
    perr_V = calculate_errors(pcov_V)
    print(f"V(t)模型拟合参数: τ = {popt_V[0]:.3f} ± {perr_V[0]:.3f}")
    
    # 拟合W(t)模型
    print("\n拟合W(t)模型...")
    popt_W, pcov_W = fit_model(t_W, W_data, W_model, p0=[1.0, 1.0], 
                             bounds=([0, 0], [np.inf, np.inf]))
    perr_W = calculate_errors(pcov_W)
    print(f"W(t)模型拟合参数: A = {popt_W[0]:.3f} ± {perr_W[0]:.3f}, τ = {popt_W[1]:.3f} ± {perr_W[1]:.3f}")
    
    # 绘制结果
    plot_results(t_V, V_data, V_model, popt_V, 'V(t)模型拟合结果')
    plot_results(t_W, W_data, W_model, popt_W, 'W(t)模型拟合结果')

if __name__ == "__main__":
    main()
