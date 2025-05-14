import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 实验数据
ENERGY = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # MeV
CROSS_SECTION = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])  # mb
ERROR = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])  # mb

def lagrange_interpolation(x_points, x_data, y_data):
    """向量化实现拉格朗日插值"""
    x_points = np.atleast_1d(x_points)
    result = np.zeros_like(x_points)
    
    for i, xi in enumerate(x_data):
        # 计算拉格朗日基函数
        mask = np.ones(len(x_data), dtype=bool)
        mask[i] = False
        others = x_data[mask]
        
        numerator = np.prod(x_points[:, np.newaxis] - others, axis=1)
        denominator = np.prod(xi - others)
        
        result += y_data[i] * numerator / denominator
    
    return result

def cubic_spline_interpolation(x_points, x_data, y_data):
    """使用scipy实现三次样条插值，设置自然边界条件"""
    spline = interp1d(x_data, y_data, kind='cubic', 
                     bounds_error=False, fill_value='extrapolate')
    return spline(x_points)

def find_peak(x_data, y_data):
    """精确寻找峰值位置和半高全宽"""
    # 找到峰值索引
    peak_idx = np.argmax(y_data)
    peak_x = x_data[peak_idx]
    peak_y = y_data[peak_idx]
    
    # 计算半高
    half_max = peak_y / 2
    
    # 使用线性插值提高FWHM精度
    left_idx = np.argmin(np.abs(y_data[:peak_idx] - half_max))
    right_idx = peak_idx + np.argmin(np.abs(y_data[peak_idx:] - half_max))
    
    # 线性插值计算更准确的半高位置
    if left_idx > 0 and left_idx < len(y_data) - 1:
        x0, x1 = x_data[left_idx-1], x_data[left_idx]
        y0, y1 = y_data[left_idx-1], y_data[left_idx]
        left_x = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)
    else:
        left_x = x_data[left_idx]
        
    if right_idx > 0 and right_idx < len(y_data) - 1:
        x0, x1 = x_data[right_idx-1], x_data[right_idx]
        y0, y1 = y_data[right_idx-1], y_data[right_idx]
        right_x = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)
    else:
        right_x = x_data[right_idx]
    
    fwhm = right_x - left_x
    return peak_x, fwhm

def analyze_resonance():
    """主分析函数：计算插值、分析峰值并生成图表"""
    # 生成高密度插值点
    x_dense = np.linspace(min(ENERGY), max(ENERGY), 1000)
    
    # 计算插值结果
    lagrange_values = lagrange_interpolation(x_dense, ENERGY, CROSS_SECTION)
    spline_values = cubic_spline_interpolation(x_dense, ENERGY, CROSS_SECTION)
    
    # 分析峰值
    lagrange_peak, lagrange_fwhm = find_peak(x_dense, lagrange_values)
    spline_peak, spline_fwhm = find_peak(x_dense, spline_values)
    
    # 绘制对比图
    plt.figure(figsize=(14, 7))
    
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    
    # 绘制原始数据点
    plt.errorbar(ENERGY, CROSS_SECTION, yerr=ERROR, fmt='o', 
                color='black', label='实验数据', capsize=5, alpha=0.8)
    
    # 绘制插值曲线
    plt.plot(x_dense, lagrange_values, '-', color='blue', 
            label='拉格朗日插值', linewidth=2)
    plt.plot(x_dense, spline_values, '--', color='orange', 
            label='三次样条插值', linewidth=2)
    
    # 标记峰值位置
    plt.axvline(lagrange_peak, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(spline_peak, color='orange', linestyle=':', alpha=0.5)
    
    # 标记半高全宽
    for method, peak, fwhm, color in zip(
        ['拉格朗日', '三次样条'], 
        [lagrange_peak, spline_peak], 
        [lagrange_fwhm, spline_fwhm], 
        ['blue', 'orange']
    ):
        left = peak - fwhm/2
        right = peak + fwhm/2
        plt.hlines(max(CROSS_SECTION)/2, left, right, color=color, linestyle='--')
        plt.text((left + right)/2, max(CROSS_SECTION)/2 + 3, 
                f'{method} FWHM: {fwhm:.2f} MeV', 
                ha='center', color=color, fontweight='bold')
    
    # 添加峰值和FWHM信息
    plt.annotate(f'拉格朗日峰值: {lagrange_peak:.2f} MeV',
                xy=(lagrange_peak, max(lagrange_values)),
                xytext=(lagrange_peak+10, max(lagrange_values)*0.9),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8))
    
    plt.annotate(f'样条峰值: {spline_peak:.2f} MeV',
                xy=(spline_peak, max(spline_values)),
                xytext=(spline_peak+10, max(spline_values)*0.8),
                arrowprops=dict(facecolor='orange', shrink=0.05, width=1.5, headwidth=8))
    
    # 设置图表属性
    plt.title('中子共振散射截面数据分析', fontsize=16)
    plt.xlabel('能量 (MeV)', fontsize=14)
    plt.ylabel('截面 (mb)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 打印分析结果
    print("\n==== 共振峰分析结果 ====")
    print(f"拉格朗日插值 - 共振峰位置: {lagrange_peak:.2f} MeV, 半高全宽: {lagrange_fwhm:.2f} MeV")
    print(f"三次样条插值 - 共振峰位置: {spline_peak:.2f} MeV, 半高全宽: {spline_fwhm:.2f} MeV")
    print(f"峰值位置差异: {abs(lagrange_peak - spline_peak):.2f} MeV")
    print(f"半高全宽差异: {abs(lagrange_fwhm - spline_fwhm):.2f} MeV")
    
    plt.show()

if __name__ == "__main__":
    analyze_resonance()
