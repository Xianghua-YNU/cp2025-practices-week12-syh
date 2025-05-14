import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def breit_wigner(E, Er, Gamma, fr):
    """Breit-Wigner共振公式"""
    numerator = fr
    denominator = (E - Er)**2 + (Gamma**2 / 4)
    return numerator / denominator

def calculate_chi_squared(energy, cross_section, errors, popt):
    """计算卡方值和约化卡方值"""
    model = breit_wigner(energy, *popt)
    residuals = (cross_section - model) / errors
    chi_squared = np.sum(residuals**2)
    degrees_of_freedom = len(energy) - len(popt)
    reduced_chi = chi_squared / degrees_of_freedom
    return chi_squared, reduced_chi

def fit_without_errors(energy, cross_section, p0=None):
    """不考虑误差的Breit-Wigner拟合"""
    if p0 is None:
        p0 = [75.0, 50.0, 10000.0]  # 初始猜测值
    
    popt, pcov = curve_fit(
        breit_wigner, 
        energy, 
        cross_section, 
        p0=p0
    )
    
    return popt, pcov

def fit_with_errors(energy, cross_section, errors, p0=None):
    """考虑误差的Breit-Wigner拟合"""
    if p0 is None:
        p0 = [75.0, 50.0, 10000.0]  # 初始猜测值
    
    popt, pcov = curve_fit(
        breit_wigner, 
        energy, 
        cross_section, 
        p0=p0,
        sigma=errors,
        absolute_sigma=True
    )
    
    return popt, pcov

def plot_fit_results(energy, cross_section, errors, popt, pcov, title, ax=None):
    """绘制拟合结果"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制数据点和误差棒
    ax.errorbar(
        energy, 
        cross_section, 
        yerr=errors, 
        fmt='o',
        color='blue', 
        markersize=6, 
        ecolor='gray',
        elinewidth=1, 
        capsize=3, 
        label='实验数据'
    )
    
    # 绘制拟合曲线
    E_fit = np.linspace(min(energy), max(energy), 500)
    cross_section_fit = breit_wigner(E_fit, *popt)
    ax.plot(E_fit, cross_section_fit, '-', color='red', linewidth=2.5, label='拟合曲线')
    
    # 计算参数误差和置信区间
    Er, Gamma, fr = popt
    Er_std = np.sqrt(pcov[0, 0])
    Gamma_std = np.sqrt(pcov[1, 1])
    fr_std = np.sqrt(pcov[2, 2])
    
    # 计算卡方值
    chi_sq, red_chi = calculate_chi_squared(energy, cross_section, errors, popt)
    
    # 添加参数信息
    param_text = (
        f'$E_r$ = {Er:.1f} ± {1.96*Er_std:.1f} MeV (95% CI)\n'
        f'$\Gamma$ = {Gamma:.1f} ± {1.96*Gamma_std:.1f} MeV (95% CI)\n'
        f'$f_r$ = {fr:.0f} ± {1.96*fr_std:.0f} (95% CI)\n'
        f'$\chi^2$/dof = {red_chi:.2f}'
    )
    
    ax.text(
        0.05, 0.95, 
        param_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
        fontsize=10
    )
    
    # 设置图表属性
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('能量 (MeV)', fontsize=12)
    ax.set_ylabel('截面 (mb)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    return ax.figure

def compare_fit_results(popt1, pcov1, popt2, pcov2, labels=['不考虑误差', '考虑误差']):
    """比较两种拟合结果的参数差异"""
    param_names = ['$E_r$ (MeV)', '$\Gamma$ (MeV)', '$f_r$']
    param_units = ['MeV', 'MeV', '']
    
    print("\n拟合结果比较:")
    for i, name in enumerate(param_names):
        val1 = popt1[i]
        val2 = popt2[i]
        err1 = 1.96 * np.sqrt(pcov1[i, i])  # 95%置信区间
        err2 = 1.96 * np.sqrt(pcov2[i, i])
        
        diff = abs(val1 - val2)
        rel_diff = diff / ((val1 + val2) / 2) * 100 if (val1 + val2) != 0 else 0
        
        print(f"{name}:")
        print(f"  {labels[0]}: {val1:.2f} ± {err1:.2f} {param_units[i]}")
        print(f"  {labels[1]}: {val2:.2f} ± {err2:.2f} {param_units[i]}")
        print(f"  差异: {diff:.2f} {param_units[i]} ({rel_diff:.2f}%)")
        print()
    
    # 计算卡方值
    chi_sq1, red_chi1 = calculate_chi_squared(energy, cross_section, errors, popt1)
    chi_sq2, red_chi2 = calculate_chi_squared(energy, cross_section, errors, popt2)
    
    print(f"拟合优度比较:")
    print(f"  {labels[0]}: χ²/dof = {red_chi1:.2f}")
    print(f"  {labels[1]}: χ²/dof = {red_chi2:.2f}")
    
    return {
        'parameters': {
            param_names[0]: [val1, val2],
            param_names[1]: [val1, val2],
            param_names[2]: [val1, val2]
        },
        'reduced_chi': [red_chi1, red_chi2]
    }

def main():
    """主函数：执行完整的拟合分析流程"""
    # 实验数据
    energy = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
    cross_section = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])
    errors = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])
    
    # 任务1：不考虑误差的拟合
    popt1, pcov1 = fit_without_errors(energy, cross_section)
    fig1 = plot_fit_results(
        energy, cross_section, errors, popt1, pcov1,
        'Breit-Wigner拟合（不考虑误差）'
    )
    
    # 任务2：考虑误差的拟合
    popt2, pcov2 = fit_with_errors(energy, cross_section, errors)
    fig2 = plot_fit_results(
        energy, cross_section, errors, popt2, pcov2,
        'Breit-Wigner拟合（考虑误差）'
    )
    
    # 显示图表
    plt.tight_layout()
    plt.show()
    
    # 任务3：结果比较
    compare_fit_results(popt1, pcov1, popt2, pcov2)
    
    # 误差处理对拟合结果的影响分析
    print("\n误差处理对拟合结果的影响分析:")
    print(f"共振能量Er差异: {abs(popt1[0]-popt2[0]):.2f} MeV")
    print(f"共振宽度Γ差异: {abs(popt1[1]-popt2[1]):.2f} MeV")
    print(f"共振强度fr差异: {abs(popt1[2]-popt2[2]):.2f}")
    
    # 评估拟合优度
    chi_sq1, red_chi1 = calculate_chi_squared(energy, cross_section, errors, popt1)
    chi_sq2, red_chi2 = calculate_chi_squared(energy, cross_section, errors, popt2)
    
    print("\n拟合优度评估:")
    print(f"不考虑误差的卡方值/自由度: {red_chi1:.2f}")
    print(f"考虑误差的卡方值/自由度: {red_chi2:.2f}")
    
    if red_chi2 < red_chi1:
        print("考虑误差的拟合结果更优（卡方值/自由度更小）")
    else:
        print("不考虑误差的拟合结果更优（卡方值/自由度更小）")

if __name__ == "__main__":
    main()
