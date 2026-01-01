import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd

# 1. 设置全局参数（关键步骤！）
"""配置学术论文级别的matplotlib样式"""
plt.rcParams.update({
    # 字体设置
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'mathtext.fontset': 'stix',
    
    # 图表尺寸
    'figure.figsize': (6, 4),  # 适合单栏
    'figure.dpi': 600,
    'savefig.dpi': 600,
    
    # 坐标轴设置
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    
    # 刻度设置
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    
    # 图例设置
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    
    # 线条设置
    'lines.linewidth': 2,
    'lines.markersize': 6,
    
    # 保存设置
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.format': 'pdf',  # 矢量图质量最好
})

avg_loss = [0.2464, 0.2128, 0.2099, 0.1787, 0.1682, 0.1617, 0.1591, 0.1571, 
            0.1532, 0.1516, 0.1520, 0.1521, 0.1511, 0.1515, 0.1527, 0.1533, 
            0.1512, 0.1516, 0.1501, 0.1494, 0.1484, 0.1471, 0.1459, 0.1456, 
            0.1460, 0.1455, 0.1459, 0.1465, 0.1463, 0.1449, 0.1447, 0.1445, 
            0.1443, 0.1435, 0.1458, 0.1451, 0.1443, 0.1438, 0.1425, 0.1422, 
            0.1422, 0.1419, 0.1417, 0.1416, 0.1416, 0.1416, 0.1415, 0.1416, 
            0.1415, 0.1416]

avg_mfm = [0.1035, 0.0722, 0.0701, 0.0390, 0.0287, 0.0221, 0.0197, 0.0176, 
            0.0139, 0.0124, 0.0123, 0.0130, 0.0120, 0.0123, 0.0137, 0.0141, 
            0.0122, 0.0127, 0.0112, 0.0105, 0.0094, 0.0083, 0.0070, 0.0066, 
            0.0073, 0.0067, 0.0071, 0.0078, 0.0075, 0.0061, 0.0059, 0.0058, 
            0.0055, 0.0048, 0.0071, 0.0064, 0.0056, 0.0051, 0.0037, 0.0035, 
            0.0035, 0.0031, 0.0030, 0.0029, 0.0029, 0.0029, 0.0028, 0.0029, 
            0.0027, 0.0029]

avg_top = [0.7148, 0.7031, 0.6989, 0.6988, 0.6977, 0.6981, 0.6970, 0.6973, 
            0.6964, 0.6956, 0.6986, 0.6958, 0.6956, 0.6956, 0.6952, 0.6958, 
            0.6948, 0.6944, 0.6947, 0.6944, 0.6949, 0.6937, 0.6943, 0.6948, 
            0.6934, 0.6939, 0.6938, 0.6938, 0.6941, 0.6940, 0.6941, 0.6935, 
            0.6941, 0.6937, 0.6934, 0.6937, 0.6934, 0.6937, 0.6937, 0.6936, 
            0.6934, 0.6937, 0.6935, 0.6935, 0.6937, 0.6935, 0.6936, 0.6935, 
            0.6937, 0.6935]

# 四种策略的Top-1准确率
random_top1 = [0.1271, 0.2148, 0.2814, 0.3752, 0.4526, 0.5269, 0.5894, 0.6294, 
                0.6868, 0.6745, 0.7545, 0.7555, 0.7565, 0.7601, 0.7796, 0.7883, 
                0.7570, 0.7929, 0.8042, 0.7791, 0.7904, 0.8011, 0.8057, 0.8175, 
                0.7806, 0.7622, 0.7837, 0.7847, 0.7786, 0.8063]

ssl_top1 = [0.0636, 0.0984, 0.1205, 0.1502, 0.1609, 0.1999, 0.2219, 0.2414, 
            0.2363, 0.2599, 0.2927, 0.3137, 0.3716, 0.3742, 0.4152, 0.4152, 
            0.4546, 0.4910, 0.5095, 0.5049, 0.5392, 0.5700, 0.5674, 0.5987, 
            0.6115, 0.6284, 0.6386, 0.6371, 0.5992, 0.6397]

two_stage_top1 = [0.0871, 0.1184, 0.1251, 0.1353, 0.1399, 0.1415, 0.1456, 0.1574, 
                    0.1553, 0.1553, 0.1963, 0.2322, 0.2583, 0.2747, 0.3147, 0.3552, 
                    0.3808, 0.4264, 0.4552, 0.4823, 0.4967, 0.5331, 0.5515, 0.5930, 
                    0.6212, 0.6187, 0.6427, 0.6715, 0.6581, 0.6643]

linear_top1 = [0.0359, 0.0615, 0.0712, 0.0774, 0.0789, 0.0779, 0.0825, 0.0871, 
                0.0897, 0.0933, 0.1015, 0.0917, 0.1025, 0.0994, 0.1040, 0.1040, 
                0.1046, 0.1199, 0.1230, 0.1225, 0.1107, 0.1123, 0.1205, 0.1297, 
                0.1230, 0.1251, 0.1215, 0.1348, 0.1322, 0.1276]
def plot_ssl_training_curves():
    """绘制SSL预训练的学习曲线"""
    # 你的数据
    epochs = np.arange(1, 51)
    """更准确地绘制SSL损失动态"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 1. 三个损失对比
    ax1 = axes[0, 0]
    ax1.plot(epochs, avg_loss, 'k-', linewidth=2, label='Total Loss')
    ax1.plot(epochs, avg_mfm, 'b-', linewidth=1.5, label='MFM Loss')
    ax1.plot(epochs, avg_top, 'r-', linewidth=1.5, label='TOP Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('(a) Three Loss Components in SSL Pre-training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 损失占比变化
    ax2 = axes[0, 1]
    mfm_ratio = np.array(avg_mfm) / np.array(avg_loss) * 100
    top_ratio = np.array(avg_top) / np.array(avg_loss) * 100
    
    ax2.plot(epochs, mfm_ratio, 'b-', linewidth=2, label='MFM % of Total')
    ax2.plot(epochs, top_ratio, 'r-', linewidth=2, label='TOP % of Total')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('(b) Loss Component Proportions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 损失变化率（导数）
    ax3 = axes[1, 0]
    mfm_grad = np.gradient(avg_mfm)
    top_grad = np.gradient(avg_top)
    
    ax3.plot(epochs[1:], mfm_grad[1:], 'b-', linewidth=2, label='MFM Gradient')
    ax3.plot(epochs[1:], top_grad[1:], 'r-', linewidth=2, label='TOP Gradient')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Gradient')
    ax3.set_title('(c) Loss Optimization Speed')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 收敛状态分析
    ax4 = axes[1, 1]
    # 计算每个epoch的损失变化百分比
    mfm_change = np.abs(np.diff(avg_mfm) / avg_mfm[:-1]) * 100
    top_change = np.abs(np.diff(avg_top) / avg_top[:-1]) * 100
    
    ax4.plot(epochs[1:], mfm_change, 'b-', label='MFM Change %')
    ax4.plot(epochs[1:], top_change, 'r-', label='TOP Change %')
    ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='1% Threshold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Relative Change (%)')
    ax4.set_title('(d) Convergence Dynamics')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ssl_loss_dynamics_corrected.pdf', dpi=600)
    plt.show()




def plot_finetune_comparison():
    """绘制四种微调策略的对比图"""
    # 准备数据
    epochs = np.arange(1, 31)
    
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 1. Top-1准确率对比
    ax1 = axes[0, 0]
    ax1.plot(epochs, random_top1, 'b-', linewidth=2, label='Random Init')
    ax1.plot(epochs, ssl_top1, 'r-', linewidth=2, label='SSL Finetune')
    ax1.plot(epochs, two_stage_top1, 'g-', linewidth=2, label='Two-stage')
    ax1.plot(epochs, linear_top1, 'k-', linewidth=2, label='Linear Probe')
    
    # 标记两阶段微调的解冻点
    ax1.axvline(x=10, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(10.5, 0.2, 'Unfreeze\nBackbone', fontsize=9, color='orange')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Top-1 Accuracy')
    ax1.set_title('(a) Top-1 Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Top-5准确率对比
    ax2 = axes[0, 1]
    # 这里需要你的Top-5数据，假设已有类似数据
    # ax2.plot(...)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Top-5 Accuracy')
    ax2.set_title('(b) Top-5 Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 性能提升分析（柱状图）
    ax3 = axes[1, 0]
    methods = ['Random', 'SSL', 'Two-stage', 'Linear']
    final_acc = [random_top1[-1], ssl_top1[-1], two_stage_top1[-1], linear_top1[-1]]
    colors = ['blue', 'red', 'green', 'black']
    
    bars = ax3.bar(methods, final_acc, color=colors, alpha=0.7)
    ax3.set_ylabel('Final Top-1 Accuracy')
    ax3.set_title('(c) Final Performance Comparison')
    
    # 在柱子上添加数值
    for bar, val in zip(bars, final_acc):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. 收敛速度分析（达到特定准确率所需的epoch数）
    ax4 = axes[1, 1]
    thresholds = [0.5, 0.6, 0.7, 0.8]
    random_epochs = [6, 11, 15, 24]  # 示例数据
    ssl_epochs = [22, 27, 30, None]  # 示例数据
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    ax4.bar(x - width/2, random_epochs, width, label='Random', alpha=0.7)
    ax4.bar(x + width/2, ssl_epochs, width, label='SSL', alpha=0.7)
    
    ax4.set_xlabel('Accuracy Threshold')
    ax4.set_ylabel('Epochs to Reach')
    ax4.set_title('(d) Convergence Speed Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{t:.1f}' for t in thresholds])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('finetune_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.show()




def plot_statistical_analysis():
    """绘制带统计检验的图表"""
    import scipy.stats as stats
    # 准备数据
    epochs = np.arange(1, 31)
    # 示例：不同epoch区间的性能分布
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 1. 箱线图对比
    data = [random_top1[:10], random_top1[10:20], random_top1[20:]]
    positions = [1, 2, 3]
    labels = ['Epoch 1-10', 'Epoch 11-20', 'Epoch 21-30']
    
    bp = axes[0].boxplot(data, positions=positions, widths=0.6, 
                         patch_artist=True, showmeans=True)
    
    # 美化箱线图
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_ylabel('Top-1 Accuracy')
    axes[0].set_title('(a) Performance Distribution by Stage')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. 散点图与回归线
    axes[1].scatter(epochs, random_top1, alpha=0.6, label='Random')
    axes[1].scatter(epochs, ssl_top1, alpha=0.6, label='SSL')
    
    # 添加回归线
    from scipy.stats import linregress
    slope_r, intercept_r, _, _, _ = linregress(epochs, random_top1)
    slope_s, intercept_s, _, _, _ = linregress(epochs, ssl_top1)
    
    axes[1].plot(epochs, intercept_r + slope_r * epochs, 'b-', linewidth=2)
    axes[1].plot(epochs, intercept_s + slope_s * epochs, 'r-', linewidth=2)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Top-1 Accuracy')
    axes[1].set_title('(b) Learning Trends with Regression')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 相关性热图
    # 创建不同指标的相关性矩阵
    metrics = pd.DataFrame({
        'Loss': avg_loss,
        'MFM': avg_mfm,
        'Top1': avg_top
    })
    
    corr_matrix = metrics.corr()
    im = axes[2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # 添加数值
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            text = axes[2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=10)
    
    axes[2].set_xticks(range(len(corr_matrix.columns)))
    axes[2].set_yticks(range(len(corr_matrix.columns)))
    axes[2].set_xticklabels(corr_matrix.columns)
    axes[2].set_yticklabels(corr_matrix.columns)
    axes[2].set_title('(c) Correlation Matrix')
    
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig('statistical_analysis.pdf', dpi=600)
    plt.show()


plot_ssl_training_curves()
# plot_finetune_comparison()
# plot_statistical_analysis()