import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# time new roman
plt.rcParams['font.family'] = 'Times New Roman'

# Define colors for each model
color_gpt4o = sns.color_palette("husl", 4)[1]
color_ernie = sns.color_palette("husl", 4)[2]
alpha = 1

def plot_acc():

    categories = ['Small\nNodules', 'Non-Small\nNodules', 
             'High-Risk\nNodules', 'Non-High-Risk\nNodules', 
             'Baseline\nscreening', 'Repeat\nscreening', 'Total\nCases'
    ]
    gpt4o_mini_accuracy = [0.938, 0.891, 0.938, 0.927, 0.959, 0.850, 0.928]
    ernie_accuracy = [0.962, 0.895, 0.938, 0.947, 0.980, 0.863, 0.946]

    # Set bar width
    bar_width = 0.15

    # Define spacing for categories
    spacing = [0, 0.4, 1.0, 1.4, 2.0, 2.4, 3.0]
    index = np.array(spacing)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set background color
    fig.patch.set_facecolor('w')
    ax.set_facecolor('w')

    # Create bars
    bar1 = ax.bar(index, gpt4o_mini_accuracy, bar_width, label='GPT-4o-mini', color=color_gpt4o, alpha=alpha)
    bar2 = ax.bar(index + bar_width, ernie_accuracy, bar_width, label='ERNIE-4.0-Turbo-8K', color=color_ernie, alpha=alpha)

    # Add labels and title with k font color
    ax.set_xlabel('Category', color='k')
    ax.set_ylabel('Accuracy', color='k')
    ax.set_title("Comparison of LLMs Accuracy in Determining Pulmonary Nodule Follow-up Intervals", color='k', pad=20, size=15) # Adjust pad value as needed
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories, rotation=0, color='k')
    ax.legend(facecolor='w', edgecolor='k', labelcolor='k')

    # Change tick colors
    ax.tick_params(colors='k')

    # Change color of x and y axis
    ax.spines['bottom'].set_color('k')
    # ax.spines['top'].set_color('k')
    ax.spines['left'].set_color('k')
    # ax.spines['right'].set_color('k')

    # Add numbers on top of bars
    for i, v in enumerate(gpt4o_mini_accuracy):
        ax.text(index[i], v + 0.01, f"{v:.3f}", color='k', ha='center')

    for i, v in enumerate(ernie_accuracy):
        ax.text(index[i] + bar_width, v + 0.01, f"{v:.3f}", color='k', ha='center')

    # Change x and y 粗细
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # 
    ax.set_xlabel('')
    # ax.set_ylabel('')

    # 让title再高一点
    ax.title.set_position([.5, 1.4])

    # legend顶,水平,再上移动
    ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(1.0, 1.05), fontsize=8)

    # Show plot
    plt.tight_layout()
    plt.savefig('model_comparison_accuracy.tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.show()

def radar_chart():
    categories = ['Small Nodules', 'Non-Small Nodules', 
                  'High-Risk Nodules', 'Non-High-Risk Nodules', 
                  'Baseline Screening', 'Repeat Screening', 'Total Cases']
    
    gpt4o_mini_accuracy = [0.938, 0.891, 0.938, 0.927, 0.959, 0.850, 0.928]
    ernie_accuracy = [0.962, 0.895, 0.938, 0.947, 0.980, 0.863, 0.946]
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    
    # 闭合图形
    gpt4o_mini_accuracy += gpt4o_mini_accuracy[:1]
    ernie_accuracy += ernie_accuracy[:1]
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, gpt4o_mini_accuracy, 'o-', linewidth=2, label='GPT-4o-mini')
    ax.fill(angles, gpt4o_mini_accuracy, alpha=0.25)
    
    ax.plot(angles, ernie_accuracy, 'o-', linewidth=2, label='ERNIE')
    ax.fill(angles, ernie_accuracy, alpha=0.25)
    
    ax.set_thetagrids(angles * 180/np.pi, categories)
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Multi-Category Performance Comparison")
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()

def plot_harm_v0():
    categories = [ 'Small\nNodules', 'Non-Small\nNodules', 
                'High-Risk\nNodules', 'Non-High-Risk\nNodules', 
                'Baseline\nscreening', 'Repeat\nscreening', 'Total\nCases'
    ]

    # Calculate harmfulness and round to two decimal places
    gpt4o_mini_harm = [0.992, 0.873, 0.906, 0.967, 0.979, 0.932, 0.965]
    ernie_harm = [0.994, 0.895, 0.969, 0.971, 0.986, 0.935, 0.971]
    gpt4o_mini_harm = [round(1 - g, 3) for g in gpt4o_mini_harm]
    ernie_harm = [round(1 - e, 3) for e in ernie_harm]


    # Set bar width
    bar_width = 0.15

    # Define spacing for categories
    spacing = [0, 0.4, 1.0, 1.4, 2.0, 2.4, 3.0]
    index = np.array(spacing)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set background color
    fig.patch.set_facecolor('w')
    ax.set_facecolor('w')

    # Create bars
    bar1 = ax.bar(index, gpt4o_mini_harm, bar_width, label='GPT-4o-mini', color=color_gpt4o, alpha=alpha)
    bar2 = ax.bar(index + bar_width, ernie_harm, bar_width, label='ERNIE-4.0-Turbo-8K', color=color_ernie, alpha=alpha)

    # Add labels and title with k font color
    ax.set_xlabel('Category', color='k')
    ax.set_ylabel('Accuracy', color='k')
    ax.set_title('Comparison of LLMs Harmfulness', color='k', pad=20, size=15) # Adjust pad value as needed
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories, rotation=0, color='k')
    ax.legend(facecolor='w', edgecolor='k', labelcolor='k')

    # Change tick colors
    ax.tick_params(colors='k')

    # Change color of x and y axis
    ax.spines['bottom'].set_color('k')
    # ax.spines['top'].set_color('k')
    ax.spines['left'].set_color('k')
    # ax.spines['right'].set_color('k')

    # Change x and y 粗细
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # Add numbers on top of bars
    for i, v in enumerate(gpt4o_mini_harm):
        ax.text(index[i], v + 0.001, f"{v:.3f}", color='k', ha='center')

    for i, v in enumerate(ernie_harm):
        ax.text(index[i] + bar_width, v + 0.001, f"{v:.3f}", color='k', ha='center')


    ax.set_xlabel('')
    # ax.set_ylabel('')

    # 让title再高一点
    ax.title.set_position([.5, 1.4])

    # legend顶,水平,再上移动
    ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(1.0, 1.05), fontsize=8)

    # Show plot
    plt.tight_layout()
    plt.savefig('model_comparison_of_harmfulness.tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.show()

def plot_harm():
    categories = ['Small\nNodules', 'Non-Small\nNodules', 
                'High-Risk\nNodules', 'Non-High-Risk\nNodules', 
                'Baseline\nscreening', 'Repeat\nscreening', 'Total\nCases'
    ]

    # 原始harmful值
    gpt4o_non_harm = [0.992, 0.873, 0.906, 0.967, 0.979, 0.932, 0.965]
    ernie_non_harm = [0.994, 0.895, 0.969, 0.971, 0.986, 0.935, 0.971]
    
    # 计算non-harmful值 (1 - harmful)
    gpt4o_harm = [round(1 - h, 3) for h in gpt4o_non_harm]
    ernie_harm = [round(1 - h, 3) for h in ernie_non_harm]
    
    # Set bar width and spacing
    bar_width = 0.35
    spacing = np.arange(len(categories))
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set background color
    fig.patch.set_facecolor('w')
    ax.set_facecolor('w')

    # 创建双组堆叠柱状图
    # GPT-4o-mini
    bar1 = ax.bar(spacing - bar_width/2, gpt4o_non_harm, bar_width, 
                label='GPT-4o-mini (Non-harmful)', color=color_gpt4o, alpha=alpha,
                hatch='')
    bar2 = ax.bar(spacing - bar_width/2, gpt4o_harm, bar_width, 
                bottom=gpt4o_non_harm, color=color_gpt4o, alpha=alpha*0.2,
                label='GPT-4o-mini (Harmful)', hatch='//')

    # ERNIE
    bar3 = ax.bar(spacing + bar_width/2, ernie_non_harm, bar_width,
                label='ERNIE (Non-harmful)', color=color_ernie, alpha=alpha,
                hatch='')
    bar4 = ax.bar(spacing + bar_width/2, ernie_harm, bar_width,
                bottom=ernie_non_harm, color=color_ernie, alpha=alpha*0.2,
                label='ERNIE (Harmful)', hatch='//')


    # Add labels and title
    ax.set_title('Comparison of LLMs Harmfulness\nin Determining Pulmonary Nodule Follow-up Recommendations', color='k', pad=40, size=15) # Adjust pad value as needed
    ax.set_ylabel('Proportion', color='k')
    plt.savefig('model_comparison_of_harmfulness.tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    ax.set_xticks(spacing)
    ax.set_xticklabels(categories, rotation=0, color='k')
    
    # 设置y轴范围为0-1
    ax.set_ylim(0, 1.1)
    
    # Change tick colors and axes properties
    ax.tick_params(colors='k')
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # Add percentage labels on harmful part bars
    for i in range(len(spacing)):
        # GPT-4o-mini harmful
        ax.text(spacing[i] - bar_width/2, 1+0.02, 
                f'{gpt4o_harm[i]:.3f}', ha='center', va='center', color='k')
        # ERNIE harmful
        ax.text(spacing[i] + bar_width/2, 1+0.02,
                f'{ernie_harm[i]:.3f}', ha='center', va='center', color='k')

    # Remove xlabel and adjust title position
    ax.set_xlabel('')
    ax.title.set_position([.5, 0.5]) 

    # Adjust legend position and style
    ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(1.0, 1.12), fontsize=8)

    # Show plot
    plt.tight_layout()
    plt.savefig('model_comparison_of_harmfulness.tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.show()

def plot_risk():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patheffects import withStroke

    # 数据
    labels = ['F1 Score', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    gpt4o_mini = [0.984, 0.999, 1.000, 0.999, 0.969, 1.000]
    ernie_4_turbo = [0.969, 0.998, 0.969, 0.999, 0.969, 0.999]


    # 计算角度
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

    # 数据封闭
    gpt4o_mini = np.concatenate((gpt4o_mini, [gpt4o_mini[0]]))
    ernie_4_turbo = np.concatenate((ernie_4_turbo, [ernie_4_turbo[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    # 创建图形
    fig = plt.figure(figsize=(6, 6), facecolor='w')
    ax = fig.add_subplot(111, polar=True)

    # 设置背景色
    ax.set_facecolor('green')
    # set_facealpha(0.5)
    ax.patch.set_alpha(0.05)

    # 添加主要数据线
    ax.plot(angles, gpt4o_mini, 'o-', linewidth=2, color=color_gpt4o, label='GPT-4o-mini', alpha=1)
    # ax.fill(angles, gpt4o_mini, color=color_gpt4o, alpha=0.25)
    ax.plot(angles, ernie_4_turbo, 'o-', linewidth=2, color=color_ernie, label='ERNIE-4.0-Turbo-8K', alpha=1)

    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    for label, angle in zip(labels, angles):
        # 将弧度转换为度数
        angle_deg = np.degrees(angle)
        
        # 调整角度，使文字垂直于环线
        if 0 <= angle_deg <= 90:
            rotation = angle_deg - 90
        elif 90 < angle_deg <= 180:
            rotation = angle_deg - 90
        elif 180 < angle_deg <= 270:
            rotation = angle_deg + 90
        else:
            rotation = angle_deg + 90
        
        # 计算标签位置
        label_position = 1.05  # 调整这个值可以改变标签距离环线的距离
        
        ax.text(angle, label_position, label, 
                size=13, 
                color='k',
                rotation=rotation,
                rotation_mode='anchor',  # 确保旋转是围绕文本的锚点
                ha='center',
                va='center',
                path_effects=[withStroke(linewidth=3, foreground='w')])
        
    # 外圈标签（两个模型的性能数值）
    for label, score1, score2, angle in zip(labels, gpt4o_mini, ernie_4_turbo, angles):
        angle_deg = np.degrees(angle)
        
        if 0 <= angle_deg <= 90:
            rotation = angle_deg - 90
        elif 90 < angle_deg <= 180:
            rotation = angle_deg - 90
        elif 180 < angle_deg <= 270:
            rotation = angle_deg + 90
        else:
            rotation = angle_deg + 90
        
        # 第一个模型的数值（红色）
        ax.text(angle, 1.12, f'{score1:.3f}', 
                size=11,
                color=color_gpt4o,  # 红色
                rotation=rotation,
                rotation_mode='anchor',
                ha='center',
                va='center',
                path_effects=[withStroke(linewidth=2, foreground='w')])
        
        # 第二个模型的数值（蓝色）
        ax.text(angle, 1.2, f'{score2:.3f}', 
                size=11,
                color=color_ernie,  # 蓝色
                rotation=rotation,
                rotation_mode='anchor',
                ha='center',
                va='center',
                path_effects=[withStroke(linewidth=2, foreground='w')])
        

    # 隐藏刻度标签
    ax.set_xticklabels([])

    # 设置网格样式
    ax.grid(True, color='k', alpha=0.9, linestyle='-', linewidth=1)

    # 设置雷达图的范围
    ax.set_ylim(0, 1.005)

    # 设置标签颜色
    ax.tick_params(axis='y', colors='k')

    # 添加数值标签
    # for angle, gpt_val, ernie_val in zip(angles[:-1], gpt4o_mini[:-1], ernie_4_turbo[:-1]):
    #     ax.text(angle, gpt_val + 0.03, f'{gpt_val:.2f}', color='#00ffff', ha='center', va='center')
    #     ax.text(angle, ernie_val - 0.03, f'{ernie_val:.2f}', color='#ff3366', ha='center', va='center')

    # 添加标题
    plt.title('Comparison of LLMs Performances in\nHigh-Risk Pulmonary Nodule Detection', size=15, color='k', pad=35,
            path_effects=[withStroke(linewidth=3, foreground='w')])

    # 添加图例
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), 
                    facecolor='w', edgecolor='k', labelcolor='k', fontsize=8)
    for text in legend.get_texts():
        text.set_color('k')

    # 调整布局
    plt.tight_layout()

    # 保存图形
    plt.savefig('model_comparison_of_risk.tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})

    # 显示图形
    plt.show()


if __name__ == '__main__':
    plot_acc()
    # radar_chart()
    plot_harm()
    plot_risk()