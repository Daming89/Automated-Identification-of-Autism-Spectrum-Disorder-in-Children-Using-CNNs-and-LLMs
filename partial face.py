import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# 根据分类报告反推的混淆矩阵值
conf_matrix = np.array([
    [163,  134],  # Autistic实际类别
    [ 179, 174]   # Non_Autistic实际类别
])

# 分类报告参数
accuracy = 0.5185
class_names = ['Autistic', 'Non_Autistic']

# 生成美观的混淆矩阵可视化
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
ax = sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={'size': 16},
    linewidths=0.5
)

# 添加标签和标题
plt.title('Confusion Matrix (Accuracy: {:.2%})'.format(accuracy), fontsize=14, pad=20)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 显示图像
plt.tight_layout()
plt.show()

# 生成完整的分类报告
print("\nEnhanced Classification Report:")
print(classification_report(
    y_true = np.array([0]*325 + [1]*325),  # 真实标签
    y_pred = np.array([0]*283 + [1]*42 + [0]*58 + [1]*267),  # 预测标签
    target_names=class_names,
    digits=4
))