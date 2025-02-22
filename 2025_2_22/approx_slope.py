from ultralytics import YOLO
import cv2
import numpy as np
import os
from scipy.ndimage import label
from sklearn.linear_model import LinearRegression
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import matplotlib.cm as cm  # 用于颜色映射

# 加载 YOLO 模型
model1 = YOLO("../best_non_cleared.pt")

# 创建输出目录（如果不存在）
output_dir = "../test_filtered"
os.makedirs(output_dir, exist_ok=True)

# 获取验证集中的图像文件
files = os.listdir("../dataset/images/val")

for m in files:
    image_path = os.path.join("../dataset/images/val", m)
    or_image = cv2.imread(image_path)       # 原始图像

    results1 = model1.predict(or_image)             # 使用模型进行预测

    result_image1 = results1[0].plot()              # 绘制预测结果
    masks = results1[0].masks.data.cpu().numpy()    # 获取掩码数据
    all_mask = np.max(masks.copy(), axis=0)         # 合并所有掩码

    # 初始化列表，存储斜率、截距和骨架坐标
    slopes = []
    intercepts = []
    skeleton_coords_list = []
    x_fits_list = []
    y_fits_list = []

    composite_mask = np.zeros_like(masks[0], dtype=bool)  # 初始化复合掩码

    # 遍历每个掩码
    for idx, mask in enumerate(masks):
        labeled_mask, num_features = label(mask)          # 标记连通区域
        regions = regionprops(labeled_mask)               # 获取区域属性
        largest_region = max(regions, key=lambda x: x.area)  # 找到最大区域
        largest_component_mask = np.zeros_like(mask)
        largest_component_mask[labeled_mask == largest_region.label] = 1  # 提取最大连通区域

        # 将掩码转换为布尔类型
        mask = largest_component_mask.astype(bool)

        # 排除与复合掩码重叠的区域
        mask = np.logical_and(mask, np.logical_not(composite_mask))

        # 如果掩码为空，跳过此循环
        if not mask.any():
            continue

        # 更新复合掩码
        composite_mask = np.logical_or(composite_mask, mask)

        # 再次标记连通区域
        labeled_mask, num_features = label(mask)
        regions = regionprops(labeled_mask)

        # 找到最大的连通组件
        largest_region = max(regions, key=lambda x: x.area)
        largest_component_mask = np.zeros_like(mask)
        largest_component_mask[labeled_mask == largest_region.label] = 1

        # 如果区域面积小于500，跳过此循环
        if largest_region.area < 500:
            continue

        # 对最大的组件进行骨架化
        skeleton = skeletonize(largest_component_mask)
        skeleton_coords = np.column_stack(np.where(skeleton))  # 获取骨架坐标

        # 对骨架坐标进行线性回归拟合
        X = skeleton_coords[:, 1].reshape(-1, 1)  # x 坐标
        Y = skeleton_coords[:, 0]                 # y 坐标

        linear_model = LinearRegression()
        linear_model.fit(X, Y)
        slope = linear_model.coef_[0]             # 斜率
        intercept = linear_model.intercept_       # 截距

        slopes.append(slope)
        intercepts.append(intercept)
        skeleton_coords_list.append((X, Y))

        x_fit = np.linspace(X.min(), X.max(), 100)
        y_fit = slope * x_fit + intercept
        x_fits_list.append(x_fit)
        y_fits_list.append(y_fit)

    # 绘制骨架和拟合的直线
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(or_image, cv2.COLOR_BGR2RGB), alpha=0.5)
    colors = cm.rainbow(np.linspace(0, 1, len(slopes)))  # 生成颜色映射

    for i in range(len(slopes)):
        X, Y = skeleton_coords_list[i]
        x_fit = x_fits_list[i]
        y_fit = y_fits_list[i]
        color = colors[i]
        plt.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                 label=f'Fit Line {i} (Slope={slopes[i]:.2f})')  # 绘制拟合的直线

    plt.legend()
    plt.axis('off')
    plt.show()

    # 保存结果图像
    output_path = os.path.join(output_dir, m)
    cv2.imwrite(output_path, result_image1)

print("所有图像已处理并保存。")
