#!/usr/bin/env python3
from ultralytics import YOLO
import cv2
import numpy as np
import os
from scipy.ndimage import label
from skimage.morphology import skeletonize
from skimage.measure import regionprops
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button

# -------------------- 配置 -------------------- #
model1 = YOLO("./best_non_cleared.pt")
images_dir = "./dataset/images/val"
output_dir = "./test_filtered"
os.makedirs(output_dir, exist_ok=True)
files = sorted(os.listdir(images_dir))
current_img_idx = 0

# -------------------- 全局图形和轴 -------------------- #
fig = plt.figure(figsize=(10, 10))
# 主图像轴 – 强制范围与图像完全匹配。
image_ax = fig.add_axes([0.1, 0.15, 0.75, 0.75])
# 滑块轴（固定位置）
slider_dist_ax = fig.add_axes([0.2, 0.05, 0.65, 0.03])
slider_time_ax = fig.add_axes([0.9, 0.2, 0.03, 0.65])
# 按钮轴（固定位置）
btn_prev_ax = fig.add_axes([0.1, 0.93, 0.1, 0.05])
btn_next_ax = fig.add_axes([0.25, 0.93, 0.1, 0.05])
btn_prev = Button(btn_prev_ax, "Previous")
btn_next = Button(btn_next_ax, "Next")

# 全局滑块变量（将为每张图像重新创建）
slider_dist = None
slider_time = None

# 全局绘制的元素和标记
vline = None  # 垂直虚线
hline = None  # 水平虚线
intersection_markers = []  # 垂直交点标记
horizontal_markers = []  # 水平交点标记

# 交点计数的全局文本对象
vert_count_text = None
horiz_count_text = None

# 存储当前图像拟合线数据的全局变量
current_fitted_data = None

# -------------------- 辅助函数 -------------------- #
def process_image(image):
    """
    使用 YOLO 处理图像：提取掩码，应用骨架化，
    并对每个检测到的掩码通过线性回归拟合一条线。
    返回一个字典，包含拟合线参数和采样点。
    """
    results1 = model1.predict(image)
    masks = results1[0].masks.data.cpu().numpy()  # shape: (num_masks, H, W)

    slopes = []
    intercepts = []
    x_range_list = []
    x_fits_list = []
    y_fits_list = []

    composite_mask = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        labeled_mask, _ = label(mask)
        regions = regionprops(labeled_mask)
        if not regions:
            continue
        # 选择最大的连通区域。
        largest = max(regions, key=lambda r: r.area)
        comp_mask = np.zeros_like(mask)
        comp_mask[labeled_mask == largest.label] = 1
        mask_bool = comp_mask.astype(bool)
        mask_bool = np.logical_and(mask_bool, np.logical_not(composite_mask))
        if not mask_bool.any():
            continue
        composite_mask = np.logical_or(composite_mask, mask_bool)
        labeled_mask, _ = label(mask_bool)
        regions = regionprops(labeled_mask)
        if not regions:
            continue
        largest = max(regions, key=lambda r: r.area)
        comp_mask = np.zeros_like(mask_bool)
        comp_mask[labeled_mask == largest.label] = 1
        if largest.area < 500:
            continue
        skeleton = skeletonize(comp_mask)
        coords = np.column_stack(np.where(skeleton))
        if coords.size == 0:
            continue
        # 使用列作为 x，行作为 y。
        X = coords[:, 1].reshape(-1, 1)
        Y = coords[:, 0]
        reg = LinearRegression().fit(X, Y)
        slopes.append(reg.coef_[0])
        intercepts.append(reg.intercept_)
        x_min = X.min()
        x_max = X.max()
        x_range_list.append((x_min, x_max))
        xs = np.linspace(x_min, x_max, 100)
        ys = reg.coef_[0] * xs + reg.intercept_
        x_fits_list.append(xs)
        y_fits_list.append(ys)

    colors = cm.rainbow(np.linspace(0, 1, len(slopes)))
    return {"slopes": slopes,
            "intercepts": intercepts,
            "x_range_list": x_range_list,
            "x_fits_list": x_fits_list,
            "y_fits_list": y_fits_list,
            "colors": colors}

def extend_line_to_bounds(m, b, width, height):
    """
    给定一条线 y = m*x + b 和图像边界 [0, width] x [0, height]，
    计算该无限线与这些边界的两个交点。
    返回一个元组：((x0, y0), (x1, y1))。
    """
    pts = []
    # 与左边界（x=0）的交点
    y_left = b
    if 0 <= y_left <= height:
        pts.append((0, y_left))
    # 与右边界（x=width）的交点
    y_right = m * width + b
    if 0 <= y_right <= height:
        pts.append((width, y_right))
    # 与下边界（y=0）的交点
    if abs(m) > 1e-6:
        x_bottom = -b / m
        if 0 <= x_bottom <= width:
            pts.append((x_bottom, 0))
    # 与上边界（y=height）的交点
    if abs(m) > 1e-6:
        x_top = (height - b) / m
        if 0 <= x_top <= width:
            pts.append((x_top, height))
    pts = list(dict.fromkeys(pts))  # 移除重复点
    if len(pts) < 2:
        pts = [(0, b), (width, m * width + b)]
    elif len(pts) > 2:
        # 选择距离最远的点对
        max_dist = -1
        best_pair = None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dist = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (pts[i], pts[j])
        return best_pair
    return pts[0], pts[1]

# --- 更新使用无限线方程的交点函数 ---
def update_intersections_vertical(x_val, fitted_data):
    """
    对于给定的垂直虚线 x = x_val，计算与每条拟合线（y = m*x + b）的无限线交点，
    并在有效时标记它们。
    同时更新垂直交点计数标签。
    """
    global intersection_markers, image_ax, vert_count_text
    for marker in intersection_markers:
        try:
            marker.remove()
        except NotImplementedError:
            marker.set_visible(False)
    intersection_markers.clear()
    ybottom, ytop = image_ax.get_ylim()
    for i, (slope, intercept) in enumerate(zip(fitted_data["slopes"], fitted_data["intercepts"])):
        y_val = slope * x_val + intercept
        if ybottom <= y_val <= ytop:
            mkr, = image_ax.plot(x_val, y_val, 'o',
                                 color=fitted_data["colors"][i],
                                 markersize=8, markeredgecolor='k')
            intersection_markers.append(mkr)
    # 更新垂直交点计数标签。
    if vert_count_text is None:
        vert_count_text = image_ax.text(0.98, 0.95,
                                        f"Vertical Intersections: {len(intersection_markers)}",
                                        transform=image_ax.transAxes,
                                        fontsize=12, color="black",
                                        verticalalignment="top", horizontalalignment="right")
    else:
        vert_count_text.set_text(f"Vertical Intersections: {len(intersection_markers)}")
    fig.canvas.draw_idle()

def update_intersections_horizontal(y_val, fitted_data):
    """
    对于给定的水平虚线 y = y_val，计算与每条拟合线的无限线交点 x = (y_val - b) / m，
    并在有效时标记它。
    同时更新水平交点计数标签。
    """
    global horizontal_markers, image_ax, horiz_count_text
    for marker in horizontal_markers:
        try:
            marker.remove()
        except NotImplementedError:
            marker.set_visible(False)
    horizontal_markers.clear()
    xleft, xright = image_ax.get_xlim()
    for i, (slope, intercept) in enumerate(zip(fitted_data["slopes"], fitted_data["intercepts"])):
        if abs(slope) > 1e-6:
            x_val = (y_val - intercept) / slope
        else:
            x_val = (xleft + xright) / 2
        if xleft <= x_val <= xright:
            mkr, = image_ax.plot(x_val, y_val, 's',
                                 color=fitted_data["colors"][i],
                                 markersize=8, markeredgecolor='k')
            horizontal_markers.append(mkr)
    # 更新水平交点计数标签。
    if horiz_count_text is None:
        horiz_count_text = image_ax.text(0.98, 0.90,
                                         f"Horizontal Intersections: {len(horizontal_markers)}",
                                         transform=image_ax.transAxes,
                                         fontsize=12, color="black",
                                         verticalalignment="top", horizontalalignment="right")
    else:
        horiz_count_text.set_text(f"Horizontal Intersections: {len(horizontal_markers)}")
    fig.canvas.draw_idle()

# -------------------- 主更新函数 -------------------- #
def update_image(idx):
    """
    清除当前图像轴，精确显示图像（无额外空白），
    将拟合线延长到图像边界，更新虚线位置、
    交点标记，并重新创建滑块。

    重要：为确保交点计数标签在每张图像上更新，
    我们在清除轴后重置全局文本变量。
    """
    global image_ax, vline, hline, slider_dist, slider_time, current_fitted_data
    global vert_count_text, horiz_count_text

    m = files[idx]
    image_path = os.path.join(images_dir, m)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # 清除主图像轴。
    image_ax.cla()
    # 重置交点计数文本全局变量，以便创建新对象。
    vert_count_text = None
    horiz_count_text = None

    # 获取图像尺寸。
    height, width = image.shape[:2]
    image_ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    alpha=0.5,
                    extent=[0, width, 0, height],
                    origin='lower')
    image_ax.set_xlim(0, width)
    image_ax.set_ylim(0, height)

    # 处理图像以获得拟合线参数。
    fitted_data = process_image(image)
    current_fitted_data = fitted_data

    # 延长并叠加拟合线。
    for i in range(len(fitted_data["slopes"])):
        m_line = fitted_data["slopes"][i]
        b_line = fitted_data["intercepts"][i]
        endpoints = extend_line_to_bounds(m_line, b_line, width, height)
        if endpoints is not None:
            (x0, y0), (x1, y1) = endpoints
            # 计算速度 = 1000 / |斜率|（如果斜率 != 0）
            if abs(m_line) < 1e-6:
                speed_str = "Inf"
            else:
                speed = 100 / abs(m_line)
                speed_str = f"{speed:.2f}"
            image_ax.plot([x0, x1], [y0, y1], '-',
                          color=fitted_data["colors"][i],
                          linewidth=2,
                          label=f'Line {i} (Slope={m_line:.2f}, Speed={speed_str} m/s)')
    image_ax.set_xlabel("Distance (meters)")
    image_ax.set_ylabel("Time (milliseconds)")
    image_ax.set_title(f"Line Fitting & Intersection for {m}")
    image_ax.legend(loc="upper left")

    # ---- 定义虚线的初始位置 ----
    init_dist = 256
    init_time = height // 2

    # 绘制垂直虚线。
    ybottom, ytop = image_ax.get_ylim()
    vline_obj, = image_ax.plot([init_dist, init_dist], [ybottom, ytop],
                               'k--', linewidth=2,
                               label=f'Vertical line at x={init_dist}')
    vline = vline_obj

    # 绘制水平虚线。
    xleft, xright = image_ax.get_xlim()
    hline_obj, = image_ax.plot([xleft, xright], [init_time, init_time],
                               'c--', linewidth=2,
                               label=f'Horizontal line at y={init_time}')
    hline = hline_obj

    # 更新交点标记。
    update_intersections_vertical(init_dist, fitted_data)
    update_intersections_horizontal(init_time, fitted_data)

    # -------------------- 重新创建滑块 -------------------- #
    slider_dist_ax.cla()
    slider_dist = Slider(slider_dist_ax, 'Distance', 0, width,
                         valinit=init_dist, valfmt='%0.0f')

    slider_time_ax.cla()
    slider_time = Slider(slider_time_ax, 'Time', 0, height,
                         valinit=init_time, orientation='vertical', valfmt='%0.0f')

    def dist_slider_update(val):
        new_dist = slider_dist.val
        ybottom, ytop = image_ax.get_ylim()
        vline.set_data([new_dist, new_dist], [ybottom, ytop])
        update_intersections_vertical(new_dist, fitted_data)
        fig.canvas.draw_idle()

    slider_dist.on_changed(dist_slider_update)

    def time_slider_update(val):
        new_time = slider_time.val
        xleft, xright = image_ax.get_xlim()
        hline.set_data([xleft, xright], [new_time, new_time])
        update_intersections_horizontal(new_time, fitted_data)
        fig.canvas.draw_idle()

    slider_time.on_changed(time_slider_update)

    fig.canvas.draw_idle()

    output_path = os.path.join(output_dir, m)
    fig.savefig(output_path)

# -------------------- 按钮回调 -------------------- #
def next_image(event):
    global current_img_idx
    if current_img_idx < len(files) - 1:
        current_img_idx += 1
        update_image(current_img_idx)
    else:
        print("Already at the last image.")

def prev_image(event):
    global current_img_idx
    if current_img_idx > 0:
        current_img_idx -= 1
        update_image(current_img_idx)
    else:
        print("Already at the first image.")

btn_next.on_clicked(next_image)
btn_prev.on_clicked(prev_image)

# -------------------- 启动查看器 -------------------- #
update_image(current_img_idx)
plt.show()