import cv2
import numpy as np


def segment_columns(image_path, window_width=1, threshold=0.99):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化处理
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    height, width = binary.shape

    # 存储分割点
    split_points = []
    consecutive_count = 0  # 连续满足阈值的计数器
    for x in range(0, width - window_width):
        window = binary[:, x:x + window_width]
        white_pixels = np.sum(window == 255)
        total_pixels = window_width * height
        if white_pixels / total_pixels >= threshold:
            consecutive_count += 1
            if consecutive_count == 10:  # 连续5次满足阈值
                split_point = x + window_width // 2
                split_points.append(split_point)
                print(f"找到分割点: {split_point}")  # 输出分割点位置
                consecutive_count = 0  # 重置计数器
        else:
            consecutive_count = 0  # 不满足阈值时重置计数器

    # 去除相邻过近的分割点
    filtered_split_points = []
    if split_points:
        filtered_split_points.append(split_points[0])
        for i in range(1, len(split_points)):
            if split_points[i] - split_points[i - 1] > window_width:
                filtered_split_points.append(split_points[i])

    # 根据分割点分割图像
    column_images = []
    start_x = 0
    for split_point in filtered_split_points:
        gray_column = binary[:, start_x:split_point]
        white_pixels = np.sum(gray_column == 255)
        sub_total_pixels = gray_column.size
        if white_pixels / sub_total_pixels < 0.95:  # 白色像素比例小于95%才保存
            column = image[:, start_x:split_point]
            column_images.append(column)
        start_x = split_point
    # 添加最后一列
    column = image[:, start_x:]
    column_images.append(column)

    return column_images


if __name__ == "__main__":
    image_path = "D:/AgentProjects/homeworkExtract/template/four_col.jpg"  # 替换为实际的图像路径
    columns = segment_columns(image_path)
    for i, column in enumerate(columns):
        cv2.imwrite(f"slice_col/column_{i + 1}.jpg", column)

