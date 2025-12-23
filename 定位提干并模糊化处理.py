import cv2


def template_matching(main_image_path, template_image_path, threshold=0.03):
    # 读取主图像（有答案那张）和模板图像（只有题干那张，可手动裁剪出题干部分作为模板 ）
    main_image = cv2.imread(main_image_path, 0)
    template = cv2.imread(template_image_path, 0)
    if main_image is None or template is None:
        raise ValueError("无法加载图像，请检查文件路径")

    h, w = template.shape[:2]

    # 进行模板匹配，这里使用平方差匹配法（TM_SQDIFF_NORMED ），值越小匹配度越高
    result = cv2.matchTemplate(main_image, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 检查匹配程度是否满足阈值
    if min_val > threshold:
        raise ValueError(f"匹配失败，匹配值 {min_val} 大于阈值 {threshold}")

    # 输出匹配成功信息
    print(f"匹配成功，匹配值: {min_val}")
    print(f"匹配区域坐标: 左上角 {min_loc}, 右下角 {(min_loc[0] + w, min_loc[1] + h)}")

    # 确定匹配区域（题干区域）的坐标
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left, bottom_right

def blur_question_area(img, top_left, bottom_right, blur_kernel=(81, 81), alpha=0.01):
    xmin, ymin = top_left
    xmax, ymax = bottom_right
    # 对题干区域进行模糊处理
    blurred = cv2.GaussianBlur(img[ymin:ymax, xmin:xmax], blur_kernel, 0)
    # 调整模糊区域的透明度（颜色变淡）
    blended = cv2.addWeighted(img[ymin:ymax, xmin:xmax], alpha, blurred, 1 - alpha, 0)
    img[ymin:ymax, xmin:xmax] = blended
    return img


if __name__ == "__main__":
    # main_image_path = 'D:/AgentProjects/homeworkExtract/output_images_test/21779/1/21779.png'
    main_image_path = 'D:/AgentProjects/homeworkExtract/output_images_test/19442/1/19442.png'
    template_image_path = 'D:/AgentProjects/homeworkExtract/template/21780.png'

    # 模板匹配找到题干区域
    top_left, bottom_right = template_matching(main_image_path, template_image_path)
    main_image = cv2.imread(main_image_path)
    # 对题干区域进行模糊处理
    blurred_img = blur_question_area(main_image, top_left, bottom_right)

    # 显示结果
    cv2.imshow('Blurred Image', blurred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()