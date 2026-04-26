import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def inject_errors(image_path, output_dir='data/error_drawings', label_dir='data/error_labels'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取: {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    errors_injected = []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    error_img = img.copy()

    if len(contours) > 3:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        cv2.rectangle(error_img, (x, y), (x + cw, y + ch), (255, 255, 255), -1)
        errors_injected.append({
            'type': '尺寸标注',
            'description': f'删除了主要轮廓区域 ({x},{y})-({x+cw},{y+ch})',
            'severity': '高'
        })

    for i in range(min(3, len(contours))):
        cnt = contours[i]
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        text_region = error_img[max(0,cy-5):min(h,cy+ch+5), max(0,cx-5):min(w,cx+cw+5)]
        if text_region.size > 0:
            noise = np.random.randint(0, 50, text_region.shape, dtype=np.uint8)
            error_img[max(0,cy-5):min(h,cy+ch+5), max(0,cx-5):min(w,cx+cw+5)] = text_region
            errors_injected.append({
                'type': '文字模糊',
                'description': f'文字区域模糊化 ({cx},{cy})',
                'severity': '中'
            })

    output_path = os.path.join(output_dir, f'error_{base_name}.png')
    cv2.imwrite(output_path, error_img)

    label_path = os.path.join(label_dir, f'error_{base_name}.txt')
    with open(label_path, 'w', encoding='utf-8') as f:
        for err in errors_injected:
            f.write(f"{err['type']}|{err['description']}|{err['severity']}\n")

    print(f"注入错误: {output_path} ({len(errors_injected)}个错误)")
    return output_path, errors_injected

if __name__ == '__main__':
    drawings_dir = 'data/drawings'
    if os.path.exists(drawings_dir):
        for filename in os.listdir(drawings_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(drawings_dir, filename)
                inject_errors(image_path)
    else:
        print(f"目录不存在: {drawings_dir}")
