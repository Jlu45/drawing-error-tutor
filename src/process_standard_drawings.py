import os
import cv2
import json
from rag_knowledge_base import DualKnowledgeBase

def split_drawing_into_parts(image_path):
    parts = []
    try:
        image = cv2.imread(image_path)
        if image is None:
            return parts
        height, width = image.shape[:2]
        grid_size = 2
        part_height = height // grid_size
        part_width = width // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                y1 = i * part_height
                y2 = (i + 1) * part_height
                x1 = j * part_width
                x2 = (j + 1) * part_width
                part = image[y1:y2, x1:x2]
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                part_filename = f"{base_name}_part_{i}_{j}.png"
                part_path = os.path.join('data', 'standard_drawings', part_filename)
                cv2.imwrite(part_path, part)
                annotation = {
                    'description': f"{base_name} 的局部区域 ({i}, {j})",
                    'position': f"({x1}, {y1}) - ({x2}, {y2})",
                    'source': base_name
                }
                parts.append((part_path, annotation))
        return parts
    except Exception as e:
        print(f"拆分图纸失败: {e}")
        return parts

def process_standard_drawings():
    kb = DualKnowledgeBase()
    standard_drawings_dir = 'data/standard_drawings'
    for filename in os.listdir(standard_drawings_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')) and 'part' not in filename and 'error_' not in filename:
            image_path = os.path.join(standard_drawings_dir, filename)
            print(f"处理文件: {filename}")
            annotation = {
                'type': 'standard_drawing',
                'name': filename,
                'description': f'标准减速器装配图 - {filename}'
            }
            try:
                kb.add_image_knowledge(image_path, annotation)
                print(f"  -> 成功添加到图像知识库")
            except Exception as e:
                print(f"  -> 添加失败: {e}")
    print("\n标准图纸处理完成!")

if __name__ == '__main__':
    process_standard_drawings()
