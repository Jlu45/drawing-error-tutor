import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

drawings_dir = 'data/drawings'
os.makedirs(drawings_dir, exist_ok=True)

def generate_example_drawing(filename, part_type):
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except:
        font = ImageFont.load_default()
    draw.text((400, 30), f'减速器{part_type}零件图', fill='black', font=font, anchor='mm')
    if part_type == '轴':
        draw.rectangle([200, 250, 600, 350], fill='white', outline='black', width=2)
        draw.rectangle([300, 270, 500, 330], fill='white', outline='black', width=1)
        draw.line([200, 200, 600, 200], fill='black', width=1)
        draw.line([200, 200, 200, 220], fill='black', width=1)
        draw.line([600, 200, 600, 220], fill='black', width=1)
        draw.text((400, 180), '400', fill='black', font=font, anchor='mm')
    elif part_type == '齿轮':
        draw.ellipse([200, 200, 600, 600], fill='white', outline='black', width=2)
        draw.ellipse([250, 250, 550, 550], fill='white', outline='black', width=1)
        draw.ellipse([375, 375, 425, 425], fill='white', outline='black', width=1)
    elif part_type == '箱体':
        draw.rectangle([150, 150, 650, 450], fill='white', outline='black', width=2)
        holes = [(200, 200), (600, 200), (200, 400), (600, 400)]
        for hole in holes:
            draw.ellipse([hole[0]-20, hole[1]-20, hole[0]+20, hole[1]+20], fill='white', outline='black', width=1)
    img.save(os.path.join(drawings_dir, filename))
    print(f'生成图纸: {filename}')

part_types = ['轴', '齿轮', '箱体']
for i in range(3):
    for part_type in part_types:
        filename = f'reducer_{part_type}_{i+1}.png'
        generate_example_drawing(filename, part_type)

print('图纸生成完成！')
print(f'共生成 {3 * len(part_types)} 张示例图纸')
print(f'图纸保存位置: {os.path.abspath(drawings_dir)}')
