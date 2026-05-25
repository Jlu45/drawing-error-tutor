"""
render_atlas_pages.py
=====================
将《机械制图常见错误经典示例图册》PDF 渲染为高清页面图片，
并自动检测/切出正误图对（错误图 a / 正确图 b）。

用法:
    python render_atlas_pages.py                     # 全部页面
    python render_atlas_pages.py --pages 1 5 10      # 指定页面
    python render_atlas_pages.py --render-only        # 仅渲染，不切图
    python render_atlas_pages.py --zoom 3             # 3x 缩放渲染
"""

import argparse
import os
import re
import sys

import cv2
import fitz
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
PDF_PATH = os.path.join(PROJECT_DIR, "data", "atlas", "raw", "机械制图常见错误经典示例图册.pdf")
PAGES_DIR = os.path.join(PROJECT_DIR, "data", "atlas", "pages")
WRONG_DIR = os.path.join(PROJECT_DIR, "data", "atlas", "images", "wrong")
CORRECT_DIR = os.path.join(PROJECT_DIR, "data", "atlas", "images", "correct")

CATEGORY_MAP = {
    "图线": "LINE",
    "线型": "LINE",
    "画法": "LINE",
    "尺寸": "DIMENSION",
    "标注": "DIMENSION",
    "表面": "SURFACE",
    "粗糙度": "SURFACE",
    "公差": "TOLERANCE",
    "配合": "TOLERANCE",
    "几何": "GEOMETRY",
    "作图": "GEOMETRY",
    "螺纹": "THREAD",
    "齿轮": "GEAR",
    "装配": "ASSEMBLY",
    "零件": "PART",
    "图样": "DRAWING",
    "剖面": "SECTION",
    "剖视": "SECTION",
    "断面": "SECTION",
    "简化": "SIMPLIFY",
    "规定": "STANDARD",
    "中心": "CENTER",
    "圆弧": "ARC",
    "倒角": "CHAMFER",
    "键": "KEY",
    "销": "PIN",
    "弹簧": "SPRING",
    "轴承": "BEARING",
    "焊接": "WELD",
    "密封": "SEAL",
}

DEFAULT_ZOOM = 2


def imwrite_chinese(path, img, ext=".png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = cv2.imencode(ext, img)[1]
    buf.tofile(path)


def imread_chinese(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def render_pages(pdf_path, pages_dir, zoom=DEFAULT_ZOOM, page_nums=None):
    os.makedirs(pages_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    total = doc.page_count
    rendered = []

    for page_idx in range(total):
        if page_nums is not None and (page_idx + 1) not in page_nums:
            continue
        page = doc[page_idx]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        out_name = f"page_{page_idx + 1:03d}.png"
        out_path = os.path.join(pages_dir, out_name)
        imwrite_chinese(out_path, img)
        rendered.append((page_idx + 1, out_path, img))
        print(f"[渲染] 第 {page_idx + 1}/{total} 页 -> {out_name}  ({pix.width}x{pix.height})")

    doc.close()
    return rendered


def extract_category_from_text(pdf_path, page_idx):
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    text = page.get_text("text")
    doc.close()

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for cn_key, en_key in CATEGORY_MAP.items():
            if cn_key in line:
                return en_key
    return "MISC"


def find_vertical_split(img):
    h, w = img.shape[:2]
    mid_x = w // 2
    search_start = max(0, mid_x - w // 10)
    search_end = min(w, mid_x + w // 10)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    v_proj = np.sum(binary, axis=0)
    min_col = np.argmin(v_proj[search_start:search_end])
    return search_start + min_col


def detect_content_blocks(column_img, close_h=80, min_height=60):
    gray = cv2.cvtColor(column_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_h))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    h_proj = np.sum(closed, axis=1)
    h, w = closed.shape
    threshold = w * 0.005

    in_content = False
    start = 0
    blocks = []

    for y in range(h):
        if h_proj[y] > threshold and not in_content:
            in_content = True
            start = y
        elif h_proj[y] <= threshold and in_content:
            in_content = False
            if y - start > min_height:
                blocks.append((start, y))

    if in_content and h - start > min_height:
        blocks.append((start, h))

    if len(blocks) <= 1:
        return blocks

    merged = [blocks[0]]
    for i in range(1, len(blocks)):
        gap = blocks[i][0] - merged[-1][1]
        if gap < close_h // 2:
            merged[-1] = (merged[-1][0], blocks[i][1])
        else:
            merged.append(blocks[i])

    return merged


def match_pairs(left_blocks, right_blocks, tolerance_ratio=0.4):
    pairs = []
    used_right = set()

    for lb in left_blocks:
        l_center = (lb[0] + lb[1]) / 2
        l_height = lb[1] - lb[0]
        best_idx = None
        best_dist = float("inf")

        for ri, rb in enumerate(right_blocks):
            if ri in used_right:
                continue
            r_center = (rb[0] + rb[1]) / 2
            dist = abs(l_center - r_center)
            max_tol = max(l_height * tolerance_ratio, 100)
            if dist < max_tol and dist < best_dist:
                best_dist = dist
                best_idx = ri

        if best_idx is not None:
            used_right.add(best_idx)
            pairs.append((lb, right_blocks[best_idx]))

    return pairs


def auto_crop_images(rendered_pages, pdf_path, wrong_dir, correct_dir):
    os.makedirs(wrong_dir, exist_ok=True)
    os.makedirs(correct_dir, exist_ok=True)

    category_counters = {}
    total_pairs = 0

    for page_num, page_path, img in rendered_pages:
        category = extract_category_from_text(pdf_path, page_num - 1)
        h, w = img.shape[:2]

        split_x = find_vertical_split(img)
        margin = 10
        left_col = img[margin:h - margin, margin:split_x - margin]
        right_col = img[margin:h - margin, split_x + margin:w - margin]

        left_blocks = detect_content_blocks(left_col)
        right_blocks = detect_content_blocks(right_col)

        if not left_blocks or not right_blocks:
            print(f"  [跳过] 第 {page_num} 页: 未检测到图区 (左{len(left_blocks)}块 右{len(right_blocks)}块)")
            continue

        pairs = match_pairs(left_blocks, right_blocks)
        page_pairs = 0

        for lb, rb in pairs:
            left_img = left_col[lb[0]:lb[1], :]
            right_img = right_col[rb[0]:rb[1], :]

            if left_img.size == 0 or right_img.size == 0:
                continue

            key = category
            category_counters[key] = category_counters.get(key, 0) + 1
            idx = category_counters[key]

            wrong_name = f"ATLAS_{category}_{idx:03d}.png"
            correct_name = f"ATLAS_{category}_{idx:03d}.png"

            imwrite_chinese(os.path.join(wrong_dir, wrong_name), left_img)
            imwrite_chinese(os.path.join(correct_dir, correct_name), right_img)

            page_pairs += 1
            total_pairs += 1

        print(f"  [切图] 第 {page_num} 页: 左{len(left_blocks)}块 右{len(right_blocks)}块 匹配{len(pairs)}对, 切出 {page_pairs} 对 (类别={category})")

    print(f"\n[完成] 共切出 {total_pairs} 对正误图")
    for cat, cnt in sorted(category_counters.items()):
        print(f"  {cat}: {cnt} 对")


def main():
    parser = argparse.ArgumentParser(description="渲染图册PDF并自动切出正误图对")
    parser.add_argument("--pages", type=int, nargs="*", default=None, help="指定渲染页码(从1开始)")
    parser.add_argument("--render-only", action="store_true", help="仅渲染页面图片，不切图")
    parser.add_argument("--zoom", type=float, default=DEFAULT_ZOOM, help="渲染缩放倍数(默认2)")
    parser.add_argument("--pdf", type=str, default=PDF_PATH, help="PDF文件路径")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        print(f"[错误] PDF文件不存在: {args.pdf}")
        sys.exit(1)

    print(f"=== 渲染图册 (zoom={args.zoom}x) ===")
    rendered = render_pages(args.pdf, PAGES_DIR, zoom=args.zoom, page_nums=args.pages)

    if args.render_only:
        print(f"[完成] 仅渲染模式, 共渲染 {len(rendered)} 页")
        return

    print(f"\n=== 自动切图 ===")
    auto_crop_images(rendered, args.pdf, WRONG_DIR, CORRECT_DIR)


if __name__ == "__main__":
    main()
