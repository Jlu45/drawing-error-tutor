"""
crop_atlas_cases.py
===================
交互式 / 批量从页面图片中切出指定区域的正误图对。

支持两种模式:
  1. 配置文件批量切图 (--config)
  2. 交互式鼠标选区切图 (--interactive)

配置文件格式 (JSON):
{
    "pages": [
        {
            "page": 1,
            "category": "LINE",
            "pairs": [
                {
                    "wrong":  [x1, y1, x2, y2],
                    "correct": [x1, y1, x2, y2]
                }
            ]
        }
    ]
}

用法:
    python crop_atlas_cases.py --config crop_config.json
    python crop_atlas_cases.py --interactive --page 5
    python crop_atlas_cases.py --interactive --page 5 --category DIMENSION
    python crop_atlas_cases.py --generate-template            # 生成空白配置模板
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
PAGES_DIR = os.path.join(PROJECT_DIR, "data", "atlas", "pages")
WRONG_DIR = os.path.join(PROJECT_DIR, "data", "atlas", "images", "wrong")
CORRECT_DIR = os.path.join(PROJECT_DIR, "data", "atlas", "images", "correct")
DEFAULT_CONFIG = os.path.join(BASE_DIR, "crop_config.json")

WINDOW_NAME = "CropAtlas - Select Region"


def imwrite_chinese(path, img, ext=".png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = cv2.imencode(ext, img)[1]
    buf.tofile(path)


def imread_chinese(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def get_page_path(page_num):
    return os.path.join(PAGES_DIR, f"page_{page_num:03d}.png")


def crop_region(img, rect):
    x1, y1, x2, y2 = rect
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)
    return img[y1:y2, x1:x2]


def batch_crop(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    os.makedirs(WRONG_DIR, exist_ok=True)
    os.makedirs(CORRECT_DIR, exist_ok=True)

    category_counters = {}
    total = 0

    for page_cfg in config.get("pages", []):
        page_num = page_cfg["page"]
        category = page_cfg.get("category", "MISC")
        page_path = get_page_path(page_num)

        if not os.path.isfile(page_path):
            print(f"[跳过] 页面图片不存在: {page_path}")
            continue

        img = imread_chinese(page_path)
        if img is None:
            print(f"[跳过] 无法读取: {page_path}")
            continue

        for pair in page_cfg.get("pairs", []):
            wrong_rect = pair.get("wrong")
            correct_rect = pair.get("correct")

            if wrong_rect is None or correct_rect is None:
                print(f"  [跳过] 第 {page_num} 页某对缺少坐标")
                continue

            wrong_img = crop_region(img, wrong_rect)
            correct_img = crop_region(img, correct_rect)

            if wrong_img.size == 0 or correct_img.size == 0:
                print(f"  [跳过] 第 {page_num} 页某对切图区域为空")
                continue

            key = category
            category_counters[key] = category_counters.get(key, 0) + 1
            idx = category_counters[key]

            wrong_name = f"ATLAS_{category}_{idx:03d}.png"
            correct_name = f"ATLAS_{category}_{idx:03d}.png"

            imwrite_chinese(os.path.join(WRONG_DIR, wrong_name), wrong_img)
            imwrite_chinese(os.path.join(CORRECT_DIR, correct_name), correct_img)

            total += 1
            print(f"  [切图] 第 {page_num} 页 -> {wrong_name} (wrong) / {correct_name} (correct)")

    print(f"\n[完成] 批量切图共 {total} 对")


class InteractiveCropper:
    def __init__(self, img, page_num, category):
        self.img = img.copy()
        self.display = img.copy()
        self.page_num = page_num
        self.category = category
        self.pair_count = 0
        self.drawing = False
        self.rect_start = None
        self.rect_end = None
        self.current_label = "wrong"
        self.wrong_rect = None
        self.results = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.rect_start = (x, y)
            self.rect_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.rect_end = (x, y)
            self.display = self.img.copy()
            color = (0, 0, 255) if self.current_label == "wrong" else (0, 255, 0)
            cv2.rectangle(self.display, self.rect_start, self.rect_end, color, 2)
            label_text = f"{'[错误 a]' if self.current_label == 'wrong' else '[正确 b]'}"
            cv2.putText(self.display, label_text,
                        (self.rect_start[0], self.rect_start[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rect_end = (x, y)
            rect = [self.rect_start[0], self.rect_start[1],
                    self.rect_end[0], self.rect_end[1]]

            if self.current_label == "wrong":
                self.wrong_rect = rect
                self.current_label = "correct"
                self.display = self.img.copy()
                color = (0, 0, 255)
                cv2.rectangle(self.display,
                              (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
                cv2.putText(self.display, "[错误 a]",
                            (rect[0], rect[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(self.display, "Now select [正确 b] region",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                correct_rect = rect
                self.pair_count += 1
                self.results.append({
                    "wrong": self.wrong_rect,
                    "correct": correct_rect
                })
                self._save_pair(self.wrong_rect, correct_rect, self.pair_count)
                self.wrong_rect = None
                self.current_label = "wrong"
                self.display = self.img.copy()
                color_w = (0, 0, 255)
                color_c = (0, 255, 0)
                cv2.rectangle(self.display,
                              (self.results[-1]["wrong"][0], self.results[-1]["wrong"][1]),
                              (self.results[-1]["wrong"][2], self.results[-1]["wrong"][3]),
                              color_w, 2)
                cv2.rectangle(self.display,
                              (correct_rect[0], correct_rect[1]),
                              (correct_rect[2], correct_rect[3]),
                              color_c, 2)
                cv2.putText(self.display, f"Pair #{self.pair_count} saved! Select next [错误 a]",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def _save_pair(self, wrong_rect, correct_rect, idx):
        os.makedirs(WRONG_DIR, exist_ok=True)
        os.makedirs(CORRECT_DIR, exist_ok=True)

        wrong_img = crop_region(self.img, wrong_rect)
        correct_img = crop_region(self.img, correct_rect)

        name = f"ATLAS_{self.category}_{idx:03d}.png"
        imwrite_chinese(os.path.join(WRONG_DIR, name), wrong_img)
        imwrite_chinese(os.path.join(CORRECT_DIR, name), correct_img)
        print(f"  [保存] {name} -> wrong/ + correct/")

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        h, w = self.img.shape[:2]
        scale = min(1600 / w, 900 / h, 1.0)
        if scale < 1.0:
            cv2.resizeWindow(WINDOW_NAME, int(w * scale), int(h * scale))

        print(f"\n=== 交互式切图 - 第 {self.page_num} 页 (类别={self.category}) ===")
        print("操作说明:")
        print("  鼠标左键拖拽: 选择区域")
        print("  先选 [错误 a] 区域, 再选 [正确 b] 区域")
        print("  u: 撤销上一对")
        print("  q/ESC: 退出并保存配置")
        print()

        while True:
            cv2.imshow(WINDOW_NAME, self.display)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                break
            elif key == ord("u"):
                if self.results:
                    removed = self.results.pop()
                    self.pair_count -= 1
                    self.current_label = "wrong"
                    self.wrong_rect = None
                    self.display = self.img.copy()
                    for i, r in enumerate(self.results, 1):
                        cv2.rectangle(self.display,
                                      (r["wrong"][0], r["wrong"][1]),
                                      (r["wrong"][2], r["wrong"][3]), (0, 0, 255), 2)
                        cv2.rectangle(self.display,
                                      (r["correct"][0], r["correct"][1]),
                                      (r["correct"][2], r["correct"][3]), (0, 255, 0), 2)
                    print(f"  [撤销] 移除第 {self.pair_count + 1} 对")

        cv2.destroyAllWindows()
        return self.results


def interactive_crop(page_num, category):
    page_path = get_page_path(page_num)
    if not os.path.isfile(page_path):
        print(f"[错误] 页面图片不存在: {page_path}")
        return

    img = imread_chinese(page_path)
    if img is None:
        print(f"[错误] 无法读取: {page_path}")
        return

    cropper = InteractiveCropper(img, page_num, category)
    results = cropper.run()

    if results:
        config_entry = {
            "page": page_num,
            "category": category,
            "pairs": results
        }
        config_path = os.path.join(BASE_DIR, f"crop_page_{page_num:03d}.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"pages": [config_entry]}, f, ensure_ascii=False, indent=2)
        print(f"\n[保存] 配置已写入: {config_path}")
    else:
        print("\n[完成] 未切出任何图对")


def generate_template(output_path):
    template = {
        "pages": [
            {
                "page": 1,
                "category": "LINE",
                "pairs": [
                    {
                        "wrong": [50, 100, 500, 400],
                        "correct": [550, 100, 1000, 400]
                    }
                ]
            }
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"[生成] 配置模板已写入: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="交互式/批量切出正误图对")
    parser.add_argument("--config", type=str, default=None, help="批量切图配置文件路径")
    parser.add_argument("--interactive", action="store_true", help="交互式选区切图")
    parser.add_argument("--page", type=int, default=None, help="交互模式指定页码")
    parser.add_argument("--category", type=str, default="MISC", help="类别标签(默认MISC)")
    parser.add_argument("--generate-template", action="store_true", help="生成空白配置模板")
    parser.add_argument("--output-config", type=str, default=DEFAULT_CONFIG, help="模板输出路径")
    args = parser.parse_args()

    if args.generate_template:
        generate_template(args.output_config)
        return

    if args.config:
        batch_crop(args.config)
        return

    if args.interactive:
        if args.page is None:
            print("[错误] 交互模式需要指定 --page 参数")
            sys.exit(1)
        interactive_crop(args.page, args.category)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
