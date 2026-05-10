import os
from typing import List, Optional, Tuple

from PIL import Image


def _dominant_color_ratio(img: Image.Image) -> float:
    small = img.resize((64, 64))
    pixel_counts = {}
    for pixel in small.getdata():
        pixel_counts[pixel] = pixel_counts.get(pixel, 0) + 1
    return max(pixel_counts.values()) / (64 * 64)


def _foreground_ratio_against_background(img: Image.Image) -> float:
    small = img.resize((96, 96)).convert("RGB")
    pixels = list(small.getdata())
    pixel_counts = {}
    for pixel in pixels:
        pixel_counts[pixel] = pixel_counts.get(pixel, 0) + 1
    background = max(pixel_counts, key=pixel_counts.get)

    foreground = 0
    for pixel in pixels:
        delta = (
            abs(pixel[0] - background[0])
            + abs(pixel[1] - background[1])
            + abs(pixel[2] - background[2])
        )
        if delta > 45:
            foreground += 1
    return foreground / len(pixels)


def is_informative_image(
    image_path: str,
    min_side: int = 50,
    min_area: int = 4000,
    dominant_threshold: float = 0.97,
    min_foreground_ratio: float = 0.025,
) -> Tuple[bool, str]:
    """Return whether a crop is visually useful enough to show as evidence."""
    if not image_path or not os.path.exists(image_path):
        return False, "图片不存在"

    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
    except Exception as e:
        return False, f"读取失败 ({e})"

    if width < min_side or height < min_side:
        return False, f"尺寸过小 ({width}x{height})"
    if width * height < min_area:
        return False, f"面积过小 ({width * height} px)"

    aspect_ratio = width / height
    if aspect_ratio > 8 or aspect_ratio < 0.125:
        return False, f"比例极端 ({aspect_ratio:.2f})"

    dominant_ratio = _dominant_color_ratio(img)
    if dominant_ratio >= dominant_threshold:
        return False, f"近似纯色背景 ({dominant_ratio:.2%})"

    foreground_ratio = _foreground_ratio_against_background(img)
    if foreground_ratio < min_foreground_ratio:
        return False, f"有效前景过少 ({foreground_ratio:.2%})"

    return True, "有效图片"


def filter_informative_images(
    image_paths: List[str],
    descriptions: Optional[List[str]] = None,
    require_description: bool = False,
) -> Tuple[List[str], List[str]]:
    """Filter image paths and keep descriptions aligned by index."""
    descriptions = descriptions or []
    kept_paths = []
    kept_descriptions = []

    for idx, path in enumerate(image_paths or []):
        desc = descriptions[idx] if idx < len(descriptions) else ""
        if require_description and not str(desc).strip():
            continue

        is_valid, _ = is_informative_image(path)
        if not is_valid:
            continue

        kept_paths.append(path)
        kept_descriptions.append(desc)

    return kept_paths, kept_descriptions
