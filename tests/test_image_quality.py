import os
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.utils.image_quality import filter_informative_images, is_informative_image


def _save(path, image):
    image.save(path)
    return path


def test_solid_and_low_foreground_images_are_filtered():
    with tempfile.TemporaryDirectory() as tmp:
        blank = _save(os.path.join(tmp, "blank.png"), Image.new("RGB", (160, 160), "white"))

        sparse = Image.new("RGB", (160, 160), "white")
        for x in range(76, 84):
            for y in range(76, 84):
                sparse.putpixel((x, y), (0, 0, 0))
        sparse_path = _save(os.path.join(tmp, "sparse.png"), sparse)

        assert is_informative_image(blank)[0] is False
        assert is_informative_image(sparse_path)[0] is False


def test_filter_keeps_informative_image_and_aligned_description():
    with tempfile.TemporaryDirectory() as tmp:
        image = Image.new("RGB", (160, 160), "white")
        for x in range(30, 130):
            for y in range(30, 130):
                image.putpixel((x, y), (34, 139, 34))
        good_path = _save(os.path.join(tmp, "bamboo_like.png"), image)

        paths, descriptions = filter_informative_images(
            [good_path],
            ["绿色竹叶图片"],
            require_description=True,
        )

        assert paths == [good_path]
        assert descriptions == ["绿色竹叶图片"]


if __name__ == "__main__":
    test_solid_and_low_foreground_images_are_filtered()
    test_filter_keeps_informative_image_and_aligned_description()
    print("test_image_quality passed")
