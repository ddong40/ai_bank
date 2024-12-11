import glob
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

test_img_path = "/workspace/dataset/"

for path in glob.glob(f'{test_img_path}/**/*.png', recursive=True):
    with Image.open(path) as image:
        image_np = np.array(image, dtype=np.uint8)

    height, width, _ = image_np.shape
    print(height, width)
    break