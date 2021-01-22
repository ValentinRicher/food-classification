from io import BytesIO

import numpy as np
from PIL import Image
from pathlib import Path


def decode_and_resize_image(raw_bytes, size):
    """Read, decode and resize raw image bytes (e.g. raw content of a jpeg file).

    Parameters
    ----------
    raw_bytes: bytes
        Image bits, e.g. jpeg image.
    size: tuple
        Tuple of two int.

    Returns
    -------
        Multidimensional numpy array representing the resized image.
    """
    return np.asarray(Image.open(BytesIO(raw_bytes)).resize(size), dtype=np.float32)


def read_image(x):
    """Reads an image from its path.

    Parameters
    ----------
    x : str
        Path to the image.

    Returns
    -------
    The file from the image.
    """
    with open(x, "rb") as f:
        return f.read()


def resize_images(dir, width, height):
    """Resizes images from a directory to the size (width, height).

    Parameters
    ----------
    dir : str
        Dataset directory.
    width : int
        Width of the destination image.
    height : int
        Height of the destination image.
    """
    print("Resizing images...")

    for path in Path(dir).rglob("*.jpg"):
        main_dir = "/".join(str(path).split("/")[:-4])
        dataset_dir = str(path).split("/")[-4]
        new_dataset_dir = dataset_dir + "_" + str(width) + "_" + str(height)
        split_directory = str(path).split("/")[-3]
        food_class = str(path).split("/")[-2]
        img_name = str(path).split("/")[-1]

        new_path = (
            Path()
            / main_dir
            / new_dataset_dir
            / split_directory
            / food_class
            / img_name
        )

        im = Image.open(path)
        resized_im = im.resize((width, height), resample=Image.BILINEAR)
        if not (new_path.parent.exists()):
            Path.mkdir(new_path.parent, parents=True, exist_ok=True)
        resized_im.save(new_path)

def read_images(dir):
    for path in Path(dir).rglob("*.jpg"):
        print("Reading {}".format(path))
        im = Image.open(path)

def convert_from_jpg_to_jpeg(old_dir, new_dir):

    for root, dirs, files in os.walk(old_dir):
        for file in files:
            if file.endswith('.jpg'):

                img = Image.open(os.path.join(root, file)).convert('RGB')
                new_file = str(file).split('.jpg')[0] + '.jpeg'
                new_root = os.path.join(new_dir, '/'.join(str(root).split('/')[-2:]))
                new_path = os.path.join(new_root, new_file)
                # print(new_path)
                if not os.path.exists(new_root):
                    os.makedirs(new_root)
                img.save(new_path, 'jpeg')
                print('New image created: {}'.format(new_path))

# resize_images("/home/vricher/example-image/data/food-101", 160, 160)
# read_images("/home/vricher/example-image/data/food-101_160_160")

