"""Conservative plant augmentation; geometry is shared with indexed labels."""
import random
from PIL import Image, ImageEnhance, ImageOps

AUGMENTATION_CONFIG = {
    "version": 1,
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,
    "rotation_degrees": [0, 90, 180, 270],
    "scale_probability": 0.5,
    "scale_range": [0.8, 1.2],
    "photometric_probability": 0.5,
    "brightness_range": [0.9, 1.1],
    "contrast_range": [0.9, 1.1],
    "image_padding_rgb": [124, 116, 104],
    "mask_padding_class": 0,
}


def random_scale_crop(image, label, scale):
    """Isotropic resize, random crop/paste back to the original canvas size."""
    width, height = image.size
    if label.size != image.size:
        raise ValueError("Image and label sizes must match")
    nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
    image = image.resize((nw, nh), Image.Resampling.BILINEAR)
    label = label.resize((nw, nh), Image.Resampling.NEAREST)
    if nw >= width:
        left = random.randint(0, nw - width)
        x, crop_x = 0, left
    else:
        x, crop_x = random.randint(0, width - nw), 0
    if nh >= height:
        top = random.randint(0, nh - height)
        y, crop_y = 0, top
    else:
        y, crop_y = random.randint(0, height - nh), 0
    box = (crop_x, crop_y, crop_x + min(width, nw), crop_y + min(height, nh))
    image, label = image.crop(box), label.crop(box)
    canvas = Image.new("RGB", (width, height), tuple(AUGMENTATION_CONFIG["image_padding_rgb"]))
    target = Image.new("L", (width, height), 0)
    canvas.paste(image, (x, y))
    target.paste(label, (x, y))
    return canvas, target


def augment_train(image, label):
    if random.random() < 0.5:
        image, label = ImageOps.mirror(image), ImageOps.mirror(label)
    if random.random() < 0.5:
        image, label = ImageOps.flip(image), ImageOps.flip(label)
    turns = random.randrange(4)
    if turns:
        method = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
                  3: Image.Transpose.ROTATE_270}[turns]
        image, label = image.transpose(method), label.transpose(method)
    if random.random() < AUGMENTATION_CONFIG["scale_probability"]:
        image, label = random_scale_crop(image, label, random.uniform(*AUGMENTATION_CONFIG["scale_range"]))
    if random.random() < AUGMENTATION_CONFIG["photometric_probability"]:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(*AUGMENTATION_CONFIG["brightness_range"]))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(*AUGMENTATION_CONFIG["contrast_range"]))
    return image, label
