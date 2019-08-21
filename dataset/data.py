import tarfile
from os import remove
from os.path import exists, join, basename
import random

from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomVerticalFlip, RandomHorizontalFlip, RandomChoice
from torchvision.transforms.functional import rotate

from .dataset import DatasetFromFolder


def download_bsd300(dest="./dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def random_rotation(img):
    throw = random.random()
    if throw > 0.5:
        return img
    else:
        return rotate(img, 90)




def augment_transform():
    return Compose([
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        random_rotation
    ])


def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size),
                             augment_transform=augment_transform())


def get_valid_set(upscale_factor):
    root_dir = download_bsd300()
    valid_dir = join(root_dir, "valid")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(valid_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    root_dir = "./dataset/urban100/images"
    valid_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(valid_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
