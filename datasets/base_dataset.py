import  os
import cv2
import math
import random
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms

def get_gaussian_band_pass_filter(shape, cutoff_high = 2, cutoff_low = 30):
    """
        Gaussian band pass filter
    """
    d0 = cutoff_low
    rows, cols = shape[:2]
    mask = np.zeros((rows, cols))
    mid_row, mid_col = int(rows / 2), int(cols / 2)
    for i in range(rows):
        for j in range(cols):
            d = math.sqrt((i - mid_row) ** 2 + (j - mid_col) ** 2)
            if d < cutoff_high:
                mask[i, j] = 0
            else:
                mask[i, j] = np.exp(-(d * d) / (2 * d0 * d0))
    mask = mask.reshape((rows, cols, 1))
    return np.tile(mask, (1, 1, 3))

def transform_JPEGcompression(image, compress_range = (30, 100)):
    '''
        Perform random JPEG Compression
    '''
    assert compress_range[0] < compress_range[1], "Lower and higher value not accepted: {} vs {}".format(compress_range[0], compress_range[1])
    image = Image.fromarray(image)
    jpegcompress_value = random.randint(compress_range[0], compress_range[1])
    out = BytesIO()
    image.save(out, 'JPEG', quality=jpegcompress_value)
    out.seek(0)
    rgb_image = Image.open(out)
    return np.array(rgb_image, dtype = np.uint8)


def transform_blur(img):
    '''
        Perform random blur
    '''
    flag = random.random()
    kernel_size = random.choice([3, 5, 7, 9])
    if flag > 0.66:
        img = cv2.blur(img, (kernel_size, kernel_size))
    elif flag > 0.33:
        img = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
    else:
        img = cv2.medianBlur(img, kernel_size)
    return img


def transform_resize(image, resize_range = (32, 112), target_size = 112):
    '''
        Perform random resize
    '''
    assert resize_range[0] < resize_range[1], "Lower and higher value not accepted: {} vs {}".format(resize_range[0], resize_range[1])
    resize_value = random.randint(resize_range[0], resize_range[1])
    resize_image = cv2.resize(image, (resize_value, resize_value))
    return cv2.resize(resize_image, (target_size, target_size))

def transform_mixup(image1, label1, image2, label2):
    assert label1 != label2 # Mixup different image
    ratio = random.random()
    image_mix = image1.astype('float32')*ratio + (1-ratio)*image2.astype('float32')
    label_mix = label1*ratio + (1-ratio)*label2
    return image_mix.astype('uint8'), label_mix

def transform_band_pass_filter(img, mask):
    '''
        Perform band pass filtering
    '''
    # 1. FFT
    fft = np.fft.fft2(img, axes = (0, 1))

    # 2. Shift the fft to the center of the low frequencies
    shift_fft = np.fft.fftshift(fft, axes = (0, 1))

    # 3. Filter the image frequency based on the mask
    filtered_image = np.multiply(mask, shift_fft)

    # 4. Compute the inverest shift
    shift_ifft = np.fft.ifftshift(filtered_image, axes = (0, 1))

    # 5. Compute the inverse fourier transform
    ifft = np.fft.ifft2(shift_ifft, axes = (0, 1))
    ifft = np.abs(ifft)

    return ifft.astype('uint8')


class FASData(Dataset):
    def __init__(self,
                data_path,
                input_size = 112,
                is_train=True,
                aug_mixup_prob = 0.0, # Disable
                aug_compress_prob = 0.15,
                aug_downscale_prob = 0.15,
                aug_blur_prob = 0.15):
        super(FASData, self).__init__()
        # Gather all data
        self.data = open(data_path).read().strip('\n').split('\n')
        # Split spoof & bodafine
        if aug_mixup_prob > 0:
            self.data_spoof = []
            self.data_bonda = []
            for record in self.data:
                label = record.split(',')[-1]
                if label == '0':
                    self.data_spoof.append(record)
                elif label == '1':
                    self.data_bonda.append(record)
                else:
                    raise NotImplementedError("Invalid record: {}".format(record))
            print('  - Total spoof: ', len(self.data_spoof))
            print('  - Total bondafine: ', len(self.data_bonda))
        self.input_size = input_size
        self.is_train = is_train
        self.aug_mixup_prob = aug_mixup_prob
        self.aug_compress_prob = aug_compress_prob
        self.aug_downscale_prob = aug_downscale_prob
        self.aug_blur_prob = aug_blur_prob

        if self.is_train:
            self.transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        else:
            self.transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        self.kernel_bpf = get_gaussian_band_pass_filter((self.input_size, self.input_size))


    def __len__(self):
        return len(self.data)

    def aug_single_image(self, img):
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if random.random() < self.aug_blur_prob:
            img = transform_blur(img)
        if random.random() < self.aug_downscale_prob:
            img = transform_resize(img, resize_range = (64, 224), target_size = 224) 
        if random.random() < self.aug_compress_prob:
            img = transform_JPEGcompression(img, compress_range = (40, 100))
        return img 

    def __getitem__(self, idx):
        img, target = self.data[idx].split(',')
        target = int(target)
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        if self.is_train:
            img = self.aug_single_image(img)
            # Mixup
            if random.random() < self.aug_mixup_prob:
                if target == 0:
                    # Select bondafine
                    img_ref, target_ref = random.choice(self.data_bonda).split(',')
                elif target == 1:
                    # Select spoof
                    img_ref, target_ref = random.choice(self.data_spoof).split(',')
                else:
                    raise NotImplementedError("Invalid record: {}".format(self.data[idx]))
                target_ref = int(target_ref)
                img_ref = cv2.cvtColor(cv2.imread(img_ref), cv2.COLOR_BGR2RGB)
                img_ref = self.aug_single_image(img_ref)
                img, target = transform_mixup(img, float(target), img_ref, float(target_ref))

        assert target >= 0 and target <= 1
        img_filted = transform_band_pass_filter(img, self.kernel_bpf.copy())

        # Transform to Tensor
        target = torch.Tensor(np.array([float(target),]))
        img = self.transforms(img)
        img_filted = self.transforms(img_filted)
        return img, img_filted, target
