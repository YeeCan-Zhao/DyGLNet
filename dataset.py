import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        # 直接使用相同的文件名
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # 检查文件是否存在
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0  # 添加归一化

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask

class ISICDataset1(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 只加载jpg图像，并确保对应的png掩码存在
        self.images = [f for f in os.listdir(image_dir) 
                      if f.endswith('.jpg') and 
                      os.path.exists(os.path.join(mask_dir, f.replace('.jpg', '.png')))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        # 将.jpg替换为.png作为掩码文件
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '.png'))

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0  # 添加归一化

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]  # 只加载jpg图像

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        # 获取不带扩展名的文件名
        base_name = os.path.splitext(self.images[index])[0]
        # 使用.png作为掩码扩展名
        mask_path = os.path.join(self.mask_dir, base_name + '.png')

        # 检查文件是否存在
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}\n"
                                 f"Image file: {image_path}\n"
                                 f"Expected mask path: {mask_path}")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0  # 添加归一化

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask

class BUSIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]  # 只加载jpg图像

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        # 获取不带扩展名的文件名
        base_name = os.path.splitext(self.images[index])[0]
        # 使用.png作为掩码扩展名
        mask_path = os.path.join(self.mask_dir, base_name + '.png')

        # 检查文件是否存在
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}\n"
                                 f"Image file: {image_path}\n"
                                 f"Expected mask path: {mask_path}")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0  # 添加归一化

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask


class PH2Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.bmp')]  # 只加载bmp图像

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        # 使用相同的文件名
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # 检查文件是否存在
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}\n"
                                 f"Image file: {image_path}\n"
                                 f"Expected mask path: {mask_path}")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0  # 添加归一化

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask


class BrainMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]  # Only load .tif images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        # Use same filename for mask
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # Check if files exist
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}\n"
                                 f"Image file: {image_path}\n"
                                 f"Expected mask path: {mask_path}")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0  # Normalize to [0,1]

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask
    
class DriveDataset(Dataset):
    def __init__(self, image_dir, mask_dir, train=True, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.train = train

        # 训练集图片名是21_training.tif ~ 40_training.tif
        # 测试集图片名是01_test.tif ~ 20_test.tif
        if train:
            self.image_ids = [f"{i}_training" for i in range(21, 41)]
        else:
            self.image_ids = [f"{str(i).zfill(2)}_test" for i in range(1, 21)]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_path = os.path.join(self.image_dir, image_id + ".tif")

        if self.train:
            mask_id = image_id.replace("_training", "_manual1")
        else:
            mask_id = image_id.replace("_test", "_manual1")

        mask_path = os.path.join(self.mask_dir, mask_id + ".gif")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0  # 归一化为0~1

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


# 在dataset.py中添加BCCSDataset类
class BCCSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 只加载.tif图像（小写）
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith('.tif')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        
        # 处理文件名转换：去除"_ccd"并将扩展名改为大写
        base_name = self.images[index].replace('_ccd', '')
        mask_name = os.path.splitext(base_name)[0] + '.TIF'  # 确保扩展名大写
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 检查文件是否存在
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}\n"
                                  f"Image file: {image_path}\n"
                                  f"Expected mask path: {mask_path}")
        
        # 加载图像和掩码
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0  # 归一化
        
        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]
            
        return image, mask
    


class TNBCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        # 直接使用相同的文件名
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # 检查文件是否存在
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0  # 添加归一化

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask