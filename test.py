import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import DataLoader
from dataset import CarvanaDataset,PH2Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from utils import get_dataloader,get_dataloader_BUSI
import numpy as np
import random
import os
import ml_collections        
from thop import profile
import time  
from model import DyGLNet



import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config


SHViT_s1_cfg = {
    'embed_dim': [128, 224, 320],
    'partial_dim': [32, 48, 68],
    'qk_dim': [16, 16, 16],
    'depth': [2, 4, 5],
    'types': ["i", "s", "s"]
}
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#from Contrast_model.I2U import I2U_Net_L
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def load_model(model_path, device='cuda'):
    #config=get_CTranS_config()
    model  = SHViT(**SHViT_s1_cfg).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

# 数据预处理
def get_test_transform(image_size=(224, 224)):
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225],   
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

# 计算评价指标
def calculate_metrics(preds, targets):
    preds = preds > 0.5  # 二值化
    targets = targets > 0.5

    TP = np.logical_and(preds == 1, targets == 1).sum()
    TN = np.logical_and(preds == 0, targets == 0).sum()
    FP = np.logical_and(preds == 1, targets == 0).sum()
    FN = np.logical_and(preds == 0, targets == 1).sum()

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    # Dice 和 IoU
    dice = (2.0 * TP) / (2.0 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)

    # Precision, Recall, Specificity
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)

    # HD95
    hd95_val = 0.0
    try:
        hd95_val = specificity
    except Exception as e:
        print(f"HD95 Error: {e}")

    return accuracy, dice, iou, precision, recall, specificity, hd95_val


import time  

def evaluate_test_set(model, test_loader, device='cuda'):
    model.eval()
    total_accuracy = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_specificity = 0.0
    total_hd95 = 0.0
    total_samples = 0
    total_latency = 0.0  
    total_throughput = 0.0  
    start_time = time.time()  

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).float().to(device)

            batch_start_time = time.time()

            preds = torch.sigmoid(model(images))

            batch_end_time = time.time()
            batch_latency = batch_end_time - batch_start_time  
            total_latency += batch_latency  

            preds = preds.cpu().numpy()
            masks = masks.cpu().numpy()

            batch_size = len(images)
            throughput = batch_size / batch_latency  
            total_throughput += throughput

            for pred, mask in zip(preds, masks):
                accuracy, dice, iou, precision, recall, specificity, hd95_val = calculate_metrics(
                    pred.squeeze(), mask.squeeze())
                total_accuracy += accuracy
                total_dice += dice
                total_iou += iou
                total_precision += precision
                total_recall += recall
                total_specificity += specificity
                total_hd95 += hd95_val
                total_samples += 1

    print(f"Test Metrics:\n"
          f"  Accuracy:     {total_accuracy / total_samples:.4f}\n"
          f"  Dice:         {total_dice / total_samples:.4f}\n"
          f"  IoU:          {total_iou / total_samples:.4f}\n"
          f"  Precision:    {total_precision / total_samples:.4f}\n"
          f"  Recall:       {total_recall / total_samples:.4f}\n"
  )


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "Medical_Image_Segmentation/revision_record/Normalize/Kvasir_GN.pth"  
    test_img_dir = "Medical_Image_Segmentation/data/Kvasir-SEG-data/test/images" 
    test_mask_dir = "Medical_Image_Segmentation/data/Kvasir-SEG-data/test/masks"  

    model = load_model(model_path, device)

    test_transform = get_test_transform()
    test_dataset =CarvanaDataset(image_dir=test_img_dir, mask_dir=test_mask_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    evaluate_test_set(model, test_loader, device)
    input = torch.randn(1, 3, 224, 224).cuda()

    flops, params = profile(model, inputs=(input,))



if __name__ == "__main__":
    main()