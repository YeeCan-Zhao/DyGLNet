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
from model import DyGLNet

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


LEARNING_RATE = 0.001 
BATCH_SIZE = 16 
NUM_EPOCHS = 130 
WARMUP_EPOCHS = 10  
WEIGHT_DECAY = 3e-5 
PIN_MEMORY = True
import os
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_IMG_PATH = "Medical_Image_Segmentation/data/Kvasir-SEG-data/train/images"
TRAIN_MASK_PATH = "Medical_Image_Segmentation/data/Kvasir-SEG-data/train/masks"
VAL_IMG_PATH = "Medical_Image_Segmentation/data/Kvasir-SEG-data/val/images"
VAL_MASK_PATH = "Medical_Image_Segmentation/data/Kvasir-SEG-data/val/masks"

IMAGE_HEIGHT = 256
IMG_WIDTH = 256
NUM_WORKERS = 2

train_losses = []
val_acc = []
val_dice = []

seed = 42  
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, target):
       
        preds = torch.sigmoid(preds)  
        intersection = (preds * target).sum()
        dice = (2.0 * intersection + self.smooth) / (preds.sum() + target.sum() + self.smooth)
        return 1 - dice

# Dice + Cross-Entropy Loss
class DiceCELoss(nn.Module):
    def __init__(self, lambda_ce=0.5, smooth=1e-8):
        super(DiceCELoss, self).__init__()
        self.lambda_ce = lambda_ce
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, target):
        dice_loss = self.dice_loss(preds, target)
        ce_loss = self.ce_loss(preds, target)
        return self.lambda_ce * ce_loss + (1 - self.lambda_ce) * dice_loss

def train_fn(loader, model, loss_fn, optimizer, scaler, epoch):
    loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=True)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for index, (data, target) in enumerate(loop):
        data = data.to(DEVICE).float()
        target = target.unsqueeze(1).float().to(DEVICE)

        with torch.amp.autocast(device_type='cuda', enabled=True):
            predict = model(data)
            loss = loss_fn(predict, target)

            preds_sigmoid = torch.sigmoid(predict)
            preds_binary = (preds_sigmoid > 0.5).float()

            # 逐batch计算Dice和IoU
            intersection = (preds_binary * target).sum()
            union = preds_binary.sum() + target.sum() - intersection
            dice = (2.0 * intersection) / (preds_binary.sum() + target.sum() + 1e-8)
            iou = intersection / (union + 1e-8)

            total_dice += dice.item()
            total_iou += iou.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item(), lr=optimizer.param_groups[0]['lr'])


    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)
    return avg_loss, avg_dice

def check_accuracy(loader, model, loss_fn, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_hd95 = 0.0
    count_hd = 0

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device).float()
            y = y.unsqueeze(1).float().to(device)

            with torch.amp.autocast(device_type='cuda', enabled=True):
                preds = torch.sigmoid(model(x))
                preds_binary = (preds > 0.5).float()
                y_binary = (y > 0.5).float()

                # 逐batch计算Dice和IoU
                intersection = (preds_binary * y_binary).sum()
                union = preds_binary.sum() + y_binary.sum() - intersection
                dice = (2.0 * intersection) / (preds_binary.sum() + y_binary.sum() + 1e-8)
                iou = intersection / (union + 1e-8)

                total_dice += dice.item()
                total_iou += iou.item()

                

                loss = loss_fn(model(x), y)
                total_loss += loss.item()

   
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)


    print(f"Validation Metrics: Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | Loss: {avg_loss:.4f}")
    model.train()
    return avg_dice, avg_iou, avg_loss

SHViT_s1_cfg = {
    'embed_dim': [128, 224, 320],
    'partial_dim': [32, 48, 68],
    'qk_dim': [16, 16, 16],
    'depth': [2, 4, 5],
    'types': ["i", "s", "s"]
}

def main():
    # 数据增强
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(IMAGE_HEIGHT, IMG_WIDTH), scale=(0.5, 1.0)), 
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2), 
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225],   
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMG_WIDTH),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225],   
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


    train_loader, val_loader = get_dataloader(
        TRAIN_IMG_PATH, TRAIN_MASK_PATH,
        VAL_IMG_PATH, VAL_MASK_PATH,
        train_transform, val_transform,
        BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,worker_init_fn=seed_worker,
    generator=g
    )

    model =  SHViT(**SHViT_s1_cfg).to(device=DEVICE)

    loss_fn = DiceCELoss(lambda_ce=0.5)  
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler()

    def poly_lr_scheduler(epoch, warmup_epochs=WARMUP_EPOCHS, power=0.9):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  
        else:
            return (1 - (epoch - warmup_epochs) / (NUM_EPOCHS - warmup_epochs)) ** power  

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_scheduler)

    best_dice = 0.0
    best_epoch = 0
    early_stop_counter = 0
    patience = 100  

    log_file = open("Best.txt", "w")
    log_file.write("Epoch\tTrain Loss\tTrain Dice\tVal Accuracy\tVal Dice\tVal IoU\tVal HD95\tVal Loss\tLearning Rate\n")

    for epoch in range(NUM_EPOCHS):
        print(f"Current Epoch: {epoch + 1}")
        train_loss, train_dice = train_fn(train_loader, model, loss_fn, optimizer, scaler, epoch)
        train_losses.append(train_loss)

        dice, iou, val_loss = check_accuracy(val_loader, model, loss_fn, device=DEVICE)
        val_acc.append(dice)
        val_dice.append(dice)

        log_file.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_dice:.4f}\t{dice:.4f}\t{dice:.4f}\t{iou:.4f}\t{val_loss:.4f}\t{optimizer.param_groups[0]['lr']:.6f}\n")
        log_file.flush() 

        if dice > best_dice:
            best_dice = dice
            best_epoch = epoch + 1
            early_stop_counter = 0
            torch.save(model.state_dict(), "Best.pth") 
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break

        scheduler.step()

    log_file.write(f"\nBest Epoch: {best_epoch}\n")
    log_file.write(f"Best Dice: {best_dice:.4f}\n")
    log_file.close()

    print(f"Best Epoch: {best_epoch}")
    print(f"Best Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()