import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from Medical_Image_Segmentation.model.DyGLNet import shvit_s1_modified 


def load_model(model_path, device='cuda'):
    model = shvit_s1_modified(pretrained=True).to(device) 
    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval()
    return model


def preprocess_image(image_path, image_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  
    return image

def predict_mask(model, image_tensor, device='cuda'):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        pred_mask = model(image_tensor)
        pred_mask = torch.sigmoid(pred_mask) 
        pred_mask = (pred_mask > 0.5).float() 
    return pred_mask.squeeze().cpu().numpy()  

def display_image_and_mask(image_path, mask):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.show()
    mask=Image.fromarray(mask.astype('uint8'),'L')
    mask.save("predict.jpg")

# 主函数
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "Medical_Image_Segmentation/record/Kvasir-SEG-4090/Mymodel2_1/Kvasir-SEG-My_4.pth"  # 训练好的模型权重路径
    image_path = "Medical_Image_Segmentation/data/Kvasir-SEG-data/test/images/cju0s690hkp960855tjuaqvv0.jpg"  # 输入息肉图像的路径

    model = load_model(model_path, device)

    image_tensor = preprocess_image(image_path)

    pred_mask = predict_mask(model, image_tensor, device)

    display_image_and_mask(image_path, pred_mask)

if __name__ == "__main__":
    main()