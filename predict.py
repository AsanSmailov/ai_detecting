import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class NestedUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(NestedUNet, self).__init__()
        nb_filter = [32, 64, 128, 256, 512]
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.conv0_0 = ConvBlock(in_ch, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        # Decoder with nested connections
        self.conv0_1 = ConvBlock(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self._up_and_pad(x1_0, x0_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up_and_pad(x2_0, x1_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up_and_pad(x3_0, x2_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up_and_pad(x4_0, x3_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up_and_pad(x1_1, x0_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up_and_pad(x2_1, x1_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up_and_pad(x3_1, x2_0)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up_and_pad(x1_2, x0_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up_and_pad(x2_2, x1_0)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up_and_pad(x1_3, x0_0)], 1))

        output = self.final(x0_4)
        return torch.sigmoid(output)

    def _up_and_pad(self, x, ref):
        x = self.up(x)
        diffY = ref.size()[2] - x.size()[2]
        diffX = ref.size()[3] - x.size()[3]
        x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2))
        return x

def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = NestedUNet(in_ch=1, out_ch=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image, size=(384, 384)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])
    
    if isinstance(image, str):  # Если это путь к файлу
        image = Image.open(image).convert('L')
    elif isinstance(image, Image.Image):  # Если это уже PIL.Image
        image = image.convert('L')
    elif isinstance(image, np.ndarray):  # Если это numpy array
        image = Image.fromarray(image).convert('L')
    else:
        raise ValueError("Unsupported image format")
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        output = model(image_tensor)
        mask = (output > threshold).float()
    return mask.squeeze().cpu().numpy()

def visualize(image, mask):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def do_predict(image, model_path="unetplusplus_best_model2.pth", threshold=0.5):
    # Загрузка модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    
    # Предобработка изображения
    image_tensor = preprocess_image(image).to(device)
    
    # Предсказание
    with torch.no_grad():
        output = model(image_tensor)
        mask = (output > threshold).float().squeeze().cpu().numpy()
    
    return mask
