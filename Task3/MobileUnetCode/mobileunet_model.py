import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class MobileUnet(nn.Module):
    def __init__(self, n_classes):
        super(MobileUnet, self).__init__()
        
        # Load the pre-trained MobileNet model with weights
        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
        # Modify the first convolution layer to accept single-channel input
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU6(inplace=True)
        
        # Encoder (Downsampling Path) - Use MobileNet
        self.encoder = mobilenet.features
        
        # Decoder (Upsampling Path) with Depthwise Separable Convolutions and Skip Connections
        self.upconv1 = nn.ConvTranspose2d(1280, 320, kernel_size=2, stride=2)
        self.conv1 = DepthwiseSeparableConv(320 + 96, 320, kernel_size=3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(320, 160, kernel_size=2, stride=2)
        self.conv2 = DepthwiseSeparableConv(160 + 32, 160, kernel_size=3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(160, 64, kernel_size=2, stride=2)
        self.conv3 = DepthwiseSeparableConv(64 + 24, 64, kernel_size=3, padding=1)
        
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = DepthwiseSeparableConv(32 + 16, 32, kernel_size=3, padding=1)
        
        self.upconv5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv5 = DepthwiseSeparableConv(16, 16, kernel_size=3, padding=1)
        
        # Final convolution layer
        self.final_conv = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder[0:2](x)    # 1st block, output: [batch, 32, 64, 64]
        enc2 = self.encoder[2:4](enc1) # 2nd block, output: [batch, 24, 32, 32]
        enc3 = self.encoder[4:7](enc2) # 3rd block, output: [batch, 32, 16, 16]
        enc4 = self.encoder[7:14](enc3) # 4th block, output: [batch, 96, 8, 8]
        enc5 = self.encoder[14:](enc4)  # 5th block, output: [batch, 1280, 4, 4]
        # print(enc1.shape, enc2.shape,enc3.shape,enc4.shape,enc5.shape,)
        # Decoder with skip connections
        dec1 = self.relu(self.upconv1(enc5))      # [batch, 320, 8, 8]
        dec1 = torch.cat((dec1, enc4), dim=1) # Concatenate with enc4, [batch, 320 + 96, 8, 8]
        dec1 = self.conv1(dec1)        # [batch, 320, 8, 8]
        # print('#'*20, 'decoder 1: ', dec1.shape)
        dec2 = self.relu(self.upconv2(dec1))      # [batch, 160, 16, 16]
        dec2 = torch.cat((dec2, enc3), dim=1) # Concatenate with enc3, [batch, 160 + 32, 16, 16]
        dec2 = self.conv2(dec2)        # [batch, 160, 16, 16]
        # print('#'*20, 'decoder 2: ', dec2.shape)
        dec3 = self.relu(self.upconv3(dec2))      # [batch, 64, 32, 32]
        dec3 = torch.cat((dec3, enc2), dim=1) # Concatenate with enc2, [batch, 64 + 24, 32, 32]
        dec3 = self.conv3(dec3)        # [batch, 64, 32, 32]
        # print('#'*20, 'decoder 3: ', dec3.shape)
        dec4 = self.relu(self.upconv4(dec3))      # [batch, 32, 64, 64]
        dec4 = torch.cat((dec4, enc1), dim=1) # Concatenate with enc1, [batch, 32 + 32, 64, 64]
        dec4 = self.conv4(dec4)        # [batch, 32, 64, 64]
        # print('#'*20, 'decoder 4: ', dec4.shape)

        dec5 = self.relu(self.upconv5(dec4))      # [batch, 16, 128, 128]
        dec5 = self.conv5(dec5)        # [batch, 16, 128, 128]
        
        # Final layer
        output = self.final_conv(dec5) # [batch, n_classes, 128, 128]
        return output

if __name__=="__main__":
    # Instantiate the model
    n_classes = 1  # Assuming binary segmentation, change as needed
    model = UNetMobileNet(n_classes)

    # Example input tensor with shape [batch_size, channels, height, width]
    input_tensor = torch.randn(1, 1, 128, 128)

    # Forward pass
    output = model(input_tensor)

    # Output shape
    print(output.shape)
