import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            # (Batch_size, Channel, height, width) -> (Batch_size, 128, height, width) 
            # Height and Width are mantained because we added padding, which means that we are 
            # adding pixels tu the image so that with the convolution the size is mantained.
            nn.Conv2d(3, 128,kernel_size=3, padding=1),
            
            # (Batch_size, 128, height, width) -> (Batch_size, 128, height, width) 
            VAE_ResidualBlock(128,128),
            
            # (Batch_size, 128, height, width) -> (Batch_size, 128, height, width) 
            VAE_ResidualBlock(128,128),
            
            # (Batch_size, 128, height, width) -> (Batch_size, 128, height / 2, width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_size, 128, height / 2, width / 2) -> (Batch_size, 256, height / 2, width / 2)            
            VAE_ResidualBlock(128,256),

            # (Batch_size, 256, height / 2, width / 2) -> (Batch_size, 256, height / 2, width / 2)                        
            VAE_ResidualBlock(256, 256),
            
            # (Batch_size, 256, height / 2, width / 2) -> (Batch_size, 256, height / 4, width / 4)            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            # (Batch_size, 256, height / 4, width / 4) -> (Batch_size, 256, height / 4, width / 4)
            VAE_ResidualBlock(256, 512),
            
            # (Batch_size, 512, height / 4, width / 4) -> (Batch_size, 512, height / 4, width / 4)
            VAE_ResidualBlock(512, 512),
            
            # (Batch_size, 512, height / 4, width / 4) -> (Batch_size, 512, height / 8, width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            # (Batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)            
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)                        
            VAE_AttentionBlock(512),
            
            # (Batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)            
            VAE_ResidualBlock(512, 512),
            
            # (Batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)            
            nn.GroupNorm(32, 512),
            
            # (Batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)            
            nn.SiLU(),
            
            
            # Because the padding=1, it means the width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8). 
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            # (Batch_size, 8, height / 8, width / 8) -> (Batch_size, 8, height / 8, width / 8)            
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channel, height, width)
        #noise: (Batch_size, Out_channels, height / 8, width / 8)
        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (Padding_left, Padding_right, Padding_top, Padding_bottom )
                # Pad with zeros on the right and bottom.
                x = F.pad(x, (0, 1, 0, 1)) #Apply asymetrical padding only on convolutions with stride = 2
            x = module(x)
            
        # (Batch_size, 8, height, height / 8, width / 8) -> two tensors of shape (Batch_size, 4, height / 8, width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 4, height / 8, width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        
        # (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 4, height / 8, width / 8)
        variance = log_variance.exp()
        
        # (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 4, height / 8, width / 8)
        stdev = variance.sqrt()
        
        # Z = N(0, 1) -> N(mean, variance) = X?
        # X = mean + stdev * Z This is why we get the noise, we want a particular seed from the noise and we sample from the distribution
        x = mean + stdev * noise
        
        #Scale the output by a constant
        x *= 0.18215 #This is a constant found in the original repo, no justification to why
        
        return x 