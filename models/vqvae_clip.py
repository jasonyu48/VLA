import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class GlobalDecoder(nn.Module):
    """
    Decoder specifically designed for global embeddings (like CLIP's pooler_output)
    that don't have spatial structure.
    """
    def __init__(
        self,
        emb_dim=768,
        hidden_dim=512,
        output_size=224,
        n_res_block=4,
        n_res_channel=128,
    ):
        super().__init__()
        
        self.output_size = output_size
        
        # Project embedding to initial feature map (4x4)
        self.project = nn.Linear(emb_dim, hidden_dim * 4 * 4)
        
        # Convolutional decoder
        blocks = []
        
        # Add residual blocks
        for i in range(n_res_block):
            blocks.append(ResBlock(hidden_dim, n_res_channel))
        
        blocks.append(nn.ReLU(inplace=True))
        
        # Progressive upsampling to reach target size
        # From 4x4 to 8x8
        blocks.append(nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        
        # From 8x8 to 16x16
        blocks.append(nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        
        # From 16x16 to 32x32
        blocks.append(nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, stride=2, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        
        # From 32x32 to 64x64
        blocks.append(nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, 4, stride=2, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        
        # Final layer to get 3 channels
        blocks.append(nn.ConvTranspose2d(hidden_dim // 16, 3, 4, stride=2, padding=1))
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, input):
        """
        input: (batch, time, 1, emb_dim) - CLIP's global embedding
        output: (batch*time, 3, output_size, output_size)
        """
        b, t, n, e = input.shape
        
        # First, squeeze the n dimension (which should be 1)
        input = input.squeeze(2)  # Now shape is (batch, time, emb_dim)
        
        # Reshape to (batch*time, emb_dim)
        x = rearrange(input, 'b t e -> (b t) e')
        
        # Project to initial feature map
        x = self.project(x)
        x = x.view(b * t, -1, 4, 4)  # (batch*time, hidden_dim, 4, 4)
        
        # Apply convolutional decoder
        x = self.blocks(x)  # (batch*time, 3, 128, 128)
        
        # Resize to target output size if needed
        if x.shape[2] != self.output_size:
            x = F.interpolate(
                x, 
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False
            )
        
        return x, torch.zeros(1).to(x.device)  # Return dummy diff for compatibility


class VQVAEClip(nn.Module):
    """
    A wrapper class that uses GlobalDecoder for CLIP's single-patch embeddings
    """
    def __init__(
        self,
        emb_dim,  # Required parameter, provided by train.py from encoder.emb_dim
        n_res_block=4,
        n_res_channel=128,
        output_size=224,
        quantize=False,
        n_embed=2048,  # Not used if quantize=False
    ):
        super().__init__()
        
        self.decoder = GlobalDecoder(
            emb_dim=emb_dim,
            hidden_dim=512,
            output_size=output_size,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
        )
        
        self.quantize = quantize
        
    def forward(self, input):
        """
        input: (batch, time, num_patches, emb_dim)
        For CLIP: num_patches = 1
        """
        # No quantization for now
        quant_b, diff_b = input, torch.zeros(1).to(input.device)
        
        # Decode
        dec, _ = self.decoder(quant_b)
        
        return dec, diff_b
    
    def decode(self, quant_b):
        dec, _ = self.decoder(quant_b)
        return dec 