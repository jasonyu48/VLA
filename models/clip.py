import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel

class CLIPEncoder(nn.Module):
    def __init__(self, name="openai/clip-vit-base-patch32", feature_key="pooler_output"):
        super().__init__()
        self.name = "clip"
        self.processor = CLIPProcessor.from_pretrained(name)
        self.base_model = CLIPVisionModel.from_pretrained(name)
        self.feature_key = feature_key
        
        # CLIP models output 768-dimensional features
        self.emb_dim = 768
        
        # Since we're using pooled output, this will be 1D
        self.latent_ndim = 1
        
        # Get patch size from model config
        self.patch_size = self.base_model.config.patch_size

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x should be [B, C, H, W] with values in [0, 1]
        # Convert to list of PIL images for CLIP processor
        device = x.device
        x = (x * 255).cpu().numpy().astype("uint8")
        
        # Convert to list of images in [H, W, C] format
        x = x.transpose(0, 2, 3, 1)  # [B, H, W, C]
        images = [img for img in x]  # Convert to list of images
        
        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
        
        # Get features
        with torch.no_grad():
            outputs = self.base_model(pixel_values)
            
        if self.feature_key == "pooler_output":
            emb = outputs.pooler_output
        elif self.feature_key == "last_hidden_state":
            emb = outputs.last_hidden_state
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")
            
        # Add dummy patch dimension to match expected shape
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)
            
        return emb 