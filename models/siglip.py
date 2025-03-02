import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel

class SigLIPEncoder(nn.Module):
    def __init__(self, name, feature_key="pooler_output"):
        super().__init__()
        self.name = "siglip"
        self.processor = AutoProcessor.from_pretrained(name)
        self.base_model = AutoModel.from_pretrained(name)
        self.feature_key = feature_key
        
        self.emb_dim = self.base_model.config.vision_config.hidden_size
        
        # Since we're using pooled output, this will be 1D
        self.latent_ndim = 1
        
        # Get patch size from model config
        self.patch_size = self.base_model.config.vision_config.patch_size

    def forward(self, x):
        device = x.device
        x = (x * 255).cpu().numpy().astype("uint8")
        
        x = x.transpose(0, 2, 3, 1)
        images = [img for img in x]
        
        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
        print(pixel_values.shape)
        
        # Get features
        with torch.no_grad():
            outputs = self.base_model(pixel_values)
            
        if self.feature_key == "pooler_output":
            emb = outputs.pooler_output
        elif self.feature_key == "last_hidden_state":
            emb = outputs.last_hidden_state
        else:
            raise ValueError(f"Invalid feature key: {self.feature_key}")
            
        # Add dummy patch dimension to match expected shape
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)
            
        return emb 