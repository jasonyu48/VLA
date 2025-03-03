import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel

class CLIPEncoder(nn.Module):
    def __init__(self, name, feature_key="pooler_output"):
        super().__init__()
        self.name = "clip"
        self.processor = CLIPProcessor.from_pretrained(name)
        self.base_model = CLIPVisionModel.from_pretrained(name)
        self.feature_key = feature_key
        
        # Get embedding dimension from the model's configuration
        self.emb_dim = self.base_model.config.hidden_size
        
        # Since we're using pooled output, this will be 1D
        self.latent_ndim = 1
        
        # Get patch size from model config
        self.patch_size = self.base_model.config.patch_size

    def forward(self, x):
        # import pdb; pdb.set_trace()

        device = x.device
        x = (x * 255).cpu().numpy().astype("uint8")
        
        x = x.transpose(0, 2, 3, 1)
        images = [img for img in x]
        
        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
        # print("-----------pixel_values.shape-----------")
        # print(pixel_values.shape)
        # [512, 3, 224, 224]
        
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
        # print("-----------emb.shape-----------")
        # print(emb.shape)
        # [512, 1, 768]
        return emb 