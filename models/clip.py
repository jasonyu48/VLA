import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

class CLIPEncoder(nn.Module):
    def __init__(self, name, feature_key="pooler_output"):
        super().__init__()
        self.name = "clip"
        # Use the unified CLIP model
        self.processor = CLIPProcessor.from_pretrained(name)
        self.model = CLIPModel.from_pretrained(name)
        self.feature_key = feature_key
        
        # Get embedding dimension from the model's configuration
        self.emb_dim = self.model.config.projection_dim  # This is the common projection space dimension
        
        self.latent_ndim = 1
        
        # Get patch size from model config if available
        if hasattr(self.model.config, 'vision_config'):
            self.patch_size = self.model.config.vision_config.patch_size
        else:
            self.patch_size = 16  # Default patch size

    def forward(self, x):
        """Process images through the CLIP vision encoder"""
        device = x.device
        x = ((x + 1) / 2 * 255).to(torch.uint8)
        
        # Process images
        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get features using the unified model
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
        # Add dummy patch dimension to match expected shape
        outputs = outputs.unsqueeze(1)
            
        return outputs
    
    def encode_text(self, text_list):
        """
        Encode text inputs using the CLIP text encoder
        
        Args:
            text_list: List of text strings to encode
            
        Returns:
            Text embeddings with shape [batch_size, 1, embedding_dim]
        """
        device = next(self.model.parameters()).device
        
        # Process text
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get text features using the unified model
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            
        outputs = outputs.unsqueeze(1)
            
        return outputs