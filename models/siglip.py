import torch
import torch.nn as nn
from transformers import SiglipProcessor, SiglipModel

class SigLIPEncoder(nn.Module):
    def __init__(self, name, feature_key="pooler_output"):
        super().__init__()
        self.name = "siglip"
        # Use the unified SigLIP model
        self.processor = SiglipProcessor.from_pretrained(name)
        self.model = SiglipModel.from_pretrained(name)
        self.feature_key = feature_key
        
        # Get embedding dimension from the model's configuration
        self.emb_dim = self.model.config.vision_config.hidden_size

        self.text_emb_dim = self.model.config.text_config.hidden_size
        if self.text_emb_dim != self.emb_dim:
            raise ValueError(f"Text embedding dimension of SigLIP {self.text_emb_dim} does not match image embedding dimension {self.emb_dim}")

        self.latent_ndim = 1
        
        # Get patch size from model config if available
        if hasattr(self.model.config, 'vision_config'):
            self.patch_size = self.model.config.vision_config.patch_size
        else:
            self.patch_size = 16  # Default patch size

    def forward(self, x):
        """Process images through the SigLIP vision encoder"""
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
        Encode text inputs using the SigLIP text encoder
        
        Args:
            text_list: List of text strings to encode
            
        Returns:
            Text embeddings with shape [batch_size, 1, embedding_dim]
        """
        device = next(self.model.parameters()).device
        
        # Process text
        # SigLIP was trained with padding="max_length"
        inputs = self.processor(text=text_list, return_tensors="pt", padding="max_length", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get text features using the unified model
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            
        outputs = outputs.unsqueeze(1)
            
        return outputs 