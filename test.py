import torch
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from PIL import Image
import numpy as np

def compute_clip_similarity(text: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model from Hugging Face
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    # Create a dummy black image (all zeros)
    dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Process the image and text
    inputs = processor(text=[text], images=[dummy_image], return_tensors="pt", padding=True).to(device)
    
    # Encode image and text using CLIPModel
    with torch.no_grad():
        image_features_clip = model.get_image_features(**{k: v for k, v in inputs.items() if k.startswith("pixel_values")})
        text_features = model.get_text_features(**{k: v for k, v in inputs.items() if k.startswith("input_ids")})
        
        # Encode image using CLIPVisionModel
        vision_features = vision_model(**{k: v for k, v in inputs.items() if k.startswith("pixel_values")}).last_hidden_state
    
    # Normalize features
    image_features_clip /= image_features_clip.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity
    similarity = (image_features_clip @ text_features.T).item()
    
    print(f"CLIPModel image encoding shape: {image_features_clip.shape}")
    print(f"CLIPVisionModel image encoding shape: {vision_features.shape}")
    
    return similarity

if __name__ == "__main__":
    text_input = "A black square"
    similarity_score = compute_clip_similarity(text_input)
    print(f"Similarity Score: {similarity_score}")
