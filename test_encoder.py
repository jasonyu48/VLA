import os
import torch
import numpy as np
import argparse
from PIL import Image
from transformers import SiglipProcessor, SiglipModel, CLIPProcessor, CLIPModel
from pathlib import Path

def load_model_and_processor(model_name="google/siglip-so400m-patch14-224"):
    """Load the model and processor based on model name."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if "siglip" in model_name:
        processor = SiglipProcessor.from_pretrained(model_name)
        model = SiglipModel.from_pretrained(model_name)
        model_type = "siglip"
    elif "clip" in model_name:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model_type = "clip"
    else:
        raise ValueError(f"Unsupported model: {model_name}. Use a SigLIP or CLIP model.")
    
    # Move model to GPU if available
    model = model.to(device)
    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    
    return model, processor, device, model_type

def encode_images(model, processor, image_paths, device, model_type):
    """Encode images using the model."""
    image_embeddings = []
    
    for img_path in image_paths:
        # Open image and convert to RGB to ensure proper channel format
        image = Image.open(img_path).convert('RGB')
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if model_type == "siglip":
                # SigLIP image encoding
                image_embedding = model.get_image_features(**inputs)
            elif model_type == "clip":
                # CLIP image encoding
                outputs = model.get_image_features(**inputs)
                image_embedding = outputs
            
            # Normalize embedding
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        image_embeddings.append(image_embedding.squeeze().cpu().numpy())
    
    return np.array(image_embeddings)

def encode_texts(model, processor, texts, device, model_type):
    """Encode text descriptions using the model."""
    text_embeddings = []
    
    for text in texts:
        # Process text
        if model_type == "siglip":
            inputs = processor(text=text, padding="max_length", truncation=True, return_tensors="pt")
        elif model_type == "clip":
            inputs = processor(text=text, padding=True, truncation=True, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if model_type == "siglip":
                # SigLIP text encoding
                text_embedding = model.get_text_features(**inputs)
            elif model_type == "clip":
                # CLIP text encoding
                outputs = model.get_text_features(**inputs)
                text_embedding = outputs
            
            # Normalize embedding
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        
        text_embeddings.append(text_embedding.squeeze().cpu().numpy())
    
    return np.array(text_embeddings)

def calculate_dot_products(image_embeddings, text_embeddings):
    """Calculate dot products between all image and text embeddings."""
    dot_products = np.zeros((len(image_embeddings), len(text_embeddings)))
    
    for i in range(len(image_embeddings)):
        for j in range(len(text_embeddings)):
            dot_products[i, j] = np.dot(image_embeddings[i], text_embeddings[j])
    
    return dot_products

def check_correspondence(dot_products):
    """Check if the highest dot product for each image is with its corresponding text."""
    correct_matches = 0
    total = len(dot_products)
    
    for i in range(total):
        max_index = np.argmax(dot_products[i])
        if max_index == i:
            correct_matches += 1
    
    return correct_matches, total

def parse_args():
    parser = argparse.ArgumentParser(description="Test image-text alignment with different models")
    parser.add_argument("--model", type=str, default="google/siglip-so400m-patch14-224",
                        help="Model to use (e.g., google/siglip-so400m-patch14-224 or openai/clip-vit-large-patch14)")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Define image paths and text descriptions
    image_dir = "/home/jyu48/dino_wm/maze_state_tests"
    image_paths = [
        os.path.join(image_dir, "maze_state_bottom-left.png"),
        os.path.join(image_dir, "maze_state_top-left.png"),
        os.path.join(image_dir, "maze_state_top-right.png"),
        os.path.join(image_dir, "maze_state_bottom-right.png"),
    ]
    
    text_descriptions = [
        "A green dot is at the bottom-left of a U-shaped maze with a thick orange border on a blue checkered background. A red dot is at the top-left, outside the maze.",
        "A green dot is at the top-left of a U-shaped maze with a thick orange border on a blue checkered background. A red dot is at the top-left, outside the maze.",
        "A green dot is at the top-right of a U-shaped maze with a thick orange border on a blue checkered background. A red dot is at the top-left, outside the maze.",
        "A green dot is at the bottom-right of a U-shaped maze with a thick orange border on a blue checkered background. A red dot is at the top-left, outside the maze."
    ]
    
    # Load model and processor
    print(f"Loading model: {args.model}...")
    model, processor, device, model_type = load_model_and_processor(args.model)
    
    # Encode images and texts
    print("Encoding images...")
    image_embeddings = encode_images(model, processor, image_paths, device, model_type)
    
    print("Encoding text descriptions...")
    text_embeddings = encode_texts(model, processor, text_descriptions, device, model_type)
    
    # Calculate dot products
    print("Calculating dot products...")
    dot_products = calculate_dot_products(image_embeddings, text_embeddings)
    
    # Print the dot product matrix
    print(f"\nDot Product Matrix for {args.model}:")
    print("             | Bottom-Left | Top-Left | Top-Right | Bottom-Right")
    print("-------------|-------------|----------|-----------|-------------")
    for i, position in enumerate(["Bottom-Left", "Top-Left", "Top-Right", "Bottom-Right"]):
        row = f"{position:12} |"
        for j in range(len(text_descriptions)):
            row += f" {dot_products[i, j]:.6f} |"
        print(row)
    
    # Check if corresponding pairs have the highest dot products
    correct_matches, total = check_correspondence(dot_products)
    print(f"\nCorrect matches: {correct_matches}/{total}")
    
    # Check if each image's highest dot product is with its corresponding text
    print("\nChecking highest dot product for each image:")
    for i, position in enumerate(["Bottom-Left", "Top-Left", "Top-Right", "Bottom-Right"]):
        max_index = np.argmax(dot_products[i])
        max_value = dot_products[i, max_index]
        correct = max_index == i
        print(f"Image: {position}, Highest match: {['Bottom-Left', 'Top-Left', 'Top-Right', 'Bottom-Right'][max_index]} (value: {max_value:.6f}), Correct: {correct}")
    
    # Check if each text's highest dot product is with its corresponding image
    print("\nChecking highest dot product for each text:")
    dot_products_transposed = dot_products.T
    for i, position in enumerate(["Bottom-Left", "Top-Left", "Top-Right", "Bottom-Right"]):
        max_index = np.argmax(dot_products_transposed[i])
        max_value = dot_products_transposed[i, max_index]
        correct = max_index == i
        print(f"Text: {position}, Highest match: {['Bottom-Left', 'Top-Left', 'Top-Right', 'Bottom-Right'][max_index]} (value: {max_value:.6f}), Correct: {correct}")

if __name__ == "__main__":
    main() 