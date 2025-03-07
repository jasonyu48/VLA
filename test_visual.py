import numpy as np
import matplotlib.pyplot as plt
import torch
from maze_visualizer import visualize_state
import os

def create_goal_visualization(goal):
    """
    Create a goal visualization image based on the goal description.
    
    Args:
        goal: String description of the goal ('bottom-left', 'top-left', 'top-right', 'bottom-right')
        
    Returns:
        A torch tensor representing the goal visualization
    """
    img = np.zeros((224, 224))
    if 'bottom-left' in goal:
        img[112:, :112] = 1
    elif 'top-left' in goal:
        img[:112, :112] = 1
    elif 'top-right' in goal:
        img[:112, 112:] = 1
    elif 'bottom-right' in goal:
        img[112:, 112:] = 1
    
    # Convert to 3-channel tensor
    tensor_img = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1)
    return tensor_img, img

def test_goal_visualizations():
    """
    Test if the goal visualization code correctly highlights the right quadrant in white.
    """
    # Create output directory
    output_dir = "goal_visualization_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test all four goal positions
    goals = ['bottom-left', 'top-left', 'top-right', 'bottom-right']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    # Create and visualize each goal
    for i, goal in enumerate(goals):
        tensor_img, np_img = create_goal_visualization(goal)
        
        # Plot the goal visualization
        axes[i].imshow(np_img, cmap='gray')
        axes[i].set_title(f"Goal: {goal}")
        axes[i].axis('off')
        
        # Save individual image
        plt.figure(figsize=(5, 5))
        plt.imshow(np_img, cmap='gray')
        plt.title(f"Goal: {goal}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"goal_{goal}.png"))
        plt.close()
        
        # Also save the tensor as an image
        tensor_np = tensor_img.numpy()
        tensor_np = np.transpose(tensor_np, (1, 2, 0))  # Convert from CxHxW to HxWxC
        plt.figure(figsize=(5, 5))
        plt.imshow(tensor_np)
        plt.title(f"Goal Tensor: {goal}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"goal_tensor_{goal}.png"))
        plt.close()
    
    # Add a title to the figure
    fig.suptitle("Goal Visualization Tests", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig(os.path.join(output_dir, "all_goals.png"))
    plt.close()
    
    print(f"Goal visualization tests completed. Images saved to {output_dir}/")

def test_maze_state_visualization():
    """
    Test if the maze state visualization correctly shows the agent at the specified positions.
    """
    # Create output directory
    output_dir = "maze_state_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define states corresponding to the four corners according to plan.py
    states = [
        [0.5, 3.1, 0.0, 0.0],  # bottom-left
        [0.5, 0.5, 0.0, 0.0],  # top-left
        [3.1, 0.5, 0.0, 0.0],  # top-right
        [3.1, 3.1, 0.0, 0.0]   # bottom-right
    ]
    
    labels = ['bottom-left', 'top-left', 'top-right', 'bottom-right']
    
    # Visualize each state
    for state, label in zip(states, labels):
        # Save the visualization
        visualize_state(
            state, 
            save_path=os.path.join(output_dir, f"maze_state_{label}.png"),
            show=False,
            show_target=False
        )
    
    print(f"Maze state visualization tests completed. Images saved to {output_dir}/")

def compare_goal_and_state():
    """
    Compare the goal visualization with the actual maze state visualization.
    """
    # Create output directory
    output_dir = "comparison_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define states corresponding to the four corners according to plan.py
    states = [
        [0.5, 3.1, 0.0, 0.0],  # bottom-left
        [0.5, 0.5, 0.0, 0.0],  # top-left
        [3.1, 0.5, 0.0, 0.0],  # top-right
        [3.1, 3.1, 0.0, 0.0]   # bottom-right
    ]
    
    labels = ['bottom-left', 'top-left', 'top-right', 'bottom-right']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(10, 20))
    
    for i, (state, label) in enumerate(zip(states, labels)):
        # Get the goal visualization
        _, goal_img = create_goal_visualization(label)
        
        # Visualize the goal
        axes[i, 0].imshow(goal_img, cmap='gray')
        axes[i, 0].set_title(f"Goal: {label}")
        axes[i, 0].axis('off')
        
        # Visualize the maze state
        maze_img = visualize_state(state, show=False, show_target=False)
        axes[i, 1].imshow(maze_img)
        axes[i, 1].set_title(f"Maze State: {label}")
        axes[i, 1].axis('off')
    
    # Add a title to the figure
    fig.suptitle("Comparison: Goal Visualization vs Maze State", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust for the suptitle
    plt.savefig(os.path.join(output_dir, "comparison.png"))
    plt.close()
    
    print(f"Comparison tests completed. Images saved to {output_dir}/")

if __name__ == "__main__":
    # Run all tests
    test_goal_visualizations()
    # test_maze_state_visualization()
    # compare_goal_and_state()
    
    print("All tests completed successfully!") 