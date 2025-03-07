import numpy as np
import matplotlib.pyplot as plt
from env.pointmaze.maze_model import MazeEnv, U_MAZE, OFF_TARGET
import os
from PIL import Image

class MazeVisualizer:
    """
    A class for visualizing states in the point maze environment.
    """
    def __init__(self, maze_spec=None, output_dir="maze_visualizations", show_target=False):
        """
        Initialize the maze visualizer.
        
        Args:
            maze_spec: The maze specification string (if None, uses the default U-maze)
            output_dir: Directory to save visualizations
            show_target: Whether to show the target (red dot) in the visualization
        """
        # Use default U_MAZE if maze_spec is None
        if maze_spec is None:
            maze_spec = U_MAZE
            
        # Create the environment with state return value first to avoid dtype issues
        self.env = MazeEnv(maze_spec=maze_spec, return_value='state')
        
        # Then set return_value to 'obs' for visualization
        self.env.return_value = 'obs'
        
        # Hide the target if requested
        if not show_target:
            # Move the target off-screen
            self.env.set_target(OFF_TARGET)
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def set_target(self, target_pos=None):
        """
        Set the target position (the red dot).
        
        Args:
            target_pos: A 2D array [x, y] representing the target position.
                       If None, the target will be hidden (moved off-screen).
        """
        if target_pos is None:
            self.env.set_target(OFF_TARGET)
        else:
            self.env.set_target(np.array(target_pos))
        
    def visualize_state(self, state, save_path=None, show=True, target_pos=None):
        """
        Visualize a specific state in the point maze environment.
        
        Args:
            state: A 4D array [x, y, vx, vy] representing the state to visualize
            save_path: Path to save the visualization (if None, uses default naming)
            show: Whether to display the visualization
            target_pos: Optional target position [x, y] to show in the visualization
            
        Returns:
            The rendered image as a numpy array
        """
        # Ensure state is a numpy array
        state = np.array(state, dtype=np.float32)
        
        # Set target position if provided
        if target_pos is not None:
            self.set_target(target_pos)
        
        # Set up the environment for rendering
        self.env.prepare_for_render()
        
        # Set the state (position and velocity)
        self.env.set_state(state[:2], state[2:])
        
        # Render the state
        img = self.env.sim.render(224, 224)
        
        # Save the image if requested
        if save_path is not None:
            plt.imsave(save_path, img)
        elif save_path is None and self.output_dir is not None:
            # Generate a default filename based on the state
            filename = f"maze_state_x{state[0]:.2f}_y{state[1]:.2f}_vx{state[2]:.2f}_vy{state[3]:.2f}.png"
            plt.imsave(os.path.join(self.output_dir, filename), img)
        
        # Display the image if requested
        if show:
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(f"State: x={state[0]:.2f}, y={state[1]:.2f}, vx={state[2]:.2f}, vy={state[3]:.2f}")
            plt.axis('off')
            plt.show()
            
        return img
    
    def visualize_trajectory(self, states, save_path=None, show=True, target_pos=None):
        """
        Visualize a trajectory of states in the point maze environment.
        
        Args:
            states: A list or array of states, each a 4D array [x, y, vx, vy]
            save_path: Path to save the visualization (if None, uses default naming)
            show: Whether to display the visualization
            target_pos: Optional target position [x, y] to show in the visualization
            
        Returns:
            A list of rendered images as numpy arrays
        """
        # Set target position if provided
        if target_pos is not None:
            self.set_target(target_pos)
            
        images = []
        
        for i, state in enumerate(states):
            # Generate a filename for this frame
            if save_path is not None:
                # If save_path is a directory, save each frame in that directory
                if os.path.isdir(save_path):
                    frame_path = os.path.join(save_path, f"frame_{i:04d}.png")
                else:
                    # If save_path has an extension, use it as a base name
                    base, ext = os.path.splitext(save_path)
                    frame_path = f"{base}_{i:04d}{ext}"
            else:
                frame_path = None
                
            # Visualize this state (don't show individual frames)
            # Don't pass target_pos here as we've already set it above
            img = self.visualize_state(state, save_path=frame_path, show=False)
            images.append(img)
        
        # Create a grid of images for display
        if show:
            n_images = len(images)
            cols = min(5, n_images)
            rows = (n_images + cols - 1) // cols
            
            plt.figure(figsize=(3*cols, 3*rows))
            for i, img in enumerate(images):
                plt.subplot(rows, cols, i+1)
                plt.imshow(img)
                plt.title(f"Frame {i}")
                plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return images
    
    def create_gif(self, states, save_path="trajectory.gif", duration=100, target_pos=None):
        """
        Create a GIF animation from a trajectory of states.
        
        Args:
            states: A list or array of states, each a 4D array [x, y, vx, vy]
            save_path: Path to save the GIF
            duration: Duration of each frame in milliseconds
            target_pos: Optional target position [x, y] to show in the visualization
            
        Returns:
            Path to the saved GIF
        """
        # Visualize all states without showing
        images = self.visualize_trajectory(states, show=False, target_pos=target_pos)
        
        # Convert numpy arrays to PIL Images
        pil_images = [Image.fromarray(img) for img in images]
        
        # Save as GIF
        pil_images[0].save(
            save_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0
        )
        
        return save_path

def visualize_state(state, maze_spec=None, save_path=None, show=True, show_target=False, target_pos=None):
    """
    Convenience function to visualize a single state without creating a MazeVisualizer instance.
    
    Args:
        state: A 4D array [x, y, vx, vy] representing the state to visualize
        maze_spec: The maze specification string (if None, uses the default U-maze)
        save_path: Path to save the visualization
        show: Whether to display the visualization
        show_target: Whether to show the default target (red dot)
        target_pos: Optional specific target position [x, y] to show
        
    Returns:
        The rendered image as a numpy array
    """
    visualizer = MazeVisualizer(maze_spec=maze_spec, show_target=show_target)
    return visualizer.visualize_state(state, save_path=save_path, show=show, target_pos=target_pos)

def visualize_trajectory(states, maze_spec=None, save_path=None, show=True, show_target=False, target_pos=None):
    """
    Convenience function to visualize a trajectory without creating a MazeVisualizer instance.
    
    Args:
        states: A list or array of states, each a 4D array [x, y, vx, vy]
        maze_spec: The maze specification string (if None, uses the default U-maze)
        save_path: Path to save the visualization
        show: Whether to display the visualization
        show_target: Whether to show the default target (red dot)
        target_pos: Optional specific target position [x, y] to show
        
    Returns:
        A list of rendered images as numpy arrays
    """
    visualizer = MazeVisualizer(maze_spec=maze_spec, show_target=show_target)
    return visualizer.visualize_trajectory(states, save_path=save_path, show=show, target_pos=target_pos)

def create_trajectory_gif(states, maze_spec=None, save_path="trajectory.gif", duration=100, show_target=False, target_pos=None):
    """
    Convenience function to create a GIF from a trajectory without creating a MazeVisualizer instance.
    
    Args:
        states: A list or array of states, each a 4D array [x, y, vx, vy]
        maze_spec: The maze specification string (if None, uses the default U-maze)
        save_path: Path to save the GIF
        duration: Duration of each frame in milliseconds
        show_target: Whether to show the default target (red dot)
        target_pos: Optional specific target position [x, y] to show
        
    Returns:
        Path to the saved GIF
    """
    visualizer = MazeVisualizer(maze_spec=maze_spec, show_target=show_target)
    return visualizer.create_gif(states, save_path=save_path, duration=duration, target_pos=target_pos)

# Example usage
if __name__ == "__main__":
    # Example 1: Visualize a single state without the target (red dot)
    target_pos = [0.0, 3.1]
    state = [0.5, 0.5, 0.0, 0.0]  # [x, y, vx, vy]
    visualize_state(state, save_path="my_state_plot_top-left.png")
    state = [0.5,3.1,0.0,0.0]
    visualize_state(state, save_path="my_state_plot_bottom-left.png")
    state = [3.1,0.5,0.0,0.0]
    visualize_state(state, save_path="my_state_plot_top-right.png")
    state = [3.1,3.1,0.0,0.0]
    visualize_state(state, save_path="my_state_plot_bottom-right.png")
    
    # Example 2: Visualize a trajectory
    trajectory = [
        [1.0, 1.0, 0.1, 0.1],
        [1.1, 1.1, 0.1, 0.1],
        [1.2, 1.2, 0.1, 0.1],
        [1.3, 1.3, 0.1, 0.1],
        [1.4, 1.4, 0.1, 0.1]
    ]
    # visualize_trajectory(trajectory, show=True, show_target=False)
    
    # Example 3: Create a GIF of a trajectory
    # create_trajectory_gif(trajectory, save_path="example_trajectory.gif", show_target=False) 