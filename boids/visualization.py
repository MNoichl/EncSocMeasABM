import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches

from .geometry import plot_toroidal_segment

def create_animation(trajectories, velocities_history=None, area_size=100, interval=100):
    """
    Creates an animation of the boids simulation.
    
    Parameters:
    -----------
    trajectories : ndarray
        Array of shape (num_iterations+1, N, dim) containing the positions at each time step.
    velocities_history : ndarray, optional
        Array of shape (num_iterations+1, N, dim) containing the velocities at each time step.
        If None, velocities are approximated from position differences.
    area_size : float, default=100
        Size of the simulation area.
    interval : int, default=100
        Interval between frames in milliseconds.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object.
    anim : matplotlib.animation.FuncAnimation
        The animation object.
    """
    num_frames, N, dim = trajectories.shape
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_aspect('equal')
    ax.set_title("Boids Simulation")
    
    # Remove grid lines and tick numbers, but keep the border
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)
        ax.set_aspect('equal')
        ax.set_title(f"Boids Simulation - Frame {frame}")
        
        # Remove grid lines and tick numbers, but keep the border
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
        
        positions = trajectories[frame]
        
        # Determine velocities for this frame
        if velocities_history is not None:
            velocities = velocities_history[frame]
        elif frame == 0:
            # For the first frame, we can't calculate from position differences
            # Just use zero velocities
            velocities = np.zeros_like(positions)
        else:
            velocities = trajectories[frame] - trajectories[frame-1]
        
        for i in range(N):
            pos = positions[i]
            vel = velocities[i]
            
            # Normalize velocity to get direction
            speed = np.linalg.norm(vel)
            if speed > 0:
                heading = vel / speed
            else:
                heading = np.array([1, 0])  # Default heading if velocity is zero
            
            # Get perpendicular vector (rotate 90 degrees)
            perp = np.array([-heading[1], heading[0]])
            
            # Create custom triangle with narrow back
            size = 2.0  # Scale factor for triangle size
            
            # Vertices: tip at front, narrower at back
            tip = pos + heading * size
            left_back = pos - heading * size + perp * (size/2)  # Half as wide at the back
            right_back = pos - heading * size - perp * (size/2)  # Half as wide at the back
            
            # Changed color to black and made semi-transparent
            triangle = patches.Polygon(np.array([tip, left_back, right_back]), closed=True, 
                                      color='black', alpha=0.7, zorder=10)
            ax.add_patch(triangle)
        
        return ax.patches
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=False)
    return fig, anim

def save_animation(anim, filename="boids.gif", fps=10, writer='pillow'):
    """
    Saves the animation to a file.
    
    Parameters:
    -----------
    anim : matplotlib.animation.FuncAnimation
        The animation to save.
    filename : str, default="boids.gif"
        The name of the output file.
    fps : int, default=10
        Frames per second.
    writer : str, default='pillow'
        The writer to use.
    """
    anim.save(filename, writer=writer, fps=fps)
    
def create_trajectory_plot(trajectories, velocities_history=None, area_size=100, show_final_positions=True):
    """
    Creates a static plot showing the trajectories and optionally the final positions of the boids.
    
    Parameters:
    -----------
    trajectories : ndarray
        Array of shape (num_iterations+1, N, dim) containing the positions at each time step.
    velocities_history : ndarray, optional
        Array of shape (num_iterations+1, N, dim) containing the velocities at each time step.
        Used only if show_final_positions is True.
    area_size : float, default=100
        Size of the simulation area.
    show_final_positions : bool, default=True
        Whether to show the final positions as triangles.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    num_frames, N, dim = trajectories.shape
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_aspect('equal')
    ax.set_title("Boids Trajectories with Direction Indicators")
    
    # Remove grid lines and tick numbers, but keep the border
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    # For each boid, draw its trajectory with older segments more transparent.
    for i in range(N):
        traj = trajectories[:, i, :]
        num_points = traj.shape[0]
        for t in range(num_points - 1):
            # Make older segments more transparent, but with max transparency of 0.9
            alpha = 0.1 + 0.8 * (t / (num_points - 1))  # Will range from 0.1 to 0.9
            # Use our helper function to plot the segment with proper wrapping.
            plot_toroidal_segment(ax, traj[t], traj[t+1], area_size, color='#0f6a9a', alpha=alpha, linewidth=1)
    
    # Add triangle markers at the final positions with their corresponding directions.
    if show_final_positions and velocities_history is not None:
        final_positions = trajectories[-1]
        final_velocities = velocities_history[-1]
        
        for i in range(N):
            pos = final_positions[i]
            vel = final_velocities[i]
            
            # Normalize velocity to get direction
            speed = np.linalg.norm(vel)
            if speed > 0:
                heading = vel / speed
            else:
                heading = np.array([1, 0])  # Default heading if velocity is zero
            
            # Get perpendicular vector (rotate 90 degrees)
            perp = np.array([-heading[1], heading[0]])
            
            # Create custom triangle with narrow back
            size = 2.0  # Scale factor for triangle size
            
            # Vertices: tip at front, narrower at back
            tip = pos + heading * size
            left_back = pos - heading * size + perp * (size/2)  # Half as wide at the back
            right_back = pos - heading * size - perp * (size/2)  # Half as wide at the back
            
            triangle = patches.Polygon(np.array([tip, left_back, right_back]), closed=True, 
                                      color='black', zorder=10, alpha=0.6)
            ax.add_patch(triangle)
    
    return fig, ax