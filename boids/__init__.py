from .core import simulate_boids
from .visualization import create_animation, create_trajectory_plot, save_animation
from .params import get_default_params

__all__ = ['simulate_boids', 'create_animation', 'create_trajectory_plot', 
           'save_animation', 'get_default_params'] 