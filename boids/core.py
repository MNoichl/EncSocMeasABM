import numpy as np
from scipy.spatial import cKDTree

from .geometry import build_extended, query_neighbors

def boids_update(positions, velocities, params, kd_tree, extended_indices, extended_positions):
    """
    Update boid velocities based on separation, alignment, and cohesion rules.
    The neighbor search is done in toroidal space using an extended KDTree.
    """
    N = positions.shape[0]
    new_velocities = np.copy(velocities)
    area_size = params["area_size"]
    
    for i in range(N):
        pos_i = positions[i]
        vel_i = velocities[i]
        
        # Separation: steer to avoid crowding neighbors.
        neighbors_sep, disps_sep = query_neighbors(i, params["separation_radius"],
                                                 positions, kd_tree, extended_indices, extended_positions, area_size)
        separation_force = np.zeros_like(pos_i)
        if len(neighbors_sep) > 0:
            for disp in disps_sep:
                # Create stronger repulsion with inverse square law
                distance = np.linalg.norm(disp)
                if distance > 0:
                    # The negative sign ensures separation (moves away from neighbor)
                    # The distance squared creates much stronger repulsion at close range
                    repulsion = -disp / (distance * distance + 0.1)  # Adding 0.1 prevents division by zero
                    separation_force += repulsion
        
        # Alignment: steer towards the average velocity of neighbors.
        neighbors_align, _ = query_neighbors(i, params["alignment_radius"],
                                           positions, kd_tree, extended_indices, extended_positions, area_size)
        alignment_force = np.zeros_like(vel_i)
        if len(neighbors_align) > 0:
            avg_velocity = np.zeros_like(vel_i)
            for j in neighbors_align:
                avg_velocity += velocities[j]
            avg_velocity /= len(neighbors_align)
            alignment_force = avg_velocity - vel_i
        
        # Cohesion: steer towards the average position of neighbors.
        neighbors_cohesion, disps_cohesion = query_neighbors(i, params["cohesion_radius"],
                                                           positions, kd_tree, extended_indices, extended_positions, area_size)
        cohesion_force = np.zeros_like(pos_i)
        if len(neighbors_cohesion) > 0:
            avg_disp = np.zeros_like(pos_i)
            for disp in disps_cohesion:
                avg_disp += disp
            avg_disp /= len(neighbors_cohesion)
            cohesion_force = avg_disp
        
        acceleration = (params["separation_weight"] * separation_force +
                        params["alignment_weight"] * alignment_force +
                        params["cohesion_weight"] * cohesion_force)
        
        new_velocities[i] = vel_i + acceleration
        
        # Limit the speed to max_speed.
        speed = np.linalg.norm(new_velocities[i])
        if speed > params["max_speed"]:
            new_velocities[i] = new_velocities[i] / speed * params["max_speed"]
    
    return new_velocities

def simulate_boids(initial_positions=None, initial_velocities=None, num_iterations=200, dt=1.0, params=None, num_boids=50, dim=2, area_size=100):
    """
    Runs the boids simulation with toroidal wrapping.
    
    Parameters:
    -----------
    initial_positions : ndarray, optional
        Array of initial positions. If None, random positions are generated.
    initial_velocities : ndarray, optional
        Array of initial velocities. If None, random velocities are generated.
    num_iterations : int, default=200
        Number of simulation steps.
    dt : float, default=1.0
        Time step size.
    params : dict, optional
        Simulation parameters. If None, default parameters are used.
    num_boids : int, default=50
        Number of boids (used only if initial_positions is None).
    dim : int, default=2
        Dimensionality of the simulation (used only if initial_positions is None).
    area_size : float, default=100
        Size of the simulation area (used only if params is None).
        
    Returns:
    --------
    trajectories : ndarray
        Array of shape (num_iterations+1, N, dim) containing the positions at each time step.
    velocities_history : ndarray
        Array of shape (num_iterations+1, N, dim) containing the velocities at each time step.
    """
    from .params import get_default_params
    
    # Generate random initial conditions if not provided
    if initial_positions is None:
        initial_positions = np.random.rand(num_boids, dim) * area_size
    
    if initial_velocities is None:
        initial_velocities = (np.random.rand(initial_positions.shape[0], dim) - 0.5) * 10
    
    # Use default parameters if not provided
    if params is None:
        params = get_default_params(area_size)
    
    positions = initial_positions.copy()
    velocities = initial_velocities.copy()
    trajectories = [positions.copy()]
    velocities_history = [velocities.copy()]
    
    for _ in range(num_iterations):
        extended_positions, extended_indices = build_extended(positions, params["area_size"])
        kd_tree = cKDTree(extended_positions)
        velocities = boids_update(positions, velocities, params, kd_tree, extended_indices, extended_positions)
        positions = positions + velocities * dt
        # Apply toroidal wrapping
        positions = positions % params["area_size"]
        
        trajectories.append(positions.copy())
        velocities_history.append(velocities.copy())
    
    return np.array(trajectories), np.array(velocities_history) 