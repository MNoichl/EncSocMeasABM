def get_default_params(area_size=100):
    """
    Returns a dictionary of default parameters for the boids simulation.
    
    Parameters:
    -----------
    area_size : float, default=100
        Size of the simulation area.
        
    Returns:
    --------
    params : dict
        Dictionary of simulation parameters.
    """
    return {
        "separation_radius": 10.0,
        "alignment_radius": 15.0,
        "cohesion_radius": 15.5,
        "separation_weight": 1.4,
        "alignment_weight": 0.01,
        "cohesion_weight": 0.01,
        "max_speed": 5.0,
        "area_size": area_size
    } 