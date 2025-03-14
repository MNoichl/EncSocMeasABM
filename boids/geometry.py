import numpy as np
from scipy.spatial import cKDTree

def toroidal_difference(pos1, pos2, area_size):
    """
    Compute the minimum image difference between two positions in a toroidal space.
    """
    diff = pos2 - pos1
    diff[diff > area_size/2] -= area_size
    diff[diff < -area_size/2] += area_size
    return diff

def build_extended(positions, area_size):
    """
    Create an extended array of positions by replicating the positions with shifts
    in both x and y. This allows us to use a KDTree to approximate toroidal distances.
    """
    N = positions.shape[0]
    shifts = np.array([[-area_size, -area_size],
                      [-area_size, 0],
                      [-area_size, area_size],
                      [0, -area_size],
                      [0, 0],
                      [0, area_size],
                      [area_size, -area_size],
                      [area_size, 0],
                      [area_size, area_size]])
    extended_positions = []
    extended_indices = []
    for shift in shifts:
        extended_positions.append(positions + shift)
        extended_indices.append(np.arange(N))
    extended_positions = np.vstack(extended_positions)
    extended_indices = np.hstack(extended_indices)
    return extended_positions, extended_indices

def query_neighbors(i, radius, positions, kd_tree, extended_indices, extended_positions, area_size):
    """
    Query neighbors for boid i using the KDTree on the extended positions.
    Returns unique neighbor indices (excluding self) along with the toroidal displacement vectors.
    """
    pos_i = positions[i]
    candidate_indices = kd_tree.query_ball_point(pos_i, radius)
    neighbors = []
    displacements = []
    for cand in candidate_indices:
        j = extended_indices[cand]
        candidate_pos = extended_positions[cand]
        diff = toroidal_difference(pos_i, candidate_pos, area_size)
        distance = np.linalg.norm(diff)
        # Skip self (distance nearly zero)
        if j == i and distance < 1e-8:
            continue
        if distance < radius:
            if j not in neighbors:
                neighbors.append(j)
                displacements.append(diff)
    return neighbors, displacements

def plot_toroidal_segment(ax, p0, p1, area_size, color, alpha, linewidth):
    """
    Plots a line segment between two wrapped points p0 and p1 so that if a boid crosses the boundary,
    the line is split at the border instead of drawing a spurious long line.
    """
    diff = toroidal_difference(p0, p1, area_size)
    p1_corr = p0 + diff  # unwrapped endpoint relative to p0
    # If the unwrapped endpoint lies within the domain, we can plot directly.
    if (0 <= p1_corr[0] <= area_size) and (0 <= p1_corr[1] <= area_size):
        ax.plot([p0[0], p1_corr[0]], [p0[1], p1_corr[1]], color=color, alpha=alpha, linewidth=linewidth)
    else:
        # Handle wrapping in x-axis if needed.
        if p1_corr[0] < 0 or p1_corr[0] > area_size:
            if p1_corr[0] < 0:
                t = (0 - p0[0]) / (p1_corr[0] - p0[0])
                x_bound = 0
                new_x = area_size
            else:  # p1_corr[0] > area_size
                t = (area_size - p0[0]) / (p1_corr[0] - p0[0])
                x_bound = area_size
                new_x = 0
            intersection = p0 + t * (p1_corr - p0)
            ax.plot([p0[0], intersection[0]], [p0[1], intersection[1]], color=color, alpha=alpha, linewidth=linewidth)
            # Start new segment from the opposite boundary at the same y-coordinate as the intersection.
            new_start = np.array([new_x, intersection[1]])
            # The remainder of the displacement:
            remainder = p1_corr - intersection
            new_end = new_start + remainder
            ax.plot([new_start[0], new_end[0]], [new_start[1], new_end[1]], color=color, alpha=alpha, linewidth=linewidth)
        # Handle wrapping in y-axis (if x was okay but y is out-of-bound)
        elif p1_corr[1] < 0 or p1_corr[1] > area_size:
            if p1_corr[1] < 0:
                t = (0 - p0[1]) / (p1_corr[1] - p0[1])
                y_bound = 0
                new_y = area_size
            else:  # p1_corr[1] > area_size
                t = (area_size - p0[1]) / (p1_corr[1] - p0[1])
                y_bound = area_size
                new_y = 0
            intersection = p0 + t * (p1_corr - p0)
            ax.plot([p0[0], intersection[0]], [p0[1], intersection[1]], color=color, alpha=alpha, linewidth=linewidth)
            new_start = np.array([intersection[0], new_y])
            remainder = p1_corr - intersection
            new_end = new_start + remainder
            ax.plot([new_start[0], new_end[0]], [new_start[1], new_end[1]], color=color, alpha=alpha, linewidth=linewidth) 