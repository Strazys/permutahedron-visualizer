import streamlit as st
import plotly.graph_objects as go
import numpy as np
import itertools
from collections import deque

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Permutahedron Explorer", layout="wide")
st.title("Permutahedron Shortest Path")
st.markdown("""
This visualizer finds the **shortest path** between two permutations.
It uses **Breadth-First Search (BFS)** on the graph of adjacent swaps.
The path is forced to follow the edges; it cannot cut through the interior.
""")

# --- 2. MATHEMATICAL SETUP ---
order = 4
items = range(1, order+1)
original_perms_list = [*itertools.permutations(items)]
permuted_items = np.array(original_perms_list)

# 3D Projection Matrix (Archimedean Solid projection)
A = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0, 0],
              [np.sqrt(6)/6, np.sqrt(6)/6, -np.sqrt(2/3), 0],
              [np.sqrt(12)/12, np.sqrt(12)/12, np.sqrt(12)/12, -np.sqrt(3)/2]])

# Apply projection
projected_items = np.einsum('ik,ak->ai', A, permuted_items)

# Map every permutation tuple to its exact 3D coordinate
coord_map = {perm: coord for perm, coord in zip(original_perms_list, projected_items)}

def get_neighbors(perm):
    """
    Returns neighbors reachable by exactly one adjacent swap.
    This DEFINES the edges of the permutahedron.
    """
    perm = list(perm)
    # Loop through adjacent pairs (0,1), (1,2), (2,3)
    for i in range(len(perm) - 1):
        p_new = perm[:]
        # Swap
        p_new[i], p_new[i+1] = p_new[i+1], p_new[i]
        yield tuple(p_new)

def bfs_shortest_path(start, end):
    """
    Standard BFS to find the shortest path in terms of number of edges (swaps).
    """
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# --- 3. USER INTERFACE ---
col1, col2 = st.columns(2)
with col1:
    start_input = st.selectbox("Start Node", options=original_perms_list, index=0)
with col2:
    # Default to index 7 just to give a nice initial path
    end_input = st.selectbox("End Node", options=original_perms_list, index=7)

# --- 4. BUILD THE BACKGROUND MESH ---
# We build the mesh using the EXACT same neighbor logic as the pathfinding
# to ensure they match perfectly.
edge_x, edge_y, edge_z = [], [], []
seen_edges = set()

for perm in original_perms_list:
    u = coord_map[perm]
    for neighbor in get_neighbors(perm):
        # Create a sorted tuple ID so we don't draw the same edge twice (A-B and B-A)
        edge_id = tuple(sorted((perm, neighbor)))
        
        if edge_id not in seen_edges:
            seen_edges.add(edge_id)
            v = coord_map[neighbor]
            # Add line segment: u -> v -> None
            edge_x.extend([u[0], v[0], None])
            edge_y.extend([u[1], v[1], None])
            edge_z.extend([u[2], v[2], None])

fig = go.Figure()

# Plot the Wireframe (White)
fig.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='white', width=3),
    hoverinfo='none', # Disable hover on the grid to reduce clutter
    name='Edges'
))

# --- 5. CALCULATE AND PLOT THE PATH ---
path_nodes = bfs_shortest_path(start_input, end_input)

if path_nodes:
    # Convert list of permutations in the path to X, Y, Z lists
    # This ensures we visit every vertex along the way
    path_coords = [coord_map[p] for p in path_nodes]
    px = [c[0] for c in path_coords]
    py = [c[1] for c in path_coords]
    pz = [c[2] for c in path_coords]

    # Plot the Path (Red Line + Small dots at stops)
    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz,
        mode='lines+markers', # Draw both lines and points
        line=dict(color='#ff0055', width=10), # Neon Red
        marker=dict(size=4, color='#ff0055'), # Small markers at vertices
        name='Shortest Path'
    ))

    # Highlight Start (Green Diamond)
    start_c = coord_map[start_input]
    fig.add_trace(go.Scatter3d(
        x=[start_c[0]], y=[start_c[1]], z=[start_c[2]],
        mode='markers+text',
        marker=dict(color='#00ff00', size=15, symbol='diamond'),
        text=[str(start_input)],
        textposition="top center",
        textfont=dict(color='white', size=12),
        name='Start'
    ))

    # Highlight End (Cyan Diamond)
    end_c = coord_map[end_input]
    fig.add_trace(go.Scatter3d(
        x=[end_c[0]], y=[end_c[1]], z=[end_c[2]],
        mode='markers+text',
        marker=dict(color='#00ffff', size=15, symbol='diamond'),
        text=[str(end_input)],
        textposition="top center",
        textfont=dict(color='white', size=12),
        name='End'
    ))

    st.success(f"Path found! Distance: {len(path_nodes) - 1} swaps.")

# --- 6. STYLING ---
fig.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    scene=dict(
        xaxis=dict(visible=False, backgroundcolor="black"),
        yaxis=dict(visible=False, backgroundcolor="black"),
        zaxis=dict(visible=False, backgroundcolor="black"),
        bgcolor='black'
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    showlegend=True,
    legend=dict(font=dict(color="white"), y=0.9),
    height=700
)

st.plotly_chart(fig, use_container_width=True)
