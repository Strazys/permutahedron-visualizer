import streamlit as st
import plotly.graph_objects as go
import numpy as np
import itertools
from collections import deque

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Permutahedron Explorer", layout="wide")
st.title("Permutahedron Shortest Path")
st.markdown("""
This visualizer finds the **shortest geometric path** along the edges of the permutahedron.
Edges connect permutations that differ by swapping **values** $k$ and $k+1$ (e.g. swapping 1 and 2).
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
coord_map = {perm: coord for perm, coord in zip(original_perms_list, projected_items)}

def get_geometric_neighbors(perm):
    """
    Returns neighbors that are geometrically closest (distance sqrt(2)).
    Rule: Swap two elements if their VALUES differ by exactly 1.
    """
    p_list = list(perm)
    neighbors = []
    
    # We want to swap value k and k+1 for k in 1..n-1
    # For order=4, we swap (1,2), (2,3), (3,4)
    for v in range(1, order):
        val1 = v
        val2 = v + 1
        
        # Find where these values are in the current permutation
        idx1 = p_list.index(val1)
        idx2 = p_list.index(val2)
        
        # Create new permutation by swapping them
        p_new = p_list[:]
        p_new[idx1], p_new[idx2] = p_new[idx2], p_new[idx1]
        neighbors.append(tuple(p_new))
        
    return neighbors

def bfs_shortest_path(start, end):
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        
        for neighbor in get_geometric_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# --- 3. UI INPUTS ---
col1, col2 = st.columns(2)
with col1:
    start_input = st.selectbox("Start Node", options=original_perms_list, index=0)
with col2:
    # Default to the reverse permutation (maximum distance)
    end_input = st.selectbox("End Node", options=original_perms_list, index=len(original_perms_list)-1)

# --- 4. BUILD THE BACKGROUND MESH ---
edge_x, edge_y, edge_z = [], [], []
seen_edges = set()

for perm in original_perms_list:
    u = coord_map[perm]
    for neighbor in get_geometric_neighbors(perm):
        # Create a sorted tuple ID to avoid drawing the same edge twice
        edge_id = tuple(sorted((perm, neighbor)))
        
        if edge_id not in seen_edges:
            seen_edges.add(edge_id)
            v = coord_map[neighbor]
            edge_x.extend([u[0], v[0], None])
            edge_y.extend([u[1], v[1], None])
            edge_z.extend([u[2], v[2], None])

fig = go.Figure()

# Plot the Wireframe (White)
fig.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='white', width=3),
    hoverinfo='none',
    name='Edges'
))

# --- 5. CALCULATE AND PLOT THE PATH ---
path_nodes = bfs_shortest_path(start_input, end_input)

if path_nodes:
    path_coords = [coord_map[p] for p in path_nodes]
    px = [c[0] for c in path_coords]
    py = [c[1] for c in path_coords]
    pz = [c[2] for c in path_coords]

    # Plot the Path (Red Line + Small dots at stops)
    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz,
        mode='lines+markers',
        line=dict(color='#ff0055', width=10),
        marker=dict(size=5, color='#ff0055'),
        name='Shortest Path'
    ))

    # Highlight Start
    start_c = coord_map[start_input]
    fig.add_trace(go.Scatter3d(
        x=[start_c[0]], y=[start_c[1]], z=[start_c[2]],
        mode='markers+text',
        marker=dict(color='#00ff00', size=15, symbol='diamond'),
        text=[str(start_input)],
        textposition="top center",
        textfont=dict(color='white', size=14),
        name='Start'
    ))

    # Highlight End
    end_c = coord_map[end_input]
    fig.add_trace(go.Scatter3d(
        x=[end_c[0]], y=[end_c[1]], z=[end_c[2]],
        mode='markers+text',
        marker=dict(color='#00ffff', size=15, symbol='diamond'),
        text=[str(end_input)],
        textposition="top center",
        textfont=dict(color='white', size=14),
        name='End'
    ))

    st.success(f"Path found! Distance: {len(path_nodes) - 1} steps.")

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
