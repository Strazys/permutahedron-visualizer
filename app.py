import streamlit as st
import plotly.graph_objects as go
import numpy as np
import itertools
from collections import deque
import ast

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Permutahedron Explorer", layout="wide")

st.title("Permutahedron Shortest Path Visualizer")
st.markdown("Enter two permutations of `(1, 2, 3, 4)` to see the shortest path between them on the permutahedron.")

# --- 2. MATH & LOGIC ---
order = 4
items = range(1, order+1)
original_perms_list = [*itertools.permutations(items)]
permuted_items = np.array(original_perms_list)

# Project to 3D
A = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0, 0],
              [np.sqrt(6)/6, np.sqrt(6)/6, -np.sqrt(2/3), 0],
              [np.sqrt(12)/12, np.sqrt(12)/12, np.sqrt(12)/12, -np.sqrt(3)/2]])

projected_items = np.einsum('ik,ak->ai', A, permuted_items)
coord_map = {perm: coord for perm, coord in zip(original_perms_list, projected_items)}

def get_neighbors(perm):
    perm = list(perm)
    for i in range(len(perm) - 1):
        p_new = perm[:]
        p_new[i], p_new[i+1] = p_new[i+1], p_new[i]
        yield tuple(p_new)

def bfs_shortest_path(start, end):
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

# --- 3. UI INPUTS ---
col1, col2 = st.columns(2)
with col1:
    start_input = st.selectbox("Start Permutation", options=original_perms_list, index=0)
with col2:
    # Default to the reverse of the start for maximum distance
    end_input = st.selectbox("End Permutation", options=original_perms_list, index=len(original_perms_list)-1)

# --- 4. DRAWING THE MESH (BLACK EDGES) ---
# We use Graph Objects directly for better control over the black edges
edge_x, edge_y, edge_z = [], [], []

for i, point in enumerate(projected_items):
    d = np.linalg.norm(projected_items-point[np.newaxis,:], axis=1)
    # Find neighbors (distance approx sqrt(2) for this projection, checking < 1e-3 tolerance on the sorted diffs)
    js = (abs(d - d[d.argsort()[1]]) < 1e-3).nonzero()[0]
    
    for j in js:
        # Add line segment (point -> neighbor -> None) to create disjoint lines
        edge_x.extend([point[0], projected_items[j][0], None])
        edge_y.extend([point[1], projected_items[j][1], None])
        edge_z.extend([point[2], projected_items[j][2], None])

fig = go.Figure()

# Add the wireframe mesh (BLACK LINES)
fig.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='black', width=2), # <--- HERE IS THE BLACK COLOR
    hoverinfo='none',
    name='Edges'
))

# --- 5. DRAWING THE PATH ---
path_nodes = bfs_shortest_path(start_input, end_input)

if path_nodes:
    path_coords = [coord_map[p] for p in path_nodes]
    px_coords = [p[0] for p in path_coords]
    py_coords = [p[1] for p in path_coords]
    pz_coords = [p[2] for p in path_coords]

    # Path Line (Red)
    fig.add_trace(go.Scatter3d(
        x=px_coords, y=py_coords, z=pz_coords,
        mode='lines+markers',
        line=dict(color='red', width=8),
        marker=dict(size=4, color='red'),
        name='Shortest Path'
    ))

    # Start Node (Green)
    start_c = coord_map[start_input]
    fig.add_trace(go.Scatter3d(
        x=[start_c[0]], y=[start_c[1]], z=[start_c[2]],
        mode='markers+text',
        marker=dict(color='green', size=10, symbol='diamond'),
        text=[str(start_input)], textposition="top center",
        name='Start'
    ))

    # End Node (Blue)
    end_c = coord_map[end_input]
    fig.add_trace(go.Scatter3d(
        x=[end_c[0]], y=[end_c[1]], z=[end_c[2]],
        mode='markers+text',
        marker=dict(color='blue', size=10, symbol='diamond'),
        text=[str(end_input)], textposition="top center",
        name='End'
    ))

    st.success(f"Path found! Steps: {len(path_nodes) - 1}")

# Clean up layout
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    showlegend=True,
    height=700
)

st.plotly_chart(fig, use_container_width=True)