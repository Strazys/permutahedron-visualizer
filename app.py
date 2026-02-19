import streamlit as st
import plotly.graph_objects as go
import numpy as np
import itertools
from collections import deque

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Permutahedron Explorer", layout="wide")
st.title("Permutahedron Visualizer by Windsor Kiang")
st.markdown("""
This visualizer finds the **shortest geometric path** along the edges of the permutahedron.
Edges connect permutations that differ by swapping **values** $k$ and $k+1$ (e.g. swapping 1 and 2).
""")
# --- 2. CORE FUNCTIONS ---

def get_projection_matrix(n):
    """
    Generates a projection matrix to map n-dimensional points to 3D.
    We map the standard basis vectors of R^n to vectors pointing to the 
    vertices of a regular simplex centered at the origin in 3D (or 2D for n=3).
    """
    if n == 3:
        # Map to 2D (Hexagon)
        # x, y coords on a circle
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.array([np.cos(angles), np.sin(angles), np.zeros(n)])
    
    if n == 4:
        # Standard S4 projection (Truncated Octahedron)
        return np.array([
            [np.sqrt(2)/2, -np.sqrt(2)/2, 0, 0],
            [np.sqrt(6)/6, np.sqrt(6)/6, -np.sqrt(2/3), 0],
            [np.sqrt(12)/12, np.sqrt(12)/12, np.sqrt(12)/12, -np.sqrt(3)/2]
        ])
    
    # For n >= 5, we use a general approach:
    # Generate n vectors in 3D that are maximally separated (Fibonacci sphere or Simplex)
    # Simple heuristic: points on a sphere
    np.random.seed(42) # Fixed seed for consistency
    # Create random matrix then orthogonalize rows
    M = np.random.randn(3, n)
    q, r = np.linalg.qr(M.T)
    return q.T

def get_geometric_neighbors(perm, order):
    """
    Returns neighbors reachable by swapping values v and v+1.
    This creates the edges of the permutahedron.
    """
    p_list = list(perm)
    neighbors = []
    
    # Swap value v and v+1 for v in 1..order-1
    for v in range(1, order):
        idx1 = p_list.index(v)
        idx2 = p_list.index(v + 1)
        
        p_new = p_list[:]
        p_new[idx1], p_new[idx2] = p_new[idx2], p_new[idx1]
        neighbors.append(tuple(p_new))
        
    return neighbors

def bfs_shortest_path(start, end, order):
    """Calculates shortest path using BFS."""
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        
        for neighbor in get_geometric_neighbors(current, order):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

def create_permutahedron_graph(order):
    """
    Main function to generate the plot for a specific order.
    """
    items = range(1, order+1)
    perms_list = [*itertools.permutations(items)]
    permuted_items = np.array(perms_list)

    # 1. Project to 3D
    A = get_projection_matrix(order)
    projected_items = np.einsum('ik,ak->ai', A, permuted_items)
    coord_map = {perm: coord for perm, coord in zip(perms_list, projected_items)}

    # 2. UI Inputs for this specific tab
    col1, col2 = st.columns(2)
    with col1:
        start_node = st.selectbox(f"Start (S{order})", options=perms_list, index=0, key=f"start_{order}")
    with col2:
        end_node = st.selectbox(f"End (S{order})", options=perms_list, index=len(perms_list)-1, key=f"end_{order}")

    # 3. Build Mesh
    edge_x, edge_y, edge_z = [], [], []
    seen_edges = set()

    # Progress bar for higher orders (S5 takes a second)
    with st.spinner(f"Calculating geometry for S{order}..."):
        for perm in perms_list:
            u = coord_map[perm]
            for neighbor in get_geometric_neighbors(perm, order):
                edge_id = tuple(sorted((perm, neighbor)))
                if edge_id not in seen_edges:
                    seen_edges.add(edge_id)
                    v = coord_map[neighbor]
                    edge_x.extend([u[0], v[0], None])
                    edge_y.extend([u[1], v[1], None])
                    edge_z.extend([u[2], v[2], None])

    # 4. Plot
    fig = go.Figure()
    
    # Edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='white', width=2),
        hoverinfo='none',
        name='Edges'
    ))

    # 5. Path
    path_nodes = bfs_shortest_path(start_node, end_node, order)
    if path_nodes:
        px = [coord_map[p][0] for p in path_nodes]
        py = [coord_map[p][1] for p in path_nodes]
        pz = [coord_map[p][2] for p in path_nodes]

        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode='lines+markers',
            line=dict(color='#ff0055', width=8),
            marker=dict(size=4, color='#ff0055'),
            name='Shortest Path'
        ))

        # Start/End Markers
        start_c = coord_map[start_node]
        end_c = coord_map[end_node]
        
        fig.add_trace(go.Scatter3d(
            x=[start_c[0]], y=[start_c[1]], z=[start_c[2]],
            mode='markers+text',
            marker=dict(color='#00ff00', size=10, symbol='diamond'),
            text=[str(start_node)], textposition="top center",
            textfont=dict(color='white'), name='Start'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[end_c[0]], y=[end_c[1]], z=[end_c[2]],
            mode='markers+text',
            marker=dict(color='#00ffff', size=10, symbol='diamond'),
            text=[str(end_node)], textposition="top center",
            textfont=dict(color='white'), name='End'
        ))
        
        st.success(f"Path Distance: {len(path_nodes) - 1} steps")

    # Layout
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
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


# --- 3. MAIN TABS ---
tab3, tab4, tab5 = st.tabs(["S3 (Hexagon)", "S4 (Truncated Octahedron)", "S5 (4D Object)"])

with tab3:
    st.header("Symmetric Group S3")
    create_permutahedron_graph(3)

with tab4:
    st.header("Symmetric Group S4")
    create_permutahedron_graph(4)

with tab5:
    st.header("Symmetric Group S5")
    st.info("Note: This is a 4D object projected into 3D. Some edges may appear to overlap.")
    create_permutahedron_graph(5)
    
