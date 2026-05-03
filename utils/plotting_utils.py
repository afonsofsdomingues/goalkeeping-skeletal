import plotly.express as px
import pandas as pd
import json

import colorsys

def generate_colors(n):
    return [
        f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
        for r, g, b in
        [colorsys.hsv_to_rgb(i / n, 0.75, 0.9) for i in range(n)]
    ]


def plot_interactive_umap_3d(emb, labels, title="Interactive UMAP", split='val', indices=None):
    """
    Creates an interactive 3D UMAP plot. 
    Attempts to read metadata to mark specific samples (e.g., Idle actions) with a different symbol.
    
    Args:
        emb: Embedding array (N, 3)
        labels: Cluster labels (N,)
        title: Plot title
        split: Which dataset split to use ('train' or 'val')
        indices: Original indices for metadata lookup (optional)
    """
    df_plot = pd.DataFrame(emb, columns=['UMAP_1', 'UMAP_2', 'UMAP_3'])
    df_plot['Cluster'] = labels.astype(str)
    
    # Use provided indices or default to sequential
    if indices is not None:
        df_plot['Index'] = indices
    else:
        df_plot['Index'] = df_plot.index
    
    symbol_col = None
    hover_data = ['Index', 'Cluster']
    
    possible_paths = [
        f'../data/tensor_native/{split}_metadata.json',
        f'../data/tensor/{split}_metadata.json',
    ]
    
    meta_loaded = False
    for path in possible_paths:
        try:
            with open(path, 'r') as f:
                meta_data = json.load(f)
        except Exception:
            continue
        
        if isinstance(meta_data, list) and len(meta_data) > 0 and isinstance(meta_data[0], dict) and 'clip_type' in meta_data[0]:
            
            # Use indices to get correct metadata entries
            if indices is not None:
                # Map indices to metadata
                relevant_meta = [meta_data[i] if i < len(meta_data) else {} for i in df_plot['Index']]
            else:
                current_len = len(df_plot)
                if len(meta_data) < current_len:
                    continue
                # Slice metadata to match embedding length
                relevant_meta = meta_data[:current_len]
            
            if relevant_meta:
                # Extract status directly
                status_list = [item.get('clip_type', 'unknown') for item in relevant_meta]
                df_plot['Status'] = status_list
                df_plot['Status'] = df_plot['Status'].apply(lambda x: 'Idle' if str(x).lower() == 'idle' else 'Active')

                if 'original_duration' in relevant_meta[0]:
                    df_plot['Duration'] = [item.get('original_duration', 0) for item in relevant_meta]
                    hover_data.append('Duration')

                symbol_col = 'Status'
                hover_data.append('Status')
                meta_loaded = True
                break
            
    if not meta_loaded:
        print("Metadata file not found or didn't match data length")

    unique_clusters = sorted(df_plot['Cluster'].unique(), key=lambda x: int(x) if x != '-1' else -99)

    colors = generate_colors(len(unique_clusters))
    color_map = {}

    for lbl, c in zip(unique_clusters, colors):
        color_map[lbl] = 'black' if lbl == '-1' else c


    fig = px.scatter_3d(
        df_plot, 
        x='UMAP_1', 
        y='UMAP_2',
        z='UMAP_3', 
        color='Cluster',
        symbol=symbol_col,
        hover_data=hover_data,
        title=title,
        width=1000,
        height=800,
        color_discrete_map=color_map,
        opacity=0.7
    )
    
    fig.update_traces(marker=dict(size=5))
    fig.show()