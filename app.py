import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Page config
st.set_page_config(
    page_title="Color Sampling Explorer",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Combining Colors: Interactive Sampling Explorer")
st.markdown("""
Explore how different sampling methods affect outcomes when combining colored samples.
Adjust the population composition and see how deterministic vs. random sampling creates different aggregate colors.
""")

# How to Use documentation
with st.expander("📖 How to Use This App", expanded=False):
    st.markdown("""
    **Getting Started:**
    1. Choose a color palette from the sidebar (RGB, CMY, RYB, or Secondary colors)
    2. Adjust the number of samples for each color using the sliders
    3. Set your desired sample size percentage (what % of the population to select)
    4. Select which sampling methods to compare (1-3 methods)
    
    **Understanding the Visualizations:**
    - **Left panel (Sample Selection)**: Shows which samples were selected
      - Grey dots = not selected
      - Colored dots = selected
      - Darker colored dots = selected multiple times (only in "with replacement" method)
    
    - **Right panel (Aggregated Color)**: Shows the resulting mixed color
      - Large circle displays the weighted RGB average of all selected samples
      - RGB values shown as (R, G, B) on the circle
    
    **Sampling Methods Explained:**
    - **Deterministic**: Always selects the first N% of each color group
      - Pros: Completely reproducible (same input = same output)
      - Cons: May introduce systematic bias, ignores natural variation
    
    - **Random (no replacement)**: Randomly selects samples, each sample ≤1 time
      - Pros: Unbiased, represents population well on average
      - Cons: Results vary between runs (use random seed for reproducibility)
    
    - **Random (with replacement)**: Randomly selects samples, samples can repeat
      - Pros: Foundation of bootstrap methods, maintains sample size exactly
      - Cons: May over-represent or under-represent some samples
    
    **Tips for Exploration:**
    - Try imbalanced populations (e.g., 300 first color, 100 second, 50 third)
    - Compare how each method handles the imbalance differently
    - Lower percentage sample sizes will show greater variability in aggregated colors
    - Change the random seed to see how random methods vary while deterministic stays constant
    - Switch color palettes to see how different color combinations mix
    """)

st.markdown("---")


# ========== DATA CREATION FUNCTIONS ==========

def make_population_df(
    counts: dict,
    x_centers: dict = None,
    x_jitter: float = 0.04,  # Reduced from 0.08 for tighter clustering
    y_jitter: float = 0.6,
    seed: int = 42
) -> pd.DataFrame:
    """Create a population DataFrame with spatial coordinates."""
    rng = np.random.default_rng(seed)

    if x_centers is None:
        # Default centers for 3 colors
        colors = list(counts.keys())
        n_colors = len(colors)
        spacing = 1.0 / (n_colors - 1) if n_colors > 1 else 0
        x_centers = {color: -0.5 + i * spacing for i, color in enumerate(colors)}

    rows = []
    for color, n in counts.items():
        for i in range(n):
            rows.append({
                "color": color,
                "color_index": i,
                "x": x_centers.get(color, 0) + rng.normal(0, x_jitter),
                "y": rng.normal(0, y_jitter)
            })

    df = pd.DataFrame(rows)
    df["global_index"] = range(len(df))
    return df


def compute_draw_count(
    df_population: pd.DataFrame,
    frac: float,
    is_subset: bool,
    replace: bool,
    random_state: int | None = None
) -> pd.Series:
    """Compute per-sample draw counts for a given sampling configuration."""
    n_total = int(len(df_population) * frac)
    draw_count = pd.Series(0, index=df_population.index)

    if is_subset:
        # Deterministic: per-color proportional truncation
        cutoffs = (
            df_population.groupby("color")["color_index"]
            .max()
            .add(1)
            .mul(frac)
            .apply(np.ceil)
            .astype(int)
        )
        selected_idx = df_population.index[
            df_population["color_index"] < df_population["color"].map(cutoffs)
        ]
        draw_count.loc[selected_idx] = 1
    else:
        # Random sampling (with or without replacement)
        sampled_idx = df_population.sample(
            n=n_total,
            replace=replace,
            random_state=random_state
        ).index
        counts = sampled_idx.value_counts()
        draw_count.loc[counts.index] = counts.values

    return draw_count


def aggregate_color_rgb(df, draw_col, color_map):
    """Compute weighted average RGB color for a given draw_count column."""
    weights = df[draw_col].values

    if weights.sum() == 0:
        return "rgb(230,230,230)", (230, 230, 230)

    rgb = np.zeros(3)
    for color, rgb_str in color_map.items():
        if color == "Grey":
            continue
        r, g, b = map(int, rgb_str[4:-1].split(","))
        mask = df["color"] == color
        rgb += np.array([r, g, b]) * weights[mask].sum()

    rgb = (rgb / weights.sum()).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", tuple(rgb)


# ========== COLOR SCHEMES ==========

COLOR_SCHEMES = {
    "RGB (Red, Green, Blue)": {
        "colors": ["Red", "Green", "Blue"],
        "color_map": {
            "Grey": "rgb(165, 165, 165)",
            "Red": "rgb(255, 0, 0)",
            "Green": "rgb(0, 180, 0)",
            "Blue": "rgb(0, 0, 255)"
        }
    },
    "CMY (Cyan, Magenta, Yellow)": {
        "colors": ["Cyan", "Magenta", "Yellow"],
        "color_map": {
            "Grey": "rgb(165, 165, 165)",
            "Cyan": "rgb(0, 255, 255)",
            "Magenta": "rgb(255, 0, 255)",
            "Yellow": "rgb(255, 255, 0)"
        }
    },
    "RYB (Red, Yellow, Blue)": {
        "colors": ["Red", "Yellow", "Blue"],
        "color_map": {
            "Grey": "rgb(165, 165, 165)",
            "Red": "rgb(255, 0, 0)",
            "Yellow": "rgb(255, 255, 0)",
            "Blue": "rgb(0, 0, 255)"
        }
    },
    "Secondary (Orange, Green, Purple)": {
        "colors": ["Orange", "Green", "Purple"],
        "color_map": {
            "Grey": "rgb(165, 165, 165)",
            "Orange": "rgb(255, 165, 0)",
            "Green": "rgb(0, 180, 0)",
            "Purple": "rgb(160, 32, 240)"
        }
    }
}


# ========== SIDEBAR CONTROLS ==========

st.sidebar.header("Color Scheme")

color_scheme = st.sidebar.selectbox(
    "Choose color palette:",
    options=list(COLOR_SCHEMES.keys()),
    index=0
)

scheme_config = COLOR_SCHEMES[color_scheme]
color_names = scheme_config["colors"]
color_map = scheme_config["color_map"]

st.sidebar.markdown("---")
st.sidebar.header("Population Settings")

# Dynamic sliders based on color scheme
counts = {}
for color in color_names:
    counts[color] = st.sidebar.slider(
        f"{color} Samples",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )

st.sidebar.markdown("---")
st.sidebar.header("Sampling Settings")

sample_percent = st.sidebar.slider("Sample Size (%)", min_value=10, max_value=100, value=50, step=10)
sample_frac = sample_percent / 100

st.sidebar.markdown("---")

# Method selection
st.sidebar.subheader("Sampling Methods")
methods = st.sidebar.multiselect(
    "Select 1-3 methods to compare:",
    options=[
        "Deterministic",
        "Random (no replacement)",
        "Random (with replacement)"
    ],
    default=["Deterministic", "Random (no replacement)", "Random (with replacement)"]
)

st.sidebar.markdown("---")
random_state = st.sidebar.number_input("Random Seed", min_value=0, max_value=1000, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Method descriptions:**
- **Deterministic**: Takes first N% of each color
- **Random (no replacement)**: Each sample selected ≤1 time
- **Random (replacement)**: Samples can be selected multiple times
""")


# ========== MAIN PROCESSING ==========

if not methods:
    st.warning("Please select at least one sampling method from the sidebar.")
    st.stop()

# Map method names to config
method_configs = {
    "Deterministic": {"is_subset": True, "replace": False},
    "Random (no replacement)": {"is_subset": False, "replace": False},
    "Random (with replacement)": {"is_subset": False, "replace": True},
}

# Generate data
df = make_population_df(counts, seed=random_state)

# Compute draw counts for selected methods
method_columns = {}
for method_name in methods:
    cfg = method_configs[method_name]
    colname = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    df[colname] = compute_draw_count(
        df_population=df,
        frac=sample_frac,
        is_subset=cfg["is_subset"],
        replace=cfg["replace"],
        random_state=random_state
    )
    method_columns[method_name] = colname


# ========== VISUALIZATIONS ==========

# Display population summary
total_samples = sum(counts.values())
st.markdown(f"""
**Population:** {' + '.join([f'{count} {color}' for color, count in counts.items()])} = {total_samples} total samples  
**Sample size:** {sample_percent}% ({int(total_samples * sample_frac)} samples)
""")

st.markdown("---")

# Create visualizations for each method (one row per method)
for method_name in methods:
    draw_col = method_columns[method_name]
    
    st.subheader(f"**{method_name}**")
    
    # Create figure with 2 columns: samples (left) and aggregate (right)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.08,
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # LEFT: Sample Selection
    draw = df[draw_col]
    plot_color = np.where(draw > 0, df["color"], "Grey")
    opacity = np.where(
        draw > 0,
        0.4 + 0.6 * np.minimum(draw, 4) / 4,
        0.4,
    )
    
    for color_name in ["Grey"] + color_names:
        mask = plot_color == color_name
        if not mask.any():
            continue
        
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, "x"],
                y=df.loc[mask, "y"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=color_map[color_name],
                    opacity=opacity[mask],
                ),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1,
        )
    
    # RIGHT: Aggregate Color
    agg_color_str, agg_rgb = aggregate_color_rgb(df, draw_col, color_map)
    
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0],
            mode="markers",
            marker=dict(
                size=200,
                color=agg_color_str,
                line=dict(color="rgba(0,0,0,0.45)", width=2),
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=2,
    )
    
    # Add RGB text annotation on aggregate circle
    fig.add_annotation(
        x=0, y=0,
        text=f"<b>({agg_rgb[0]}, {agg_rgb[1]}, {agg_rgb[2]})</b>",
        showarrow=False,
        font=dict(size=13, color="white" if sum(agg_rgb) < 400 else "black"),
        xref="x2", yref="y2"
    )
    
    # Format axes - expanded range to prevent cropping
    fig.update_xaxes(visible=False, range=[-1.1, 1.1], row=1, col=1)
    fig.update_yaxes(visible=False, range=[-1.1, 1.1], row=1, col=1)
    fig.update_xaxes(visible=False, range=[-1.1, 1.1], row=1, col=2)
    fig.update_yaxes(visible=False, range=[-1.1, 1.1], row=1, col=2)
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),  # Reduced margins
        plot_bgcolor="white",
    )
    
    # Display with labels
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.markdown("**Sample Selection**")
        # Only show "darker dots" caption for replacement method
        if "with replacement" in method_name.lower():
            st.caption("Darker dots = selected more times")
        else:
            st.caption("Grey dots not selected, colored dots selected")
    
    with col2:
        st.markdown("**Aggregated Color**")
        st.caption(f"RGB: ({agg_rgb[0]}, {agg_rgb[1]}, {agg_rgb[2]})")
    
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{method_name.replace(' ', '_')}")
    st.markdown("---")


# Key Insights
with st.expander("💡 Key Statistical Insights"):
    st.markdown("""
    **Central Limit Theorem in Action:**
    - As sample size increases, the aggregate color converges toward the population mean
    - Random sampling captures this convergence with variation
    - Deterministic sampling shows convergence without variation
    
    **Sampling Bias:**
    - Deterministic selection can introduce bias if the first N% differs from the full population
    - In our case, samples are ordered by color, so deterministic is unbiased
    - In real datasets, ordering matters! (e.g., sorted by time, location, ID)
    
    **Bootstrap Methods:**
    - "With replacement" sampling is the foundation of bootstrap resampling
    - Allows us to estimate sampling distributions without knowing the true population
    - The variation you see across random seeds shows sampling uncertainty
    
    **Practical Applications:**
    - **Survey sampling**: How you select respondents affects results
    - **A/B testing**: Random assignment vs. convenience sampling
    - **Machine learning**: Train/test splits and cross-validation
    - **Quality control**: How sampling strategy affects defect detection
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built by <a href='https://pixelprocess.org' target='_blank'>PixelProcess</a> | 
    Part of <a href='https://dexterousdata.com' target='_blank'>Dexterous Data</a><br>
    <a href='https://github.com/pixel-process-dev/combining-colors' target='_blank'>View Source on GitHub</a> | 
    <a href='https://pixelprocess.org/build-models/combining-colors.html' target='_blank'>Full Tutorial</a></p>
</div>
""", unsafe_allow_html=True)
