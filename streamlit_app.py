import streamlit as st

# --- PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
from scipy.stats import linregress
import textwrap

# --- PDF & Map Specific Imports ---
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from matplotlib.ticker import MaxNLocator

# --- CONFIGURATION ---
LOCATION_DATA_PATH = 'HDI_lake_district.csv'
HEALTH_DATA_PATH = "lake_health_data.csv"

# --- PARAMETER DICTIONARY ---
PARAMETER_PROPERTIES = {
    'Air Temperature': {'impact': 'negative', 'type': 'climate'},
    'Evaporation': {'impact': 'positive', 'type': 'climate'},
    'Precipitation': {'impact': 'positive', 'type': 'climate'},
    'Lake Water Surface Temperature': {'impact': 'negative', 'type': 'water_quality'},
    'Water Clarity': {'impact': 'negative', 'type': 'water_quality'},
    'Barren Area': {'impact': 'negative', 'type': 'land_cover'},
    'Urban Area': {'impact': 'negative', 'type': 'land_cover'},
    'Vegetation Area': {'impact': 'positive', 'type': 'land_cover'},
    'HDI': {'impact': 'positive', 'type': 'socioeconomic'},
    'Area': {'impact': 'positive', 'type': 'physical'}
}
LAND_COVER_INTERNAL_COLS = ['Barren Area', 'Urban Area', 'Vegetation Area']


# --- CORE DATA & ANALYSIS FUNCTIONS ---
@st.cache_data
def prepare_all_data(health_path, location_path):
    try:
        df_health = pd.read_csv(health_path)
        df_location = pd.read_csv(location_path)
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}.")
        return None, None, None
    health_col_map = {'Air_Temperature': 'Air Temperature', 'Evaporation': 'Evaporation', 'Precipitation': 'Precipitation','Barren': 'Barren Area', 'Urban and Built-up': 'Urban Area', 'Vegetation': 'Vegetation Area','Lake_Water_Surface_Temperature': 'Lake Water Surface Temperature', 'Water_Clarity': 'Water Clarity', 'Water_Clarity(FUI)': 'Water Clarity','Area': 'Area'}
    df_health = df_health.rename(columns=health_col_map)
    potential_id_cols = ['Lake_ID', 'Lake_id', 'lake_id']
    health_id_col = next((col for col in potential_id_cols if col in df_health.columns), None)
    loc_id_col = next((col for col in potential_id_cols if col in df_location.columns), None)
    if not health_id_col or not loc_id_col:
        st.error(f"Critical Error: Could not find a lake identifier column in one or both CSV files.")
        return None, None, None
    df_health = df_health.rename(columns={health_id_col: 'Lake_ID'})
    df_location = df_location.rename(columns={loc_id_col: 'Lake_ID'})
    df_health['Lake_ID'] = pd.to_numeric(df_health['Lake_ID'], errors='coerce').dropna().astype(int)
    df_location['Lake_ID'] = pd.to_numeric(df_location['Lake_ID'], errors='coerce').dropna().astype(int)
    location_subset = df_location[['Lake_ID', 'HDI']].copy()
    df_merged = pd.merge(df_health, location_subset, on='Lake_ID', how='left')
    available_data_cols = [col for col in df_merged.columns if col in PARAMETER_PROPERTIES]
    ui_options = [p for p in available_data_cols if p not in LAND_COVER_INTERNAL_COLS]
    if any(p in available_data_cols for p in LAND_COVER_INTERNAL_COLS): ui_options.append("Land Cover")
    if not ui_options:
        st.error("No valid analysis parameters found in the data files.")
        return None, None, None
    return df_merged, df_location, sorted(ui_options)

def get_effective_weights(selected_ui_options, all_df_columns):
    effective_weights = {}
    num_main_groups = len(selected_ui_options)
    w_main = 1.0 / num_main_groups if num_main_groups > 0 else 0.0
    if "Land Cover" in selected_ui_options:
        available_lc_cols_in_data = [p for p in LAND_COVER_INTERNAL_COLS if p in all_df_columns]
        num_land_cover_items = len(available_lc_cols_in_data)
        w_sub_landcover = 1.0 / num_land_cover_items if num_land_cover_items > 0 else 0.0
        for lc_param in available_lc_cols_in_data: effective_weights[lc_param] = w_main * w_sub_landcover
    for param in selected_ui_options:
        if param != "Land Cover": effective_weights[param] = w_main
    return effective_weights

@st.cache_data
def calculate_lake_health_score(_df, selected_ui_options, lake_ids_tuple):
    df = _df.copy()
    if df.empty or not selected_ui_options: return pd.DataFrame(), {}
    def norm(x): return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    def rev_norm(x): return 1.0 - norm(x)
    final_weights = get_effective_weights(selected_ui_options, df.columns)
    params_to_process = list(final_weights.keys())
    df_imputed = df.copy()
    for param in params_to_process:
        if param in df_imputed.columns:
            df_imputed = df_imputed.sort_values(by=['Lake_ID', 'Year'])
            df_imputed[param] = df_imputed.groupby('Lake_ID')[param].transform(lambda x: x.bfill().ffill())
            df_imputed[param] = df_imputed[param].fillna(df_imputed[param].median())
    latest_year_data = df_imputed.loc[df_imputed.groupby('Lake_ID')['Year'].idxmax()].copy().set_index('Lake_ID')
    total_score = pd.Series(0.0, index=latest_year_data.index)
    calculation_details = {lake_id: {} for lake_id in latest_year_data.index}
    for param in params_to_process:
        props = PARAMETER_PROPERTIES[param]
        latest_values = latest_year_data[param]
        
        raw_pv_score = norm(latest_values) if props['impact'] == 'positive' else rev_norm(latest_values)
        present_value_score_result = raw_pv_score.fillna(0.5) if isinstance(raw_pv_score, pd.Series) else raw_pv_score
        
        if param == 'HDI':
            factor_score_result = present_value_score_result
            for lake_id in latest_year_data.index:
                pv_score = present_value_score_result.loc[lake_id] if isinstance(present_value_score_result, pd.Series) else present_value_score_result
                factor_score = factor_score_result.loc[lake_id] if isinstance(factor_score_result, pd.Series) else factor_score_result
                calculation_details[lake_id][param] = {'Raw Value': latest_values.loc[lake_id], 'Norm Pres.': pv_score, 'Norm Trend': 'N/A', 'Norm P-Val': 'N/A', 'Factor Score': factor_score, 'Weight': final_weights[param], 'Contribution': factor_score * final_weights[param]}
        else:
            trends = df_imputed.groupby('Lake_ID').apply(lambda x: linregress(x['Year'], x[param]) if len(x['Year'].unique()) > 1 else (0,0,0,1,0))
            slopes = trends.apply(lambda x: x.slope if not isinstance(x, tuple) else x[0])
            p_values = trends.apply(lambda x: x.pvalue if not isinstance(x, tuple) else x[3])
            
            raw_slope_score = norm(slopes) if props['impact'] == 'positive' else rev_norm(slopes)
            slope_norm_result = raw_slope_score.fillna(0.5) if isinstance(raw_slope_score, pd.Series) else raw_slope_score

            raw_p_value_score = norm(p_values)
            if isinstance(raw_p_value_score, pd.Series):
                p_value_norm_result = (1.0 - raw_p_value_score).fillna(0.5)
            else:
                p_value_norm_result = 1.0 - raw_p_value_score
            
            factor_score_result = (present_value_score_result + slope_norm_result + p_value_norm_result) / 3.0
            
            for lake_id in latest_year_data.index:
                pv_score = present_value_score_result.loc[lake_id] if isinstance(present_value_score_result, pd.Series) else present_value_score_result
                s_norm = slope_norm_result.loc[lake_id] if isinstance(slope_norm_result, pd.Series) else slope_norm_result
                p_norm = p_value_norm_result.loc[lake_id] if isinstance(p_value_norm_result, pd.Series) else p_value_norm_result
                factor_score = factor_score_result.loc[lake_id] if isinstance(factor_score_result, pd.Series) else factor_score_result
                calculation_details[lake_id][param] = {'Raw Value': latest_values.loc[lake_id], 'Norm Pres.': pv_score, 'Norm Trend': s_norm, 'Norm P-Val': p_norm, 'Factor Score': factor_score, 'Weight': final_weights[param], 'Contribution': factor_score * final_weights[param]}
        
        total_score += final_weights[param] * factor_score_result
        
    latest_year_data['Health Score'] = total_score.fillna(0.0)
    
    latest_year_data['Rank'] = latest_year_data['Health Score'].rank(
        ascending=False, method='min', na_option='bottom'
    ).fillna(0).astype(int)
    
    final_results = latest_year_data.reset_index().sort_values('Rank').reset_index(drop=True)
    
    return final_results, calculation_details

# --- THIS IS THE NEW, CORRECTED CODE TO USE INSTEAD ---
@st.cache_data
def calculate_historical_scores(_df_full, selected_ui_options, lake_ids_tuple):
    df_full = _df_full.copy()
    if df_full.empty or not selected_ui_options:
        return pd.DataFrame()

    final_weights = get_effective_weights(selected_ui_options, df_full.columns)
    params_to_process = list(final_weights.keys())

    # --- Pre-calculate GLOBAL normalization boundaries ---
    bounds = {}
    df_imputed_full = df_full.copy()
    for param in params_to_process:
        if param in df_imputed_full.columns:
            df_imputed_full = df_imputed_full.sort_values(by=['Lake_ID', 'Year'])
            df_imputed_full[param] = df_imputed_full.groupby('Lake_ID')[param].transform(lambda x: x.bfill().ffill())
            df_imputed_full[param] = df_imputed_full[param].fillna(df_imputed_full[param].median())

            value_min = df_imputed_full[param].min()
            value_max = df_imputed_full[param].max()

            slope_min, slope_max = 0, 0
            if param != 'HDI':
                all_slopes = df_imputed_full.groupby('Lake_ID').apply(
                    lambda x: linregress(x['Year'], x[param]).slope if len(x['Year'].unique()) > 1 else 0
                )
                slope_min = all_slopes.min()
                slope_max = all_slopes.max()

            bounds[param] = {
                'value_min': value_min, 'value_max': value_max,
                'slope_min': slope_min, 'slope_max': slope_max
            }

    def norm(val, v_min, v_max):
        if v_max == v_min: return 0.5
        return (val - v_min) / (v_max - v_min)

    all_historical_data = []
    years = sorted(df_full['Year'].unique())

    for year in years:
        df_subset = df_imputed_full[df_imputed_full['Year'] <= year].copy()
        if df_subset.empty: continue

        current_year_data = df_subset.loc[df_subset.groupby('Lake_ID')['Year'].idxmax()].set_index('Lake_ID')
        total_score_for_year = pd.Series(0.0, index=current_year_data.index) if len(current_year_data.index) > 1 else 0.0

        for param in params_to_process:
            if param not in current_year_data.columns: continue

            props = PARAMETER_PROPERTIES[param]
            b = bounds[param]

            raw_values = current_year_data[param]
            pv_score = norm(raw_values, b['value_min'], b['value_max'])
            if props['impact'] == 'negative':
                pv_score = 1.0 - pv_score

            if param == 'HDI':
                factor_score = pv_score
            else:
                trends = df_subset.groupby('Lake_ID').apply(
                    lambda x: linregress(x['Year'], x[param]) if len(x['Year'].unique()) > 1 else (0,0,0,1,0)
                )
                slopes = trends.apply(lambda x: x.slope if not isinstance(x, tuple) else x[0])
                p_values = trends.apply(lambda x: x.pvalue if not isinstance(x, tuple) else x[3])

                slope_score = norm(slopes, b['slope_min'], b['slope_max'])
                if props['impact'] == 'negative':
                    slope_score = 1.0 - slope_score

                if isinstance(p_values, pd.Series):
                    p_value_score = 1.0 - ( (p_values - p_values.min()) / (p_values.max() - p_values.min()) if p_values.max() != p_values.min() else 0.5)
                else:
                    p_value_score = 0.5

                safe_slope_score = slope_score.fillna(0.5) if isinstance(slope_score, pd.Series) else slope_score
                safe_p_value_score = p_value_score.fillna(0.5) if isinstance(p_value_score, pd.Series) else p_value_score
                factor_score = (pv_score + safe_slope_score + safe_p_value_score) / 3.0

            total_score_for_year += final_weights[param] * factor_score

        if isinstance(total_score_for_year, float):
            year_results = pd.DataFrame({
                'Year': [year],
                'Lake_ID': [current_year_data.index[0]],
                'Health Score': [total_score_for_year]
            })
        else:
            year_results = pd.DataFrame({
                'Year': year,
                'Lake_ID': total_score_for_year.index,
                'Health Score': total_score_for_year.values
            })
        all_historical_data.append(year_results)

    if not all_historical_data:
        return pd.DataFrame()

    historical_df = pd.concat(all_historical_data).reset_index(drop=True)
    historical_df['Rank'] = historical_df.groupby('Year')['Health Score'].rank(ascending=False, method='min').fillna(0).astype(int)
    return historical_df

# --- PLOTTING, AI, and PDF FUNCTIONS ---
@st.cache_data
def generate_grouped_plots_by_metric(_df, lake_ids_tuple, metrics):
    df = _df.copy()
    grouped_images = []
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_cycle = prop_cycle.by_key()['color']
    
    for metric in metrics:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        has_data = False
        color_map = {}

        for i, lake_id in enumerate(lake_ids_tuple):
            color_map[lake_id] = colors_cycle[i % len(colors_cycle)]
            lake_df = df[df['Lake_ID'] == lake_id].copy().sort_values("Year")
            if not lake_df.empty and metric in lake_df:
                lake_df[metric] = pd.to_numeric(lake_df[metric], errors='coerce')
                if lake_df[metric].notna().sum() > 0:
                    ax.plot(lake_df["Year"], lake_df[metric], marker='o', linestyle='-', label=f"Lake {lake_id}", color=color_map[lake_id])
                    has_data = True
                if lake_df[metric].notna().sum() > 1:
                    x = lake_df["Year"][lake_df[metric].notna()]
                    y = lake_df[metric][lake_df[metric].notna()]
                    slope, intercept, *_ = linregress(x, y)
                    ax.plot(x, intercept + slope * x, linestyle='--', alpha=0.7, color=color_map[lake_id])
        
        if not has_data:
            plt.close(fig)
            continue
            
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        grouped_images.append((f"Trend for: {metric}", buf, False))
    return grouped_images

@st.cache_data
def plot_radar_chart(calc_details, lake_ids_tuple):
    if not calc_details: return None, None, None
    params = sorted(list(next(iter(calc_details.values())).keys()))
    wrapped_params = [ '\n'.join(textwrap.wrap(p, 15)) for p in params ]
    data = {f"Lake {lake_id}": [details[p]['Factor Score'] for p in params] for lake_id, details in calc_details.items()}
    df_scores = pd.DataFrame(data, index=params)
    num_vars = len(params)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150, subplot_kw=dict(polar=True))
    for lake_name in df_scores.columns:
        values = df_scores[lake_name].tolist() + [df_scores[lake_name].tolist()[0]]
        ax.plot(angles, values, label=lake_name, linewidth=2); ax.fill(angles, values, alpha=0.2)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], color="grey", size=9)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(wrapped_params, size=10)
    ax.set_title("Lake Health Fingerprint", size=20, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    buf = BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.3); plt.close(fig)
    return "Figure 1: Lake Health Fingerprint", buf, False

# --- ADD THESE THREE FUNCTIONS TO YOUR SCRIPT ---

@st.cache_data
def plot_health_score_evolution_globally_normalized(_df, confirmed_params, lake_ids_tuple, start_year=2010):
    """
    (NEW FOR STREAMLIT)
    Generates a multi-panel plot showing the health score evolution for several lakes
    from a specific start year, with scores normalized on a global 0-1 scale for
    that period to enhance comparability.
    """
    historical_scores = calculate_historical_scores(_df, confirmed_params, lake_ids_tuple)
    if historical_scores.empty: return None, None, None

    recent_historical_scores = historical_scores[historical_scores['Year'] >= start_year].copy()
    if recent_historical_scores.empty: return None, None, None

    global_min_score = recent_historical_scores['Health Score'].min()
    global_max_score = recent_historical_scores['Health Score'].max()

    if (global_max_score - global_min_score) != 0:
        recent_historical_scores['Global Normalized Score'] = (recent_historical_scores['Health Score'] - global_min_score) / (global_max_score - global_min_score)
    else:
        recent_historical_scores['Global Normalized Score'] = 0.5

    lake_ids = sorted(list(lake_ids_tuple))
    n_lakes = len(lake_ids)
    ncols = min(n_lakes, 3); nrows = (n_lakes - 1) // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), dpi=150, sharey=True)
    axes = np.array(axes).flatten()

    for i, lake_id in enumerate(lake_ids):
        ax = axes[i]
        lake_data = recent_historical_scores[recent_historical_scores['Lake_ID'] == lake_id]
        if not lake_data.empty:
            ax.plot(lake_data['Year'], lake_data['Global Normalized Score'], marker='o', linestyle='-')
            ax.set_title(f"Lake {lake_id}")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        else:
            ax.set_title(f"Lake {lake_id} (No Data)")

    for ax in axes: ax.set_ylim(-0.05, 1.05)
    fig.suptitle(f'Globally Normalized Lake Health ({start_year}-Present)', fontsize=20, y=0.98)
    fig.supxlabel('Year', fontsize=14); fig.supylabel('Normalized Health Score (Global)', fontsize=14, x=0.01)
    for i in range(n_lakes, len(axes)): axes[i].set_visible(False)
    fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0, rect=[0.03, 0.03, 1, 0.95]);
    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return f"Figure 2: Globally Normalized Health ({start_year}-Present)", buf, False

@st.cache_data
def plot_self_normalized_evolution(_df, confirmed_params, lake_ids_tuple, start_year=2010):
    """
    (NEW FOR STREAMLIT)
    Generates individual plots for each lake's self-normalized journey.
    """
    historical_scores = calculate_historical_scores(_df, confirmed_params, lake_ids_tuple)
    if historical_scores.empty: return []

    recent_historical_scores = historical_scores[historical_scores['Year'] >= start_year].copy()
    if recent_historical_scores.empty: return []

    recent_historical_scores['Normalized Score'] = recent_historical_scores.groupby('Lake_ID')['Health Score'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0.5
    )

    image_list = []
    for lake_id in sorted(list(lake_ids_tuple)):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        lake_data = recent_historical_scores[recent_historical_scores['Lake_ID'] == lake_id]
        if lake_data.empty:
            plt.close(fig)
            continue

        ax.plot(lake_data['Year'], lake_data['Normalized Score'], marker='o', linestyle='-', color='darkgreen')
        if len(lake_data) > 1:
            worst = lake_data.loc[lake_data['Normalized Score'].idxmin()]
            best = lake_data.loc[lake_data['Normalized Score'].idxmax()]
            ax.annotate(f"Worst: {int(worst['Year'])}", (worst['Year'], worst['Normalized Score']), xytext=(0, -15), textcoords="offset points", ha='center', arrowprops=dict(arrowstyle="->", color='red'))
            ax.annotate(f"Best: {int(best['Year'])}", (best['Year'], best['Normalized Score']), xytext=(0, 15), textcoords="offset points", ha='center', arrowprops=dict(arrowstyle="->", color='blue'))

        ax.set_title(f"Recent Health Trajectory for Lake {lake_id}", fontsize=16)
        ax.set_ylim(-0.15, 1.15)
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
        image_list.append((f"Self-Normalized Journey: Lake {lake_id}", buf, True)) # Mark as grouped
    return image_list

# REPLACE the existing 'plot_holistic_trajectory_matrix_normalized' with this complete, corrected version:

@st.cache_data
def plot_holistic_trajectory_matrix_normalized(_df, _results, confirmed_params, lake_ids_tuple, start_year=2010):
    """
    (CORRECTED FOR STREAMLIT)
    Generates a normalized holistic trajectory matrix with quadrant labels.
    """
    historical_scores = calculate_historical_scores(_df, confirmed_params, lake_ids_tuple)
    if historical_scores.empty: return None, None, None

    recent_historical_scores = historical_scores[historical_scores['Year'] >= start_year].copy()
    if recent_historical_scores.empty: return None, None, None
    
    recent_trends = recent_historical_scores.groupby('Lake_ID').apply(lambda x: linregress(x['Year'], x['Health Score']).slope if len(x['Year'].unique()) > 1 else 0)
    plot_df = _results.set_index('Lake_ID').copy()
    plot_df['Recent Trend'] = recent_trends
    plot_df.dropna(subset=['Recent Trend', 'Health Score'], inplace=True)

    score_min, score_max = plot_df['Health Score'].min(), plot_df['Health Score'].max()
    plot_df['Normalized Status'] = (plot_df['Health Score'] - score_min) / (score_max - score_min) if score_max != score_min else 0.5

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.scatterplot(data=plot_df, x='Normalized Status', y='Recent Trend', hue=plot_df.index.astype(str), s=150, palette='viridis', legend=False, ax=ax)
    for i, row in plot_df.iterrows(): ax.text(row['Normalized Status'] + 0.01, row['Recent Trend'], f"Lake {i}", fontsize=9)
    
    avg_norm_score = plot_df['Normalized Status'].mean()
    ax.axhline(0, ls='--', color='gray'); ax.axvline(avg_norm_score, ls='--', color='gray')
    ax.set_title(f'Recent Holistic Lake Trajectory ({start_year}-Present)', fontsize=16, pad=20)
    ax.set_xlabel('Latest Normalized Health Score (Status)', fontsize=12)
    ax.set_ylabel(f'Recent Health Score Trend (Slope, {start_year}-Present)', fontsize=12)
    
    # --- THIS IS THE MISSING PART THAT HAS BEEN ADDED BACK ---
    # Get the y-axis limits to place the text correctly at the top and bottom
    y_lim_bottom, y_lim_top = ax.get_ylim()

    # Add the quadrant labels
    plt.text(avg_norm_score + 0.01, y_lim_top, 'Healthy & Resilient', ha='left', va='top', color='green', alpha=0.7, fontsize=12)
    plt.text(avg_norm_score + 0.01, y_lim_bottom, 'Healthy but Vulnerable', ha='left', va='bottom', color='orange', alpha=0.7, fontsize=12)
    plt.text(avg_norm_score - 0.01, y_lim_top, 'In Recovery', ha='right', va='top', color='blue', alpha=0.7, fontsize=12)
    plt.text(avg_norm_score - 0.01, y_lim_bottom, 'Critical Condition', ha='right', va='bottom', color='red', alpha=0.7, fontsize=12)
    # --- END OF ADDED PART ---

    plt.tight_layout(); 
    buf = BytesIO(); 
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.3); 
    plt.close(fig)
    return f"Figure 3: Holistic Trajectory ({start_year}-Present)", buf, False

@st.cache_data
def plot_hdi_vs_health_correlation(_results, lake_ids_tuple):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    if 'HDI' not in _results.columns or _results['HDI'].isnull().all():
        ax.text(0.5, 0.5, 'HDI data not available for this analysis.', ha='center', va='center')
    else:
        clean_results = _results.dropna(subset=['HDI', 'Health Score'])
        sns.regplot(data=clean_results, x='HDI', y='Health Score', ax=ax, ci=95, scatter_kws={'s': 100})
        for i, row in clean_results.iterrows(): ax.text(row['HDI'], row['Health Score'] + 0.01, f"Lake {row['Lake_ID']}", fontsize=9)
        if len(clean_results) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(clean_results['HDI'], clean_results['Health Score'])
            ax.text(0.05, 0.95, f'$R^2 = {r_value**2:.2f}$\np-value = {p_value:.3f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Socioeconomic Context: HDI vs. Lake Health', fontsize=16, pad=20)
    ax.set_xlabel('Human Development Index (HDI)', fontsize=12); ax.set_ylabel('Final Health Score', fontsize=12)
    plt.tight_layout(); buf = BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.3); plt.close(fig)
    return "Figure 4: HDI vs. Lake Health", buf, False

@st.cache_data
def build_detailed_ai_prompt(results, calc_details, lake_ids_tuple):
    prompt = ("You are an expert environmental data analyst. Your goal is to provide qualitative insights, not just quantitative comparisons. "
              "Generate a detailed comparative analysis of the following lakes based on their health parameters. For each parameter group (e.g., Climate, Water Quality), "
              "identify the 'best-in-class' and 'most-at-risk' lakes. Explain the *implications* of these differences. Use numbers only to support your qualitative statements.\n\n"
              "### Lake Data Profiles:\n")
    for _, row in results.iterrows():
        lake_id = row['Lake_ID']
        prompt += f"--- Lake {lake_id} ---\n"
        prompt += f"Final Health Score: {row['Health Score']:.3f} (Rank: {row['Rank']})\n"
        if lake_id in calc_details:
            for param, details in calc_details[lake_id].items():
                prompt += f"- {param}: {details['Raw Value']:.2f} (Factor Score: {details['Factor Score']:.3f})\n"
    prompt += "\n### Analysis Task:\n"
    prompt += "1. **Overall Summary:** Briefly state which lakes are healthiest and which are of most concern overall.\n"
    prompt += "2. **Parameter Group Analysis:** For each group (Climate, Water Quality, Land Cover, Socioeconomic), provide a paragraph comparing the lakes. Focus on insights, not just data. Example: 'In terms of Water Quality, Lake X stands out with excellent clarity, while Lake Y's high surface temperature is a significant concern for its ecosystem.'\n"
    return prompt

@st.cache_data
def build_figure_specific_ai_prompt(figure_title, data_summary):
    prompt = f"You are an environmental data analyst interpreting a figure for a report. The figure is titled '{figure_title}'. Below is a summary of the data used to create this figure.\n\n"
    prompt += "### Data Summary:\n" + data_summary + "\n\n"
    prompt += ("### Your Task:\nWrite a concise, insightful paragraph (3-5 sentences) that interprets this figure. "
               "Explain what the visual pattern reveals about the lakes being compared. Do not just list the data; "
               "provide a high-level interpretation of the findings shown in the chart.")
    return prompt

@st.cache_data
def generate_ai_insight(prompt):
    API_KEY = st.secrets.get("OPENROUTER_API_KEY")
    if not API_KEY: return "Error: API Key not found. Please configure it in Streamlit secrets."
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"model": "deepseek/deepseek-chat-v3.1:free", "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(API_URL, json=data, headers=headers, timeout=90)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 402: return "AI Analysis Failed: 402 Payment Required. Please check your OpenRouter account balance or rate limits."
        return f"AI Analysis Failed: HTTP Error {e.response.status_code} - {e}"
    except requests.exceptions.RequestException as e: return f"AI Analysis Failed: Network error - {e}"
    except (KeyError, IndexError): return "AI Analysis Failed: Could not parse a valid response from the AI model."


# --- PDF GENERATION WITH ROBUST TWO-PASS MANUAL CANVAS METHOD ---
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.drawRightString(A4[0] - 20, 20, f"Page {self._pageNumber} of {page_count}")


def _draw_report_content(c, bookmark_locations=None):
    """
    A helper function to draw all content onto the canvas.
    If `bookmark_locations` is provided, it draws the Table of Contents with page numbers.
    Otherwise, it just draws the content and records bookmark locations.
    """
    # This dictionary will be populated on the first pass
    if bookmark_locations is None:
        bookmark_locations = {}

    # Unpack data from the canvas object if it was attached
    df, results, calc_details, lake_ids, selected_ui_options = c.data_package
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='Title', parent=styles['h1'], alignment=1, fontSize=22, spaceAfter=20, textColor=colors.darkblue)
    header_style = ParagraphStyle(name='Header', parent=styles['h2'], alignment=0, fontSize=14, spaceBefore=12, spaceAfter=8, textColor=colors.darkslateblue)
    subtitle_style = ParagraphStyle(name='Subtitle', parent=styles['Normal'], alignment=1, fontSize=9, textColor=colors.grey, spaceAfter=12)
    justified_style = ParagraphStyle(name='Justified', parent=styles['Normal'], alignment=4, fontSize=10, leading=14)

    def draw_paragraph(text, style, x, y, width, height):
        p = Paragraph(str(text).replace('\n', '<br/>'), style)
        p.wrapOn(c, width, height)
        p.drawOn(c, x, y - p.height)
        return p.height
    
    # --- Page 1: Title & Table of Contents ---
    c.setTitle("Lake Health Report")
    y_cursor = A4[1] - 50
    y_cursor -= draw_paragraph("Lake Health Report", title_style, 40, y_cursor, A4[0] - 80, 100)
    y_cursor -= 20
    y_cursor -= draw_paragraph("", header_style, 40, y_cursor, A4[0] - 80, 50)
    
    params_to_plot = sorted([p for p in get_effective_weights(selected_ui_options, df.columns).keys() if p != 'HDI'])
    bookmarks = [("Health Score Ranking", "ranking", 0), ("AI-Powered Detailed Comparison", "ai_comparison", 0),
                 ("Health Score Calculation Breakdown", "breakdown", 0), ("Parameter Trend Plots", "parameter_plots", 0)]
    for p in params_to_plot:
        bookmarks.append((f"Trend for: {p}", f"plot_{p.replace(' ', '_')}", 1))
    bookmarks.append(("Case Study Analysis", "case_study", 0))

    for title, key, level in bookmarks:
        c.addOutlineEntry(title, key, level=level, closed=False)
        # If we have the locations (second pass), draw the page number
        if key in bookmark_locations:
            page_num_str = str(bookmark_locations[key])
            c.drawString(60 + (level*20), y_cursor, title)
            c.drawRightString(A4[0] - 60, y_cursor, page_num_str)
            c.linkRect("", key, (50, y_cursor - 5, A4[0] - 50, y_cursor + 15), relative=1, thickness=0.5, color=colors.blue)
        else: # First pass, just draw text
            c.drawString(60 + (level*20), y_cursor, title)
        y_cursor -= 20
    c.showPage()

    # --- Ranking Page ---
    bookmark_locations['ranking'] = c.getPageNumber()
    c.bookmarkPage('ranking')
    y_cursor = A4[1] - 80
    y_cursor -= draw_paragraph("Health Score Ranking", header_style, 40, y_cursor, A4[0]-80, 50)
    # FIX: Added explicit description
    y_cursor -= draw_paragraph("Scores are calculated on a normalized scale where 1 is the highest possible health score and 0 is the lowest.", subtitle_style, 40, y_cursor, A4[0]-80, 50)
    bar_start_x = 60; bar_height = 18; max_bar_width = A4[0] - bar_start_x - 150
    for _, row in results.iterrows():
        if y_cursor < 200:
            c.showPage(); y_cursor = A4[1] - 80
        score = row['Health Score']; rank = int(row['Rank'])
        color = colors.darkgreen if score > 0.75 else colors.orange if score > 0.5 else colors.firebrick
        c.setFillColor(color); c.rect(bar_start_x, y_cursor - bar_height, max(0, score) * max_bar_width, bar_height, fill=1, stroke=0)
        c.setFillColor(colors.black); c.setFont("Helvetica", 9); c.drawString(bar_start_x + 5, y_cursor - bar_height + 5, f"Lake {row['Lake_ID']} (Rank {rank}) - Score: {score:.3f}")
        y_cursor -= (bar_height + 10)
    
    # FIX: Add the legend
    y_cursor -= 20
    c.setFont("Helvetica-Bold", 10); c.drawString(bar_start_x, y_cursor, "Legend:")
    y_cursor -= 15; c.setFillColor(colors.darkgreen); c.rect(bar_start_x, y_cursor, 10, 10, fill=1)
    c.setFillColor(colors.black); c.drawString(bar_start_x + 15, y_cursor, "Good (Score > 0.75)")
    y_cursor -= 15; c.setFillColor(colors.orange); c.rect(bar_start_x, y_cursor, 10, 10, fill=1)
    c.setFillColor(colors.black); c.drawString(bar_start_x + 15, y_cursor, "Moderate (0.5 < Score <= 0.75)")
    y_cursor -= 15; c.setFillColor(colors.firebrick); c.rect(bar_start_x, y_cursor, 10, 10, fill=1)
    c.setFillColor(colors.black); c.drawString(bar_start_x + 15, y_cursor, "Poor (Score <= 0.5)")
    c.showPage()
    
    # --- AI Comparison Page ---
    bookmark_locations['ai_comparison'] = c.getPageNumber()
    c.bookmarkPage('ai_comparison')
    ai_prompt = build_detailed_ai_prompt(results, calc_details, tuple(lake_ids))
    ai_narrative = generate_ai_insight(ai_prompt).replace('\n', '<br/>')
    story = [Paragraph(ai_narrative, justified_style)]
    y_cursor = A4[1] - 50
    y_cursor -= draw_paragraph("AI-Powered Detailed Comparison", title_style, 40, y_cursor, A4[0]-80, 100)
    page_width = A4[0] - 80; x_pos = 40
    y_cursor -= 20
    available_height = y_cursor - 60
    while story:
        p = story.pop(0)
        frags = p.split(page_width, available_height)
        if len(frags) < 2 and frags:
            p.wrapOn(c, page_width, available_height); p.drawOn(c, x_pos, y_cursor - p.height)
        else:
            if frags:
                frags[0].wrapOn(c, page_width, available_height); frags[0].drawOn(c, x_pos, y_cursor - frags[0].height)
            story.extend(frags[1:])
            c.showPage()
            y_cursor = A4[1] - 80
            available_height = y_cursor - 60
    c.showPage()
    
    # --- Breakdown Page ---
    bookmark_locations['breakdown'] = c.getPageNumber()
    c.bookmarkPage('breakdown')
    y_cursor = A4[1] - 50
    y_cursor -= draw_paragraph("Health Score Calculation Breakdown", title_style, 40, y_cursor, A4[0]-80, 100)
    for lake_id in lake_ids:
        if y_cursor < 200: c.showPage(); y_cursor = A4[1] - 80
        y_cursor -= draw_paragraph(f"Breakdown for Lake {lake_id}", header_style, 40, y_cursor, A4[0]-80, 50)
        table_data = [['Parameter', 'Raw Val', 'Norm Pres.', 'Norm Trend', 'Norm P-Val', 'Factor Score', 'Weight', 'Contrib.']]
        for param, details in sorted(calc_details.get(lake_id, {}).items()):
             table_data.append([param[:18], f"{details.get('Raw Value', 0):.2f}", f"{details.get('Norm Pres.', 0):.3f}", f"{details.get('Norm Trend', 'N/A')}" if isinstance(details.get('Norm Trend'), str) else f"{details.get('Norm Trend', 0):.3f}", f"{details.get('Norm P-Val', 'N/A')}" if isinstance(details.get('Norm P-Val'), str) else f"{details.get('Norm P-Val', 0):.3f}", f"{details.get('Factor Score', 0):.3f}", f"{details.get('Weight', 0):.3f}", f"{details.get('Contribution', 0):.3f}",])
        table = Table(table_data, colWidths=[110, 60, 60, 60, 60, 60, 50, 60])
        table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.darkslategray), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,0), 10), ('BACKGROUND', (0,1), (-1,-1), colors.antiquewhite), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
        w, h = table.wrap(A4[0]-80, y_cursor)
        if y_cursor < h + 40: c.showPage(); y_cursor = A4[1] - 80
        table.drawOn(c, 40, y_cursor - h); y_cursor -= (h + 20)
    c.showPage()

    # --- Parameter Plots Page ---
    bookmark_locations['parameter_plots'] = c.getPageNumber()
    c.bookmarkPage("parameter_plots")
    plots = generate_grouped_plots_by_metric(df, tuple(lake_ids), params_to_plot)
    for i, (title, buf, _) in enumerate(plots):
        bookmark_key = f"plot_{title.replace('Trend for: ', '').replace(' ', '_')}"
        bookmark_locations[bookmark_key] = c.getPageNumber()
        if i > 0 and i % 2 == 0: c.showPage()
        y_pos = A4[1] - 50 if i % 2 == 0 else A4[1] * 0.5 - 60
        img_y_pos = A4[1] * 0.5 - 20 if i % 2 == 0 else 20
        c.bookmarkPage(bookmark_key)
        c.setFont("Helvetica-Bold", 14); c.drawCentredString(A4[0] / 2, y_pos, title)
        c.drawImage(ImageReader(buf), 40, img_y_pos, width=A4[0] - 80, height=A4[1] * 0.45 - 40, preserveAspectRatio=True)
        if i % 2 == 0 and i + 1 < len(plots): c.line(40, A4[1]*0.5 - 40, A4[0] - 40, A4[1]*0.5 - 40)
    c.showPage()

   # PASTE THIS ENTIRE NEW BLOCK IN ITS PLACE:

    # --- Case Study and New Individual Trajectory Pages ---
    # This part replaces the old "Case Study" section to integrate the new plots

    # First, add the new section for the Self-Normalized plots to the Table of Contents
    bookmark_locations['self_norm_plots'] = c.getPageNumber()
    c.bookmarkPage("self_norm_plots")
    c.addOutlineEntry("Self-Normalized Trajectories", "self_norm_plots", level=0, closed=False)

    # Generate and draw the self-normalized plots (this function returns a list of images)
    self_norm_plots = plot_self_normalized_evolution(df, tuple(selected_ui_options), tuple(lake_ids))
    for i, (title, buf, _) in enumerate(self_norm_plots):
        if i > 0 and i % 2 == 0: c.showPage()
        y_pos = A4[1] - 50 if i % 2 == 0 else A4[1] * 0.5 - 60
        img_y_pos = A4[1] * 0.5 - 20 if i % 2 == 0 else 20
        c.setFont("Helvetica-Bold", 14); c.drawCentredString(A4[0] / 2, y_pos, title)
        c.drawImage(ImageReader(buf), 40, img_y_pos, width=A4[0] - 80, height=A4[1] * 0.45 - 40, preserveAspectRatio=True)
        if i % 2 == 0 and i + 1 < len(self_norm_plots): c.line(40, A4[1]*0.5 - 40, A4[0] - 40, A4[1]*0.5 - 40)
    c.showPage()

    # Now, define and draw the main case study figures using the NEW plot functions
    bookmark_locations['case_study'] = c.getPageNumber()
    c.bookmarkPage('case_study')
    c.addOutlineEntry("Comparative Case Study", "case_study", level=0, closed=False)

    case_study_figures = [
        plot_radar_chart(calc_details, tuple(lake_ids)),
        # Use the NEW globally normalized evolution plot
        plot_health_score_evolution_globally_normalized(df, tuple(selected_ui_options), tuple(lake_ids)),
        # Use the NEW normalized holistic trajectory plot
        plot_holistic_trajectory_matrix_normalized(df, results, tuple(selected_ui_options), tuple(lake_ids)),
        # Keep the HDI plot
        plot_hdi_vs_health_correlation(results, tuple(lake_ids))
    ]

    # The rest of this loop remains the same
    for i, fig_data in enumerate(case_study_figures):
        if fig_data is None or fig_data[1] is None: continue
        if i > 0 : c.showPage()
        title, buf, _ = fig_data
        y_cursor = A4[1] - 80
        y_cursor -= draw_paragraph(title, header_style, 40, y_cursor, A4[0]-80, 100)
        c.drawImage(ImageReader(buf), 40, y_cursor - (A4[1] * 0.5), width=A4[0]-80, height=A4[1] * 0.5, preserveAspectRatio=True)
        y_cursor -= (A4[1] * 0.5 + 20)
        ai_prompt = build_figure_specific_ai_prompt(title, f"Analysis of lakes {lake_ids}.")
        ai_narrative = generate_ai_insight(ai_prompt).replace('\n', '<br/>')
        draw_paragraph(ai_narrative, justified_style, 40, y_cursor, A4[0]-80, A4[1]*0.4 - 40)
    c.showPage()

def generate_comparative_pdf_report(df, results, calc_details, lake_ids, selected_ui_options):
    buffer = BytesIO()
    
    data_package = (df, results, calc_details, lake_ids, selected_ui_options)
    
    # Pass 1: Draw to a dummy canvas to count pages and find bookmark locations
    dummy_canvas = canvas.Canvas(BytesIO(), pagesize=A4)
    dummy_canvas.data_package = data_package
    bookmark_locations = {}
    _draw_report_content(dummy_canvas, bookmark_locations)
    
    # Pass 2: Draw to the final canvas with the correct total page count
    final_canvas = NumberedCanvas(buffer, pagesize=A4)
    final_canvas.data_package = data_package
    _draw_report_content(final_canvas, bookmark_locations)
    final_canvas.save()
    
    buffer.seek(0)
    return buffer

# --- STREAMLIT APP LAYOUT (SINGLE PAGE DASHBOARD) ---
st.title("Lake Health Report")

# --- INITIALIZE APP & STATE ---
df_health_full, df_location, ui_options = prepare_all_data(HEALTH_DATA_PATH, LOCATION_DATA_PATH)
if df_health_full is None: st.stop()

if 'confirmed_parameters' not in st.session_state: st.session_state.confirmed_parameters = []
if 'lake_id_text' not in st.session_state: st.session_state.lake_id_text = ""

# --- NEW 3-COLUMN LAYOUT for NO SCROLLING ---
col1, col2, col3 = st.columns([1.5, 1.5, 2.0])

# --- COLUMN 1: Parameter Selection ---
with col1:
    with st.container(border=True):
        st.subheader("1. Select Parameters")
        with st.form(key='parameter_form'):
            cb_cols = st.columns(2)
            temp_selections = {}
            for i, param in enumerate(ui_options):
                with cb_cols[i % 2]:
                    temp_selections[param] = st.checkbox(param, value=(param in st.session_state.confirmed_parameters))
            
            if st.form_submit_button("Set Parameters", use_container_width=True):
                st.session_state.confirmed_parameters = [p for p, selected in temp_selections.items() if selected]
                st.session_state.pop('analysis_complete', None)
                st.rerun()

        if st.session_state.confirmed_parameters:
            st.info(f"**Active:** {', '.join(f'`{p}`' for p in st.session_state.confirmed_parameters)}")

# --- COLUMN 2: Lake Selection and Analysis ---
with col2:
    with st.container(border=True):
        st.subheader("2. Select Lakes")
        sorted_states = sorted(df_location['State'].unique())
        selected_state = st.selectbox("Filter by State:", sorted_states)
        
        filtered_districts = df_location[df_location['State'] == selected_state]['District'].unique()
        selected_district = st.selectbox("Filter by District:", sorted(filtered_districts))
        
        available_lakes = df_location[(df_location['State'] == selected_state) & (df_location['District'] == selected_district)]['Lake_ID'].unique()
        
        if len(available_lakes) > 0:
            add_col1, add_col2 = st.columns([2,1])
            with add_col1:
                lake_to_add = st.selectbox("Select from list:", sorted(available_lakes), label_visibility="collapsed")
            with add_col2:
                if st.button("Add"):
                    current_ids = set(int(x.strip()) for x in st.session_state.lake_id_text.split(',') if x.strip())
                    current_ids.add(lake_to_add)
                    st.session_state.lake_id_text = ", ".join(map(str, sorted(list(current_ids))))
                    st.session_state.pop('analysis_complete', None)
                    
        st.text_area("Selected Lakes (manual entry):", key="lake_id_text", height=100, on_change=lambda: st.session_state.pop('analysis_complete', None))
        
        try:
            lake_ids_to_analyze_initial = sorted([int(x.strip()) for x in st.session_state.lake_id_text.split(',') if x.strip()])
        except (ValueError, TypeError):
            st.error("Invalid Lake IDs. Please use comma-separated numbers.")
            lake_ids_to_analyze_initial = []

    with st.container(border=True):
        st.subheader("3. Run Analysis")
        is_disabled = not lake_ids_to_analyze_initial or not st.session_state.confirmed_parameters
        if st.button("🚀 Analyze Selected Lakes", disabled=is_disabled, use_container_width=True, type="primary"):
            st.session_state.pop('analysis_complete', None)
            st.session_state.pop('analysis_results', None)
            st.session_state.pop('pdf_buffer', None)

            all_available_lakes = set(df_health_full['Lake_ID'].unique())
            valid_lakes = [lid for lid in lake_ids_to_analyze_initial if lid in all_available_lakes]
            missing_lakes = [lid for lid in lake_ids_to_analyze_initial if lid not in all_available_lakes]
            
            if missing_lakes:
                st.warning(f"Warning: No health data found for Lake IDs: {', '.join(map(str, missing_lakes))}. They will be excluded from the analysis.")

            if not valid_lakes:
                st.error("None of the selected lakes have available health data for analysis.")
            else:
                with st.spinner("Analyzing and generating PDF report..."):
                    try:
                        selected_df = df_health_full[df_health_full["Lake_ID"].isin(valid_lakes)].copy()
                        
                        results, calc_details = calculate_lake_health_score(
                            selected_df, 
                            st.session_state.confirmed_parameters,
                            tuple(sorted(valid_lakes))
                        )
                        
                        if results.empty:
                            st.error("Analysis could not be completed for the selected lakes.")
                        else:
                            pdf_buffer = generate_comparative_pdf_report(selected_df, results, calc_details, valid_lakes, st.session_state.confirmed_parameters)
                            st.session_state.analysis_results = results
                            st.session_state.pdf_buffer = pdf_buffer
                            st.session_state.analysis_complete = True
                    except Exception as e: 
                        st.error("A critical error occurred during analysis.")
                        st.exception(e)

# --- COLUMN 3: MAP & RESULTS ---
with col3:
    with st.container(border=True):
        st.subheader(f"Choose Lake ID from Map")
        filtered_lakes_by_loc = df_location[(df_location['State'] == selected_state) & (df_location['District'] == selected_district)]
        if not filtered_lakes_by_loc.empty:
            map_center = [filtered_lakes_by_loc['Lat'].mean(), filtered_lakes_by_loc['Lon'].mean()]
            m = folium.Map(location=map_center, zoom_start=8)
            marker_cluster = MarkerCluster().add_to(m)
            for _, row in filtered_lakes_by_loc.iterrows():
                is_selected = row['Lake_ID'] in lake_ids_to_analyze_initial
                folium.Marker(
                    [row['Lat'], row['Lon']],
                    popup=f"<b>Lake ID:</b> {row['Lake_ID']} {'(Selected)' if is_selected else ''}",
                    tooltip=f"Lake ID: {row['Lake_ID']}",
                    icon=folium.Icon(color='green' if is_selected else 'blue', icon='water')
                ).add_to(marker_cluster)
            st_folium(m, height=275, use_container_width=True)

    with st.container(border=True):
        st.subheader("Results & Downloads")
        if st.session_state.get('analysis_complete'):
            if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
                st.dataframe(st.session_state.analysis_results[["Lake_ID", "Health Score", "Rank"]].style.format({"Health Score": "{:.3f}"}), use_container_width=True)
                
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    if 'pdf_buffer' in st.session_state and st.session_state.pdf_buffer:
                        report_lake_ids = st.session_state.analysis_results['Lake_ID'].tolist()
                        st.download_button(
                            label="📄 Download PDF Report", 
                            data=st.session_state.pdf_buffer, 
                            file_name=f"Full_Report_{'_'.join(map(str, report_lake_ids))}.pdf", 
                            mime="application/pdf", 
                            use_container_width=True
                        )
                with dl_col2:
                    report_lake_ids = st.session_state.analysis_results['Lake_ID'].tolist()
                    csv_data = df_health_full[df_health_full["Lake_ID"].isin(report_lake_ids)].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Data (CSV)", csv_data, f"data_{'_'.join(map(str, report_lake_ids))}.csv", "text/csv", use_container_width=True)
        else: 
            st.info("Results will appear here after analysis.")
