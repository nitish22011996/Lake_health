import streamlit as st
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
from reportlab.lib.pagesizes import A4, landscape
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
    'Evaporation': {'impact': 'negative', 'type': 'climate'},
    'Precipitation': {'impact': 'positive', 'type': 'climate'},
    'Lake Water Surface Temperature': {'impact': 'negative', 'type': 'water_quality'},
    'Water Clarity': {'impact': 'positive', 'type': 'water_quality'},
    'Barren Area': {'impact': 'negative', 'type': 'land_cover'},
    'Urban Area': {'impact': 'negative', 'type': 'land_cover'},
    'Vegetation Area': {'impact': 'positive', 'type': 'land_cover'},
    'HDI': {'impact': 'positive', 'type': 'socioeconomic'},
    'Area': {'impact': 'positive', 'type': 'physical'}
}
LAND_COVER_INTERNAL_COLS = ['Barren Area', 'Urban Area', 'Vegetation Area']


# --- CORE DATA & ANALYSIS FUNCTIONS (Unchanged) ---
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
def calculate_lake_health_score(_df, selected_ui_options):
    df = _df.copy()
    if not selected_ui_options: return pd.DataFrame(), {}
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
        present_value_score_result = norm(latest_values) if props['impact'] == 'positive' else rev_norm(latest_values)
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
            slope_norm_result = norm(slopes) if props['impact'] == 'positive' else rev_norm(slopes)
            p_value_norm_result = 1.0 - norm(p_values)
            factor_score_result = (present_value_score_result + slope_norm_result + p_value_norm_result) / 3.0
            for lake_id in latest_year_data.index:
                pv_score = present_value_score_result.loc[lake_id] if isinstance(present_value_score_result, pd.Series) else present_value_score_result
                s_norm = slope_norm_result.loc[lake_id] if isinstance(slope_norm_result, pd.Series) else slope_norm_result
                p_norm = p_value_norm_result.loc[lake_id] if isinstance(p_value_norm_result, pd.Series) else p_value_norm_result
                factor_score = factor_score_result.loc[lake_id] if isinstance(factor_score_result, pd.Series) else factor_score_result
                calculation_details[lake_id][param] = {'Raw Value': latest_values.loc[lake_id], 'Norm Pres.': pv_score, 'Norm Trend': s_norm, 'Norm P-Val': p_norm, 'Factor Score': factor_score, 'Weight': final_weights[param], 'Contribution': factor_score * final_weights[param]}
        total_score += final_weights[param] * factor_score_result
    latest_year_data['Health Score'] = total_score
    latest_year_data['Rank'] = latest_year_data['Health Score'].rank(ascending=False, method='min').astype(int)
    return latest_year_data.reset_index().sort_values('Rank'), calculation_details

@st.cache_data
def calculate_historical_scores(_df_full, selected_ui_options):
    df_full = _df_full.copy()
    all_historical_data = []
    years = sorted(df_full['Year'].unique())
    for year in years:
        df_subset = df_full[df_full['Year'] <= year]
        if not df_subset.empty:
            results, _ = calculate_lake_health_score(df_subset, selected_ui_options)
            if not results.empty:
                results['Year'] = year
                all_historical_data.append(results[['Year', 'Lake_ID', 'Health Score', 'Rank']])
    if not all_historical_data: return pd.DataFrame()
    historical_df = pd.concat(all_historical_data).reset_index(drop=True)
    historical_df['Rank'] = historical_df.groupby('Year')['Health Score'].rank(ascending=False, method='min').astype(int)
    return historical_df

# --- PLOTTING, AI, and PDF FUNCTIONS (Unchanged) ---
@st.cache_data
def generate_grouped_plots_by_metric(_df, lake_ids, metrics):
    df = _df.copy()
    grouped_images = []
    for metric in metrics:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        has_data = False
        for lake_id in lake_ids:
            lake_df = df[df['Lake_ID'] == lake_id].copy().sort_values("Year")
            if not lake_df.empty and metric in lake_df:
                lake_df[metric] = pd.to_numeric(lake_df[metric], errors='coerce')
                if lake_df[metric].notna().sum() > 0:
                    ax.plot(lake_df["Year"], lake_df[metric], marker='o', linestyle='-', label=f"Lake {lake_id}")
                    has_data = True
                if lake_df[metric].notna().sum() > 1:
                    x = lake_df["Year"][lake_df[metric].notna()]; y = lake_df[metric][lake_df[metric].notna()]
                    slope, intercept, *_ = linregress(x, y); ax.plot(x, intercept + slope * x, linestyle='--', alpha=0.7)
        if not has_data: plt.close(fig); continue
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Year", fontsize=12); ax.set_ylabel(metric, fontsize=12)
        ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        buf = BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.3); plt.close(fig)
        grouped_images.append((f"Trend for: {metric}", buf, False))
    return grouped_images

@st.cache_data
def plot_radar_chart(calc_details):
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

@st.cache_data
def plot_health_score_evolution(_df, confirmed_params):
    historical_scores = calculate_historical_scores(_df, confirmed_params)
    if historical_scores.empty: return None, None, None
    lake_ids = sorted(historical_scores['Lake_ID'].unique())
    n_lakes = len(lake_ids)
    ncols = min(n_lakes, 3); nrows = (n_lakes - 1) // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), dpi=150, sharey=True)
    axes = np.array(axes).flatten()
    for i, lake_id in enumerate(lake_ids):
        ax = axes[i]
        lake_data = historical_scores[historical_scores['Lake_ID'] == lake_id]
        ax.plot(lake_data['Year'], lake_data['Health Score'], marker='o', linestyle='-')
        ax.set_title(f"Lake {lake_id}"); ax.grid(True, linestyle='--', alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    fig.suptitle('Evolution of Overall Lake Health Score', fontsize=20, y=0.98)
    fig.supxlabel('Year', fontsize=14); fig.supylabel('Health Score', fontsize=14, x=0.01)
    for i in range(n_lakes, len(axes)): axes[i].set_visible(False)
    fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0, rect=[0.03, 0.03, 1, 0.95]);
    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return "Figure 2: Evolution of Overall Health Score", buf, False

@st.cache_data
def plot_holistic_trajectory_matrix(_df, _results, confirmed_params):
    historical_scores = calculate_historical_scores(_df, confirmed_params)
    if historical_scores.empty: return None, None, None
    trends = historical_scores.groupby('Lake_ID').apply(lambda x: linregress(x['Year'], x['Health Score']).slope if len(x['Year'].unique()) > 1 else 0)
    plot_df = _results.set_index('Lake_ID').copy(); plot_df['Overall Trend'] = trends
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.scatterplot(data=plot_df, x='Health Score', y='Overall Trend', hue=plot_df.index.astype(str), s=150, palette='viridis', legend=False, ax=ax)
    for i, row in plot_df.iterrows(): ax.text(row['Health Score'] + 0.005, row['Overall Trend'], f"Lake {i}", fontsize=9)
    avg_score = plot_df['Health Score'].mean()
    ax.axhline(0, ls='--', color='gray'); ax.axvline(avg_score, ls='--', color='gray')
    ax.set_title('Holistic Lake Trajectory Analysis', fontsize=16, pad=20)
    ax.set_xlabel('Latest Health Score (Status)', fontsize=12); ax.set_ylabel('Overall Health Score Trend (Slope)', fontsize=12)
    plt.text(avg_score + 0.01, ax.get_ylim()[1], 'Healthy & Resilient', ha='left', va='top', color='green', alpha=0.7)
    plt.text(avg_score + 0.01, ax.get_ylim()[0], 'Healthy but Vulnerable', ha='left', va='bottom', color='orange', alpha=0.7)
    plt.text(avg_score - 0.01, ax.get_ylim()[1], 'In Recovery', ha='right', va='top', color='blue', alpha=0.7)
    plt.text(avg_score - 0.01, ax.get_ylim()[0], 'Critical Condition', ha='right', va='bottom', color='red', alpha=0.7)
    plt.tight_layout(); buf = BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.3); plt.close(fig)
    return "Figure 3: Holistic Lake Trajectory", buf, False

@st.cache_data
def plot_hdi_vs_health_correlation(_results):
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
def build_detailed_ai_prompt(results, calc_details):
    prompt = ("You are an expert environmental data analyst. Your goal is to provide qualitative insights, not just quantitative comparisons. "
              "Generate a detailed comparative analysis of the following lakes based on their health parameters. For each parameter group (e.g., Climate, Water Quality), "
              "identify the 'best-in-class' and 'most-at-risk' lakes. Explain the *implications* of these differences. Use numbers only to support your qualitative statements.\n\n"
              "### Lake Data Profiles:\n")
    for _, row in results.iterrows():
        lake_id = row['Lake_ID']
        prompt += f"--- Lake {lake_id} ---\n"
        prompt += f"Final Health Score: {row['Health Score']:.3f} (Rank: {row['Rank']})\n"
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
    data = {"model": "deepseek/deepseek-chat:free", "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(API_URL, json=data, headers=headers, timeout=90)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 402: return "AI Analysis Failed: 402 Payment Required. Please check your OpenRouter account balance or rate limits."
        return f"AI Analysis Failed: HTTP Error {e.response.status_code} - {e}"
    except requests.exceptions.RequestException as e: return f"AI Analysis Failed: Network error - {e}"
    except (KeyError, IndexError): return "AI Analysis Failed: Could not parse a valid response from the AI model."

def generate_comparative_pdf_report(df, results, calc_details, lake_ids, selected_ui_options):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='Title', parent=styles['h1'], alignment=1, fontSize=22, spaceAfter=20, textColor=colors.darkblue)
    header_style = ParagraphStyle(name='Header', parent=styles['h2'], alignment=0, fontSize=14, spaceBefore=12, spaceAfter=8, textColor=colors.darkslateblue)
    subtitle_style = ParagraphStyle(name='Subtitle', parent=styles['Normal'], alignment=1, fontSize=9, textColor=colors.grey, spaceAfter=12)
    justified_style = ParagraphStyle(name='Justified', parent=styles['Normal'], alignment=4, fontSize=10, leading=14)
    
    def draw_paragraph(canvas_obj, text, style, x, y, width, height):
        p = Paragraph(str(text).replace('\n', '<br/>'), style)
        p.wrapOn(canvas_obj, width, height)
        p.drawOn(canvas_obj, x, y - p.height)
        return p.height
        
    # Page 1: Title and Ranking
    draw_paragraph(c, "Dynamic Lake Health Report", title_style, 40, A4[1] - 50, A4[0] - 80, 100)
    y_cursor = A4[1] - 120
    draw_paragraph(c, "Health Score Ranking", header_style, 40, y_cursor, A4[0]-80, 50); y_cursor -= 35
    draw_paragraph(c, "Scores are calculated on a normalized scale from 0 (lowest health) to 1 (highest health).", subtitle_style, 40, y_cursor, A4[0]-80, 50); y_cursor -= 40
    bar_start_x = 60; bar_height = 18; max_bar_width = A4[0] - bar_start_x - 150
    for _, row in results.iterrows():
        if y_cursor < 80: c.showPage(); y_cursor = A4[1] - 80
        score = row['Health Score']; rank = int(row['Rank'])
        color = colors.darkgreen if score > 0.75 else colors.orange if score > 0.5 else colors.firebrick
        c.setFillColor(color); c.rect(bar_start_x, y_cursor - bar_height, score * max_bar_width, bar_height, fill=1, stroke=0)
        c.setFillColor(colors.black); c.setFont("Helvetica", 9); c.drawString(bar_start_x + 5, y_cursor - bar_height + 5, f"Lake {row['Lake_ID']} (Rank {rank}) - Score: {score:.3f}")
        y_cursor -= (bar_height + 10)

    # Page 2: Detailed AI Comparison
    c.showPage()
    with st.spinner("Generating detailed AI parameter analysis..."):
        ai_prompt = build_detailed_ai_prompt(results, calc_details)
        ai_narrative = generate_ai_insight(ai_prompt)
    draw_paragraph(c, "AI-Powered Detailed Comparison", title_style, 40, A4[1] - 50, A4[0] - 80, 100)
    draw_paragraph(c, ai_narrative, justified_style, 40, A4[1] - 120, A4[0] - 80, A4[1] - 160)
    
    # Page 3: Calculation Breakdown Table
    c.showPage()
    y_cursor = A4[1] - 50
    draw_paragraph(c, "Health Score Calculation Breakdown", title_style, 40, y_cursor, A4[0]-80, 100); y_cursor -= 80
    for lake_id in lake_ids:
        table_data = [['Parameter', 'Raw Val', 'Norm Pres.', 'Norm Trend', 'Norm P-Val', 'Factor Score', 'Weight', 'Contrib.']]
        for param, details in sorted(calc_details[lake_id].items()):
             table_data.append([param[:18], f"{details.get('Raw Value', ''):.2f}", f"{details.get('Norm Pres.', ''):.3f}", f"{details.get('Norm Trend', 'N/A')}" if isinstance(details.get('Norm Trend'), str) else f"{details.get('Norm Trend', ''):.3f}", f"{details.get('Norm P-Val', 'N/A')}" if isinstance(details.get('Norm P-Val'), str) else f"{details.get('Norm P-Val', ''):.3f}", f"{details.get('Factor Score', ''):.3f}", f"{details.get('Weight', ''):.3f}", f"{details.get('Contribution', ''):.3f}",])
        table = Table(table_data, colWidths=[110, 60, 60, 60, 60, 60, 50, 60])
        table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.darkslategray), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,0), 10), ('BACKGROUND', (0,1), (-1,-1), colors.antiquewhite), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
        table_height = table.wrap(A4[0]-80, A4[1])[1]
        if y_cursor < table_height + 40: c.showPage(); y_cursor = A4[1] - 80
        draw_paragraph(c, f"Breakdown for Lake {lake_id}", header_style, 40, y_cursor, 200, 40); y_cursor -= 40
        table.drawOn(c, 40, y_cursor - table_height); y_cursor -= (table_height + 20)
    
    # Pages 4+: Individual Parameter Plots
    final_weights = get_effective_weights(selected_ui_options, df.columns)
    params_to_plot = sorted([p for p in final_weights.keys() if p != 'HDI'])
    plots = generate_grouped_plots_by_metric(df, lake_ids, params_to_plot)
    for i in range(0, len(plots), 2):
        c.showPage()
        title1, buf1, _ = plots[i]
        c.setFont("Helvetica-Bold", 14); c.drawCentredString(A4[0] / 2, A4[1] - 50, title1)
        c.drawImage(ImageReader(buf1), 40, A4[1] * 0.5 - 20, width=A4[0] - 80, height=A4[1] * 0.45, preserveAspectRatio=True)
        if i + 1 < len(plots):
            title2, buf2, _ = plots[i + 1]
            c.line(40, A4[1]*0.5 - 40, A4[0] - 40, A4[1]*0.5 - 40)
            c.setFont("Helvetica-Bold", 14); c.drawCentredString(A4[0] / 2, A4[1] * 0.5 - 60, title2)
            c.drawImage(ImageReader(buf2), 40, 20, width=A4[0] - 80, height=A4[1] * 0.45 - 40, preserveAspectRatio=True)

    # Case Study Section
    c.showPage()
    y_cursor = A4[1] - 50
    y_cursor -= draw_paragraph(c, "Case Study Analysis", title_style, 40, y_cursor, A4[0]-80, 100)
    y_cursor -= 20
    
    with st.spinner("Generating case study figures and insights..."):
        case_study_figures = [plot_radar_chart(calc_details), plot_health_score_evolution(df, selected_ui_options), plot_holistic_trajectory_matrix(df, results, selected_ui_options), plot_hdi_vs_health_correlation(results)]
        for i, fig_data in enumerate(case_study_figures):
            if fig_data is None or fig_data[1] is None: continue
            title, buf, is_landscape = fig_data
            if i > 0: c.showPage(); y_cursor = A4[1] - 50
            data_summary = f"Analysis of lakes {lake_ids} with parameters {selected_ui_options}."
            ai_prompt = build_figure_specific_ai_prompt(title, data_summary)
            ai_narrative = generate_ai_insight(ai_prompt)
            page_width, page_height = (landscape(A4) if is_landscape else A4)
            if is_landscape: c.setPageSize((page_width, page_height))
            y_cursor -= draw_paragraph(c, title, header_style, 40, y_cursor, page_width-80, 100)
            c.drawImage(ImageReader(buf), 40, y_cursor - (page_height * 0.5), width=page_width-80, height=page_height * 0.5, preserveAspectRatio=True)
            y_cursor -= (page_height * 0.5 + 20)
            draw_paragraph(c, ai_narrative, justified_style, 40, y_cursor, page_width-80, page_height*0.4 - 40)
            if is_landscape: c.setPageSize(A4)

    c.save(); buffer.seek(0)
    return buffer


# --- STREAMLIT APP LAYOUT (STABLE VERSION) ---
st.set_page_config(layout="wide")
st.title("🌊 Dynamic Lake Health Dashboard")

# --- INITIALIZE APP ---
df_health_full, df_location, ui_options = prepare_all_data(HEALTH_DATA_PATH, LOCATION_DATA_PATH)
if df_health_full is None: st.stop()

# --- STATE MANAGEMENT ---
if 'confirmed_parameters' not in st.session_state:
    st.session_state.confirmed_parameters = []
if 'selected_lake_ids' not in st.session_state:
    st.session_state.selected_lake_ids = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- NEW 2-COLUMN LAYOUT (CONTROLS | MAIN CONTENT) ---
col_controls, col_main = st.columns([1, 3])

# --- COLUMN 1: CONTROLS (Static and Always Visible) ---
with col_controls:
    with st.container(border=True):
        st.subheader("1. Select Parameters")
        # Using st.multiselect for a stable, clean selection experience.
        # This acts like a checklist and is the standard, robust way to handle this.
        confirmed_parameters = st.multiselect(
            "Select parameters for analysis:",
            options=ui_options,
            default=st.session_state.confirmed_parameters,
            key="parameter_selector"
        )
        st.session_state.confirmed_parameters = confirmed_parameters

    with st.container(border=True):
        st.subheader("2. Select Lakes")
        sorted_states = sorted(df_location['State'].unique())
        selected_state = st.selectbox("Filter by State:", sorted_states, key="state_selector")
        
        filtered_districts = df_location[df_location['State'] == selected_state]['District'].unique()
        selected_district = st.selectbox("Filter by District:", sorted(filtered_districts), key="district_selector")
        
        # All lakes from the filtered district are available for selection
        available_lakes = df_location[(df_location['State'] == selected_state) & (df_location['District'] == selected_district)]['Lake_ID'].unique()
        
        # The multiselect widget is the single source of truth for selected lakes
        selected_lake_ids = st.multiselect(
            "Select lakes for analysis:",
            options=sorted(available_lakes),
            default=st.session_state.selected_lake_ids,
            key="lake_selector"
        )
        st.session_state.selected_lake_ids = selected_lake_ids

    # --- ANALYSIS BUTTON ---
    is_disabled = not st.session_state.selected_lake_ids or not st.session_state.confirmed_parameters
    if st.button("🚀 Analyze Selected Lakes", disabled=is_disabled, use_container_width=True, type="primary"):
        st.session_state.analysis_results = None
        with st.spinner("Analyzing... This may take a moment."):
            try:
                lake_ids_to_analyze = st.session_state.selected_lake_ids
                selected_df = df_health_full[df_health_full["Lake_ID"].isin(lake_ids_to_analyze)].copy()
                if selected_df.empty:
                    st.error(f"No health data found for the selected Lake IDs: {lake_ids_to_analyze}")
                else:
                    results, calc_details = calculate_lake_health_score(selected_df, st.session_state.confirmed_parameters)
                    st.session_state.analysis_results = results
                    st.session_state.calc_details = calc_details
                    st.session_state.pdf_buffer = None # Clear old PDF
            except Exception as e: 
                st.error(f"A critical error occurred during analysis.")
                st.exception(e)

# --- COLUMN 2: MAIN CONTENT (MAP & RESULTS) ---
with col_main:
    tab_map, tab_results = st.tabs(["📍 Map View", "📊 Analysis & Downloads"])

    with tab_map:
        st.subheader(f"Map of Lakes in {selected_district}, {selected_state}")
        filtered_lakes_by_loc = df_location[(df_location['State'] == selected_state) & (df_location['District'] == selected_district)]
        if not filtered_lakes_by_loc.empty:
            map_center = [filtered_lakes_by_loc['Lat'].mean(), filtered_lakes_by_loc['Lon'].mean()]
            m = folium.Map(location=map_center, zoom_start=8)
            marker_cluster = MarkerCluster().add_to(m)
            for _, row in filtered_lakes_by_loc.iterrows():
                # Highlight selected lakes
                is_selected = row['Lake_ID'] in st.session_state.selected_lake_ids
                folium.Marker(
                    [row['Lat'], row['Lon']],
                    popup=f"<b>Lake ID:</b> {row['Lake_ID']}",
                    tooltip=f"Lake ID: {row['Lake_ID']}",
                    icon=folium.Icon(color='green' if is_selected else 'blue', icon='water')
                ).add_to(marker_cluster)
            st_folium(m, height=600, use_container_width=True)

    with tab_results:
        # This section will now render correctly
        if st.session_state.analysis_results is not None and not st.session_state.analysis_results.empty:
            with st.container(border=True):
                st.subheader("Health Score Results")
                st.dataframe(st.session_state.analysis_results[["Lake_ID", "Health Score", "Rank"]].style.format({"Health Score": "{:.3f}"}), use_container_width=True)
            
            with st.container(border=True):
                st.subheader("Download Center")
                # Use two columns for download buttons
                dl_col1, dl_col2 = st.columns(2)
                
                with dl_col1:
                    # PDF generation is computationally expensive, so we do it on demand
                    if st.button("📄 Generate Full PDF Report", use_container_width=True):
                         with st.spinner("Generating PDF Report... Please wait."):
                            pdf_buffer = generate_comparative_pdf_report(df_health_full[df_health_full["Lake_ID"].isin(st.session_state.selected_lake_ids)], st.session_state.analysis_results, st.session_state.calc_details, st.session_state.selected_lake_ids, st.session_state.confirmed_parameters)
                            st.session_state.pdf_buffer = pdf_buffer
                    
                    if 'pdf_buffer' in st.session_state and st.session_state.pdf_buffer is not None:
                        st.download_button(
                            label="✅ Click to Download PDF", 
                            data=st.session_state.pdf_buffer, 
                            file_name=f"Full_Report_{'_'.join(map(str, st.session_state.selected_lake_ids))}.pdf", 
                            mime="application/pdf", 
                            use_container_width=True
                        )

                with dl_col2:
                    csv_data = df_health_full[df_health_full["Lake_ID"].isin(st.session_state.selected_lake_ids)].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Filtered Data (CSV)", csv_data, f"data_{'_'.join(map(str, st.session_state.selected_lake_ids))}.csv", "text/csv", use_container_width=True)

        else: 
            st.info("ℹ️ Select parameters and lakes, then click 'Analyze Selected Lakes'.")
