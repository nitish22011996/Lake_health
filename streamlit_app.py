import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
from scipy.stats import linregress

# --- PDF & Map Specific Imports ---
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Paragraph
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

def calculate_lake_health_score(df, selected_ui_options):
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
            df_imputed[param] = df_imputed[param].fillna(0)
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
            
            # --- FIX STARTS HERE: Check if the result is a standard tuple or a linregress object ---
            slopes = trends.apply(lambda x: x.slope if not isinstance(x, tuple) else x[0])
            p_values = trends.apply(lambda x: x.pvalue if not isinstance(x, tuple) else x[3])
            # --- FIX ENDS HERE ---
            
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

# --- PLOTTING FUNCTIONS FOR PDF CASE STUDY ---
def plot_health_score_composition(results, calc_details):
    composition_data = []
    for _, row in results.iterrows():
        lake_id = row['Lake_ID']
        details = calc_details[lake_id]
        comp = {'Lake_ID': f"Lake {lake_id}"}
        for param, detail_values in details.items():
            param_type = PARAMETER_PROPERTIES[param]['type']
            if param_type not in comp: comp[param_type] = 0
            comp[param_type] += detail_values['Contribution']
        composition_data.append(comp)
    df_comp = pd.DataFrame(composition_data).set_index('Lake_ID')
    df_comp = df_comp.reindex(index=[f"Lake {l}" for l in results['Lake_ID']])
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    df_comp.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title('Health Score Composition by Parameter Group', fontsize=16, pad=20)
    ax.set_xlabel('Lakes (sorted by rank)', fontsize=12)
    ax.set_ylabel('Weighted Contribution to Health Score', fontsize=12)
    ax.legend(title='Parameter Group', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return buf

def plot_holistic_trajectory_matrix(df, results, confirmed_params):
    historical_scores = calculate_historical_scores(df, confirmed_params)
    if historical_scores.empty: return None
    trends = historical_scores.groupby('Lake_ID').apply(lambda x: linregress(x['Year'], x['Health Score']).slope if len(x['Year'].unique()) > 1 else 0)
    plot_df = results.set_index('Lake_ID').copy()
    plot_df['Overall Trend'] = trends
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.scatterplot(data=plot_df, x='Health Score', y='Overall Trend', hue=plot_df.index.astype(str), s=150, palette='viridis', legend=False, ax=ax)
    for i, row in plot_df.iterrows():
        ax.text(row['Health Score'] + 0.005, row['Overall Trend'], f"Lake {i}", fontsize=9)
    avg_score = plot_df['Health Score'].mean()
    ax.axhline(0, ls='--', color='gray'); ax.axvline(avg_score, ls='--', color='gray')
    ax.set_title('Holistic Lake Trajectory Analysis', fontsize=16, pad=20)
    ax.set_xlabel('Latest Health Score (Status)', fontsize=12); ax.set_ylabel('Overall Health Score Trend (Slope)', fontsize=12)
    plt.text(avg_score, ax.get_ylim()[1], 'Healthy & Resilient', ha='left', va='top', color='green', alpha=0.7)
    plt.text(avg_score, ax.get_ylim()[0], 'Healthy but Vulnerable', ha='left', va='bottom', color='orange', alpha=0.7)
    plt.text(avg_score, ax.get_ylim()[1], 'In Recovery', ha='right', va='top', color='blue', alpha=0.7)
    plt.text(avg_score, ax.get_ylim()[0], 'Critical Condition', ha='right', va='bottom', color='red', alpha=0.7)
    plt.tight_layout(); buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return buf

def plot_hdi_vs_health_correlation(results):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    if 'HDI' not in results.columns or results['HDI'].isnull().all():
        ax.text(0.5, 0.5, 'HDI data not available for this analysis.', ha='center', va='center')
    else:
        clean_results = results.dropna(subset=['HDI', 'Health Score'])
        sns.regplot(data=clean_results, x='HDI', y='Health Score', ax=ax, ci=95, scatter_kws={'s': 100})
        for i, row in clean_results.iterrows():
            ax.text(row['HDI'], row['Health Score'] + 0.01, f"Lake {row['Lake_ID']}", fontsize=9)
        if len(clean_results) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(clean_results['HDI'], clean_results['Health Score'])
            ax.text(0.05, 0.95, f'$R^2 = {r_value**2:.2f}$\np-value = {p_value:.3f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Socioeconomic Context: HDI vs. Lake Health', fontsize=16, pad=20)
    ax.set_xlabel('Human Development Index (HDI)', fontsize=12)
    ax.set_ylabel('Final Health Score', fontsize=12)
    plt.tight_layout(); buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return buf

def plot_health_score_evolution(df, confirmed_params):
    historical_scores = calculate_historical_scores(df, confirmed_params)
    if historical_scores.empty: return None
    lake_ids = sorted(historical_scores['Lake_ID'].unique())
    n_lakes = len(lake_ids)
    ncols = min(n_lakes, 3); nrows = (n_lakes - 1) // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=150, sharey=True)
    axes = np.array(axes).flatten()
    for i, lake_id in enumerate(lake_ids):
        ax = axes[i]
        lake_data = historical_scores[historical_scores['Lake_ID'] == lake_id]
        ax.plot(lake_data['Year'], lake_data['Health Score'], marker='o', linestyle='-')
        ax.set_title(f"Lake {lake_id}"); ax.grid(True, linestyle='--', alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle('Evolution of Overall Lake Health Score', fontsize=20, y=1.02)
    fig.supxlabel('Year', fontsize=14); fig.supylabel('Health Score', fontsize=14)
    for i in range(n_lakes, len(axes)): axes[i].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.98]); buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return buf

def plot_rank_evolution(df, confirmed_params):
    historical_scores = calculate_historical_scores(df, confirmed_params)
    if historical_scores.empty: return None
    n_lakes = historical_scores['Lake_ID'].nunique()
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    for lake_id in sorted(historical_scores['Lake_ID'].unique()):
        lake_data = historical_scores[historical_scores['Lake_ID'] == lake_id].sort_values('Year')
        ax.plot(lake_data['Year'], lake_data['Rank'], marker='o', linestyle='-', label=f'Lake {lake_id}')
    ax.set_title('Evolution of Lake Health Rankings Over Time', fontsize=16, pad=20)
    ax.set_xlabel('Year', fontsize=12); ax.set_ylabel('Rank', fontsize=12)
    ax.invert_yaxis(); ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=n_lakes))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title='Lakes', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]); buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return buf

def build_detailed_ai_prompt(results, df, selected_ui_options):
    # This function is added for the AI analysis in the PDF. It's unchanged.
    def norm(x): return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    prompt = "You are an expert environmental data analyst. Generate a report with two sections: 'Overall Comparative Summary' and 'Parameter-Specific Analysis'.\n\n"
    prompt += "In the summary, provide a high-level conclusion comparing the lakes. In the parameter analysis, for each parameter group, explain how it contributes to the differences in health scores across the lakes. Focus on *why* they differ. Use the provided raw values and trends to support your analysis.\n\n"
    prompt += "### Lake Data Profiles\n"
    latest_year_df = df.loc[df.groupby('Lake_ID')['Year'].idxmax()].set_index('Lake_ID')
    final_weights = get_effective_weights(selected_ui_options, df.columns)
    params_to_process = list(final_weights.keys())
    norm_scores_all = {param: norm(latest_year_df[param]) for param in params_to_process if param in latest_year_df}
    for _, row in results.iterrows():
        lake_id = row['Lake_ID']
        prompt += f"--- Lake {lake_id} ---\n"
        prompt += f"Final Health Score: {row['Health Score']:.3f} (Rank: {row['Rank']})\n"
        lake_latest_data = latest_year_df.loc[lake_id]
        for param in params_to_process:
            if param not in lake_latest_data: continue
            props = PARAMETER_PROPERTIES[param]; raw_value = lake_latest_data[param]
            norm_result = norm_scores_all.get(param)
            if norm_result is None: continue
            single_norm_score = norm_result.loc[lake_id] if isinstance(norm_result, pd.Series) else norm_result
            assessment = "Good" if (single_norm_score > 0.5 if props['impact'] == 'positive' else single_norm_score < 0.5) else "Poor"
            prompt += f"- {param}: {raw_value:.2f} ({assessment})"
            if param != 'HDI':
                lake_history = df[df['Lake_ID'] == lake_id]
                if len(lake_history['Year'].unique()) > 1:
                    slope = linregress(lake_history['Year'], lake_history[param]).slope
                    trend_dir = "Increasing" if slope > 1e-3 else "Decreasing" if slope < -1e-3 else "Stable"
                    trend_impact_good = (slope > 0 and props['impact'] == 'positive') or (slope < 0 and props['impact'] == 'negative')
                    trend_impact = "Positive" if trend_impact_good else "Negative"
                    prompt += f", Trend: {trend_dir} ({trend_impact})\n"
                else: prompt += ", Trend: N/A\n"
            else: prompt += "\n"
    prompt += "\n### Required Analysis\n"
    prompt += "1.  **Overall Comparative Summary:**\n"
    prompt += "2.  **Parameter-Specific Analysis:** (Provide a bullet point for each selected parameter group)\n"
    return prompt

def generate_ai_insight_combined(prompt):
    # This function is added for the AI analysis in the PDF. It's unchanged.
    API_KEY = st.secrets.get("OPENROUTER_API_KEY")
    if not API_KEY: return "API Key not found. Please add your OpenRouter API key to Streamlit secrets to enable this feature."
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"model": "deepseek/deepseek-chat:free", "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(API_URL, json=data, headers=headers, timeout=90)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e: return f"Network error during AI analysis: {e}"
    except (KeyError, IndexError): return "Failed to parse AI response. The model may be temporarily unavailable."

# --- PDF GENERATION ENGINE ---
def generate_comparative_pdf_report(df, results, calc_details, lake_ids, selected_ui_options):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    styles = getSampleStyleSheet()
    justified_style = ParagraphStyle(name='Justified',parent=styles['Normal'],alignment=4,fontSize=10,leading=14)
    title_style = ParagraphStyle(name='Title', parent=styles['h1'], alignment=1, fontSize=20)
    header_style = ParagraphStyle(name='Header', parent=styles['h2'], alignment=0, spaceBefore=12, spaceAfter=6)
    y_cursor = height - margin
    def draw_paragraph(text, style, available_width):
        nonlocal y_cursor
        p = Paragraph(text.replace('\n', '<br/>'), style)
        p_width, p_height = p.wrapOn(c, available_width, height)
        if y_cursor - p_height < margin: c.showPage(); y_cursor = height - margin
        p.drawOn(c, margin, y_cursor - p_height)
        y_cursor -= (p_height + style.spaceAfter)
    
    # --- PAGE 1: Title, Ranking & Weights ---
    draw_paragraph("Dynamic Lake Health Report", title_style, width - 2 * margin); y_cursor -= 10
    draw_paragraph(f"<b>Lakes Analyzed:</b> {', '.join(map(str, lake_ids))}", justified_style, width - 2*margin)
    draw_paragraph(f"<b>Parameters Considered:</b> {', '.join(selected_ui_options)}", justified_style, width - 2*margin)
    final_weights = get_effective_weights(selected_ui_options, df.columns)
    weights_text = "<b>Effective Weights Used:</b><br/>" + "<br/>".join([f"- {param}: {weight:.3f}" for param, weight in sorted(final_weights.items())])
    draw_paragraph(weights_text, justified_style, width - 2*margin)
    draw_paragraph("Health Score Ranking", header_style, width - 2*margin)
    bar_start_x = margin + 20; bar_height = 18; max_bar_width = width - bar_start_x - 150
    for _, row in results.iterrows():
        if y_cursor - (bar_height + 10) < margin: c.showPage(); y_cursor = height - margin
        score = row['Health Score']; rank = int(row['Rank'])
        color = colors.darkgreen if score > 0.75 else colors.orange if score > 0.5 else colors.firebrick
        c.setFillColor(color); c.rect(bar_start_x, y_cursor - bar_height, score * max_bar_width, bar_height, fill=1, stroke=0)
        c.setFillColor(colors.black); c.setFont("Helvetica", 9); c.drawString(bar_start_x + 5, y_cursor - bar_height + 5, f"Lake {row['Lake_ID']} (Rank {rank}) - Score: {score:.3f}")
        y_cursor -= (bar_height + 10)

    # --- PAGE 2: AI Analysis ---
    c.showPage(); y_cursor = height - margin
    draw_paragraph("AI-Generated Comparative Analysis", header_style, width - 2*margin)
    with st.spinner("Generating AI analysis for PDF..."):
        detailed_prompt = build_detailed_ai_prompt(results, df, selected_ui_options)
        ai_text = generate_ai_insight_combined(detailed_prompt)
    draw_paragraph(ai_text, justified_style, width - 2*margin)

    # --- PAGE 3+: Detailed Tables ---
    c.showPage(); y_cursor = height - margin
    draw_paragraph("Detailed Calculation Breakdown", header_style, width - 2*margin)
    for lake_id in lake_ids:
        table_data = [['Parameter', 'Raw Val', 'Norm Pres.', 'Norm Trend', 'Norm P-Val', 'Factor Score', 'Weight', 'Contrib.']]
        for param, details in sorted(calc_details[lake_id].items()):
             table_data.append([param[:18], f"{details.get('Raw Value', ''):.2f}", f"{details.get('Norm Pres.', ''):.3f}", f"{details.get('Norm Trend', 'N/A')}" if isinstance(details.get('Norm Trend'), str) else f"{details.get('Norm Trend', ''):.3f}", f"{details.get('Norm P-Val', 'N/A')}" if isinstance(details.get('Norm P-Val'), str) else f"{details.get('Norm P-Val', ''):.3f}", f"{details.get('Factor Score', ''):.3f}", f"{details.get('Weight', ''):.3f}", f"{details.get('Contribution', ''):.3f}",])
        table = Table(table_data, colWidths=[110, 60, 60, 60, 60, 60, 50, 60])
        table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,0), 10), ('BACKGROUND', (0,1), (-1,-1), colors.beige), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
        _ , table_height = table.wrapOn(c, width, height)
        header_p = Paragraph(f"Breakdown for Lake {lake_id}", styles['h3']); _ , header_height = header_p.wrapOn(c, width - 2*margin, height)
        total_element_height = table_height + header_height + 20
        if y_cursor - total_element_height < margin: c.showPage(); y_cursor = height - margin
        header_p.drawOn(c, margin, y_cursor - header_height); y_cursor -= (header_height + 5)
        table.drawOn(c, margin, y_cursor - table_height); y_cursor -= (table_height + 15)
    
    # --- Case Study Section ---
    c.showPage(); y_cursor = height - margin
    draw_paragraph("Case Study: Holistic Lake Analysis", title_style, width - 2 * margin); y_cursor -= 10
    draw_paragraph("The following section presents a series of holistic visualizations designed to compare the lakes based on their overall health scores, trends, and contextual factors, avoiding focus on any single parameter.", justified_style, width - 2 * margin)
    with st.spinner("Generating advanced case study figures... (This may take a moment)"):
        case_study_figures = [
            ("Figure 1: Health Score Composition by Parameter Group", plot_health_score_composition, (results, calc_details), True),
            ("Figure 2: Holistic Lake Trajectory Analysis", plot_holistic_trajectory_matrix, (df, results, selected_ui_options), False),
            ("Figure 3: Socioeconomic Context - HDI vs. Lake Health", plot_hdi_vs_health_correlation, (results,), False),
            ("Figure 4: Evolution of Overall Lake Health Score", plot_health_score_evolution, (df, selected_ui_options), False),
            ("Figure 5: Evolution of Lake Health Rankings Over Time", plot_rank_evolution, (df, selected_ui_options), True)
        ]
        for title, func, args, use_landscape in case_study_figures:
            c.showPage()
            page_width, page_height = (landscape(A4) if use_landscape else A4)
            c.setPageSize((page_width, page_height))
            c.setFont("Helvetica-Bold", 14); c.drawCentredString(page_width / 2, page_height - 35, title)
            img_buf = func(*args)
            if img_buf: c.drawImage(ImageReader(img_buf), 40, 40, width=page_width-80, height=page_height-90, preserveAspectRatio=True)
            if use_landscape: c.setPageSize(A4)
            
    c.save(); buffer.seek(0)
    return buffer

# --- STREAMLIT APP LAYOUT ---
st.set_page_config(layout="wide")
st.title("🌊 Dynamic Lake Health Dashboard")
df_health_full, df_location, ui_options = prepare_all_data(HEALTH_DATA_PATH, LOCATION_DATA_PATH)
if df_health_full is None: st.stop()
if 'confirmed_parameters' not in st.session_state: st.session_state.confirmed_parameters = []
if "selected_lake_ids" not in st.session_state: st.session_state.selected_lake_ids = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

with st.sidebar:
    st.header("1. Select & Set Parameters")
    temp_selected_params = st.multiselect("Choose parameters for health score:", options=ui_options, default=st.session_state.get('confirmed_parameters', []))
    if st.button("Set Parameters"):
        st.session_state.confirmed_parameters = temp_selected_params; st.session_state.analysis_results = None; st.success("Parameters set!")
    if st.session_state.confirmed_parameters:
        st.markdown("---"); st.markdown("**Confirmed Parameters for Analysis:**");
        for param in st.session_state.confirmed_parameters: st.markdown(f"- `{param}`")
    st.markdown("---"); st.header("2. Select Lakes")
    sorted_states = sorted(df_location['State'].unique())
    selected_state = st.selectbox("Select State", sorted_states)
    filtered_districts = df_location[df_location['State'] == selected_state]['District'].unique()
    selected_district = st.selectbox("Select District", sorted(filtered_districts))
    filtered_lakes_by_loc = df_location[(df_location['State'] == selected_state) & (df_location['District'] == selected_district)]
    lake_ids_in_district = sorted(filtered_lakes_by_loc['Lake_ID'].unique())
    if lake_ids_in_district:
        selected_lake_id = st.selectbox("Select a Lake ID to Add", lake_ids_in_district)
        if st.button("Add Lake to Comparison"):
            if selected_lake_id not in st.session_state.selected_lake_ids: st.session_state.selected_lake_ids.append(selected_lake_id)
            st.session_state.analysis_results = None
    else: st.warning("No lakes found in this district.")

col1, col2 = st.columns([0.6, 0.4])
with col1:
    st.subheader(f"Map of Lakes in {selected_district}, {selected_state}")
    if not filtered_lakes_by_loc.empty:
        map_center = [filtered_lakes_by_loc['Lat'].mean(), filtered_lakes_by_loc['Lon'].mean()]
        m = folium.Map(location=map_center, zoom_start=8)
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in filtered_lakes_by_loc.iterrows():
            folium.Marker([row['Lat'], row['Lon']], popup=f"<b>Lake ID:</b> {row['Lake_ID']}", tooltip=f"Lake ID: {row['Lake_ID']}", icon=folium.Icon(color='blue', icon='water')).add_to(marker_cluster)
        st_folium(m, height=550, use_container_width=True)
with col2:
    st.subheader("Lakes Selected for Analysis")
    ids_text = ", ".join(map(str, st.session_state.selected_lake_ids))
    edited_ids_text = st.text_area("Edit Lake IDs (comma-separated)", ids_text, height=80)
    try:
        updated_ids = [int(x.strip()) for x in edited_ids_text.split(",") if x.strip()] if edited_ids_text else []
        if updated_ids != st.session_state.selected_lake_ids: st.session_state.analysis_results = None
        st.session_state.selected_lake_ids = updated_ids
    except (ValueError, TypeError): st.warning("Invalid input. Please enter comma-separated numbers.")
    lake_ids_to_analyze = st.session_state.get("selected_lake_ids", [])
    is_disabled = not lake_ids_to_analyze or not st.session_state.confirmed_parameters
    if st.button("Analyze Selected Lakes", disabled=is_disabled, use_container_width=True):
        st.session_state.analysis_results = None
        with st.spinner("Analyzing... This may take a moment."):
            try:
                selected_df = df_health_full[df_health_full["Lake_ID"].isin(lake_ids_to_analyze)].copy()
                if selected_df.empty:
                    st.error(f"No health data found for the selected Lake IDs: {lake_ids_to_analyze}")
                else:
                    results, calc_details = calculate_lake_health_score(selected_df, st.session_state.confirmed_parameters)
                    st.session_state.analysis_results = results; st.session_state.calc_details = calc_details
                    st.session_state.pdf_buffer = generate_comparative_pdf_report(selected_df, results, calc_details, lake_ids_to_analyze, st.session_state.confirmed_parameters)
            except Exception as e: st.error(f"A critical error occurred during analysis."); st.exception(e)
    st.markdown("---")
    if st.session_state.analysis_results is not None and not st.session_state.analysis_results.empty:
        st.subheader("Health Score Results")
        st.dataframe(st.session_state.analysis_results[["Lake_ID", "Health Score", "Rank"]].style.format({"Health Score": "{:.3f}"}), height=200)
        st.subheader("Download Center")
        csv_data = df_health_full[df_health_full["Lake_ID"].isin(lake_ids_to_analyze)].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Data (CSV)", csv_data, f"data_{'_'.join(map(str, lake_ids_to_analyze))}.csv", "text/csv", use_container_width=True)
        if 'pdf_buffer' in st.session_state and st.session_state.pdf_buffer: st.download_button("📄 Download PDF Report with Case Study", st.session_state.pdf_buffer, f"report_{'_'.join(map(str, lake_ids_to_analyze))}.pdf", "application/pdf", use_container_width=True)
    else: st.info("ℹ️ Set parameters and add at least one lake, then click 'Analyze'.")
