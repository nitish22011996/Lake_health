import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from scipy.stats import linregress

# --- PDF & Map Specific Imports ---
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

# --- CONFIGURATION ---
LOCATION_DATA_PATH = 'HDI_lake_district.csv'
HEALTH_DATA_PATH = "lake_health_data.csv"


# --- FUNCTION DEFINITIONS ---

@st.cache_data
def load_data(file_path):
    """Loads a CSV file into a pandas DataFrame with caching."""
    return pd.read_csv(file_path)

def calculate_lake_health_score(df,
                                vegetation_weight=1/6, barren_weight=1/6, urban_weight=1/6,
                                precipitation_weight=1/6, evaporation_weight=1/6, air_temperature_weight=1/6):
    """
    Calculates a health score for lakes based on latest values and historical trends.
    """
    def norm(x): return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    def rev_norm(x): return 1 - norm(x)

    required_columns = ['Lake', 'Year', 'Vegetation Area', 'Barren Area', 'Urban Area',
                        'Precipitation', 'Evaporation', 'Air Temperature']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in health data: {col}")

    for col in ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    latest_year_data = df[df['Year'] == df['Year'].max()].copy()
    if latest_year_data.empty:
        return pd.DataFrame()

    latest_year_data['Vegetation Area Normalized'] = norm(latest_year_data['Vegetation Area'])
    latest_year_data['Barren Area Normalized'] = rev_norm(latest_year_data['Barren Area'])
    latest_year_data['Urban Area Normalized'] = rev_norm(latest_year_data['Urban Area'])
    latest_year_data['Precipitation Normalized'] = norm(latest_year_data['Precipitation'])
    latest_year_data['Evaporation Normalized'] = rev_norm(latest_year_data['Evaporation'])
    latest_year_data['Air Temperature Normalized'] = rev_norm(latest_year_data['Air Temperature'])

    for col in latest_year_data.columns:
        if 'Normalized' in col:
            latest_year_data[col] = latest_year_data[col].replace([np.inf, -np.inf, np.nan], 0)

    def get_slope_and_p(x, y):
        if len(x.unique()) < 2: return 0, 1
        slope, _, _, p_value, _ = linregress(x, y)
        return slope, p_value

    trends = df.groupby('Lake').apply(lambda x: pd.Series({
        'Vegetation Area Trend': get_slope_and_p(x['Year'], x['Vegetation Area'])[0],
        'Vegetation Area PValue': get_slope_and_p(x['Year'], x['Vegetation Area'])[1],
        'Barren Area Trend': get_slope_and_p(x['Year'], x['Barren Area'])[0],
        'Barren Area PValue': get_slope_and_p(x['Year'], x['Barren Area'])[1],
        'Urban Area Trend': get_slope_and_p(x['Year'], x['Urban Area'])[0],
        'Urban Area PValue': get_slope_and_p(x['Year'], x['Urban Area'])[1],
        'Precipitation Trend': get_slope_and_p(x['Year'], x['Precipitation'])[0],
        'Precipitation PValue': get_slope_and_p(x['Year'], x['Precipitation'])[1],
        'Evaporation Trend': get_slope_and_p(x['Year'], x['Evaporation'])[0],
        'Evaporation PValue': get_slope_and_p(x['Year'], x['Evaporation'])[1],
        'Air Temperature Trend': get_slope_and_p(x['Year'], x['Air Temperature'])[0],
        'Air Temperature PValue': get_slope_and_p(x['Year'], x['Air Temperature'])[1],
    })).reset_index()

    for factor, desirable in [
        ('Vegetation Area', 'positive'), ('Barren Area', 'negative'), ('Urban Area', 'negative'),
        ('Precipitation', 'positive'), ('Evaporation', 'negative'), ('Air Temperature', 'negative')
    ]:
        slope_col = f"{factor} Trend"
        pval_col = f"{factor} PValue"
        trends[f"{slope_col} Normalized"] = norm(trends[slope_col]) if desirable == 'positive' else rev_norm(trends[slope_col])
        trends[f"{pval_col} Normalized"] = 1 - norm(trends[pval_col])

    for col in trends.columns:
        if 'Normalized' in col:
            trends[col] = trends[col].replace([np.inf, -np.inf, np.nan], 0)

    latest_year_data = latest_year_data.set_index('Lake')
    trends = trends.set_index('Lake')
    combined_data = latest_year_data.join(trends, how='inner')

    def factor_score(factor, weight):
        return weight * (
            (combined_data[f'{factor} Normalized'] +
             combined_data[f'{factor} Trend Normalized'] +
             combined_data[f'{factor} PValue Normalized']) / 3
        )

    combined_data['Health Score'] = (
        factor_score('Vegetation Area', vegetation_weight) + factor_score('Barren Area', barren_weight) +
        factor_score('Urban Area', urban_weight) + factor_score('Precipitation', precipitation_weight) +
        factor_score('Evaporation', evaporation_weight) + factor_score('Air Temperature', air_temperature_weight)
    )

    combined_data['Health Score'] = norm(combined_data['Health Score'])
    combined_data['Rank'] = combined_data['Health Score'].rank(ascending=False, method='min')
    return combined_data.reset_index().sort_values('Rank')


def generate_ai_insight_combined(prompt):
    API_KEY = st.secrets.get("OPENROUTER_API_KEY")
    if not API_KEY:
        return "API Key not found. Please set OPENROUTER_API_KEY in Streamlit secrets."
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"model": "deepseek/deepseek-chat:free", "messages": [{"role": "user", "content": prompt}]}
    
    try:
        response = requests.post(API_URL, json=data, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Failed to generate insight due to a network error: {e}"
    except (KeyError, IndexError):
        return "Failed to parse the AI response. The response format may have changed."


def generate_grouped_plots_by_metric(df, lake_ids, metrics):
    grouped_images = []
    for metric in metrics:
        plt.figure(figsize=(10, 6), dpi=150)
        has_data = False
        for lake in lake_ids:
            lake_df = df[df['Lake'].astype(str) == str(lake)].copy()
            if not lake_df.empty and metric in lake_df:
                lake_df = lake_df.sort_values("Year")
                lake_df[metric] = pd.to_numeric(lake_df[metric], errors='coerce')
                if lake_df[metric].notna().sum() > 0:
                    plt.plot(lake_df["Year"], lake_df[metric], marker='o', linestyle='-', label=f"Lake {lake}")
                    has_data = True
                if lake_df[metric].notna().sum() > 1:
                    x = lake_df["Year"][lake_df[metric].notna()]
                    y = lake_df[metric][lake_df[metric].notna()]
                    slope, intercept, *_ = linregress(x, y)
                    plt.plot(x, intercept + slope * x, linestyle='--', alpha=0.7)

        if not has_data:
            plt.close()
            continue
            
        plt.title(f"{metric} Over Time for Selected Lakes", fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        grouped_images.append((metric, buf))
    return grouped_images


def generate_comparative_pdf_report(df, results, lake_ids):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    def writeln(text, step=16):
        nonlocal y
        lines = text.split('\n')
        for line in lines:
            wrapped_lines = [line[i:i+95] for i in range(0, len(line), 95)]
            for wrapped_line in wrapped_lines:
                if y < 60:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = height - 50
                c.drawString(40, y, wrapped_line)
                y -= step
    
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, y, "Comparative Lake Health Report")
    y -= 50
    c.setFont("Helvetica", 12)
    writeln(f"Analysis of Lakes: {', '.join(map(str, lake_ids))}")
    c.showPage()
    y = height - 50

    c.setFont("Helvetica-Bold", 14)
    writeln("Health Score Ranking")
    y -= 10
    c.setFont("Helvetica", 10)
    bar_start_x = 60
    bar_height = 18
    max_bar_width = width - bar_start_x - 150 
    
    for _, row in results.iterrows():
        if y < 100:
            c.showPage()
            y = height - 50
        score = row['Health Score']
        rank = int(row['Rank'])
        if score > 0.75: color = colors.darkgreen
        elif score > 0.5: color = colors.orange
        else: color = colors.firebrick
        c.setFillColor(color)
        bar_width = score * max_bar_width
        c.rect(bar_start_x, y, bar_width, bar_height, fill=1, stroke=0)
        c.setFillColor(colors.black)
        label = f"Lake {row['Lake']} (Rank {rank}) - Score: {score:.3f}"
        c.drawString(bar_start_x + 5, y + 5, label)
        y -= (bar_height + 10)
    y -= 20
    
    if y < 250:
        c.showPage()
        y = height - 50
        
    c.setFont("Helvetica-Bold", 14)
    writeln("AI-Generated Comparative Analysis")
    c.setFont("Helvetica", 10)
    prompt = f"Provide a detailed comparative analysis for the following lakes: {', '.join(map(str, lake_ids))}.\n\n"
    for _, row in results.iterrows():
        prompt += f"- Lake {row['Lake']} has a health score of {row['Health Score']:.3f} (Rank {int(row['Rank'])}).\n"
    prompt += "\nBased on this, discuss the likely contributing factors (Vegetation, Urban Area, Climate) and compare their overall health trajectories. Be concise and structured."
    ai_text = generate_ai_insight_combined(prompt)
    writeln(ai_text)
    
    metrics = ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']
    plots = generate_grouped_plots_by_metric(df, lake_ids, metrics)

    for i in range(0, len(plots), 2):
        c.showPage()
        y_positions = [height / 2 + 20, 50]
        for j in range(2):
            if i + j < len(plots):
                metric, img_buf = plots[i + j]
                img = ImageReader(img_buf)
                c.setFont("Helvetica-Bold", 12)
                c.drawCentredString(width / 2, y_positions[j] + 280, f"Comparison of: {metric}")
                c.drawImage(img, 40, y_positions[j], width=width-80, height=270, preserveAspectRatio=True, anchor='n')
    
    c.save()
    buffer.seek(0)
    return buffer


# --- STREAMLIT APP LAYOUT ---

st.set_page_config(layout="wide")
st.title("🌊 Lake Health Comparative Dashboard")

# Initialize session state for analysis results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
    st.session_state.selected_df_for_download = None
    st.session_state.pdf_buffer = None

with st.sidebar:
    st.header("Lake Selector")
    try:
        df_location = load_data(LOCATION_DATA_PATH)
        df_location.columns = df_location.columns.str.strip()
        df_location['State'] = df_location['State'].astype(str).str.strip()
        df_location['District'] = df_location['District'].astype(str).str.strip()

        sorted_states = sorted(df_location['State'].unique())
        selected_state = st.selectbox("Select State", sorted_states)

        filtered_districts = df_location[df_location['State'] == selected_state]['District'].unique()
        sorted_districts = sorted(filtered_districts)
        selected_district = st.selectbox("Select District", sorted_districts)

        filtered_lakes = df_location[(df_location['State'] == selected_state) & (df_location['District'] == selected_district)]
        lake_ids_in_district = sorted(filtered_lakes['Lake_ID'].unique())
        selected_lake_id = st.selectbox("Select a Lake ID to Add", lake_ids_in_district)

        if "selected_lake_ids" not in st.session_state:
            st.session_state.selected_lake_ids = []

        if st.button("Add Lake to Comparison"):
            if selected_lake_id not in st.session_state.selected_lake_ids:
                st.session_state.selected_lake_ids.append(selected_lake_id)
            else:
                st.info(f"Lake {selected_lake_id} is already in the list.")
            # Clear previous results when the list is modified
            st.session_state.analysis_results = None 
            st.session_state.pdf_buffer = None
            st.session_state.selected_df_for_download = None

    except FileNotFoundError:
        st.error(f"Location data file not found at '{LOCATION_DATA_PATH}'.")
        filtered_lakes = pd.DataFrame() 

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.subheader(f"Map of Lakes in {selected_district}, {selected_state}")
    if not filtered_lakes.empty:
        map_center = [filtered_lakes['Lat'].mean(), filtered_lakes['Lon'].mean()]
        m = folium.Map(location=map_center, zoom_start=8)
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in filtered_lakes.iterrows():
            popup_html = f"<b>Lake ID:</b> {row['Lake_ID']}<br><b>District:</b> {row['District']}"
            folium.Marker(
                location=[row['Lat'], row['Lon']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Lake ID: {row['Lake_ID']}",
                icon=folium.Icon(color='blue', icon='water')
            ).add_to(marker_cluster)
        st_folium(m, height=600, use_container_width=True)
    else:
        st.warning("No lake location data found for the selected district.")

with col2:
    st.subheader("Lakes Selected for Analysis")
    
    ids_text = ", ".join(map(str, st.session_state.selected_lake_ids))
    edited_ids_text = st.text_area(
        "Edit Lake IDs (comma-separated)", 
        value=ids_text,
        height=100,
        help="Add lakes using the sidebar or type IDs directly here."
    )
    
    try:
        if edited_ids_text:
            updated_ids = [int(x.strip()) for x in edited_ids_text.split(",") if x.strip().isdigit()]
        else:
            updated_ids = []
        
        if updated_ids != st.session_state.selected_lake_ids:
            st.session_state.analysis_results = None
            st.session_state.pdf_buffer = None
            st.session_state.selected_df_for_download = None
        
        st.session_state.selected_lake_ids = updated_ids
    except (ValueError, TypeError):
        st.warning("Invalid input. Please enter only comma-separated numbers.")

    lake_ids_to_analyze = st.session_state.get("selected_lake_ids", [])
    
    if st.button("Analyze Selected Lakes", disabled=not lake_ids_to_analyze, use_container_width=True):
        st.session_state.analysis_results = None
        st.session_state.pdf_buffer = None
        st.session_state.selected_df_for_download = None
        
        with st.spinner("Analyzing..."):
            try:
                df_health = load_data(HEALTH_DATA_PATH)
                lake_ids_str = [str(i) for i in lake_ids_to_analyze]
                selected_df = df_health[df_health["Lake"].astype(str).isin(lake_ids_str)]

                if selected_df.empty:
                    st.error(f"No health data found for the selected Lake IDs: {', '.join(lake_ids_str)}")
                else:
                    results = calculate_lake_health_score(selected_df)
                    st.session_state.analysis_results = results
                    st.session_state.selected_df_for_download = selected_df
                    
                    try:
                        st.session_state.pdf_buffer = generate_comparative_pdf_report(selected_df, results, lake_ids_to_analyze)
                    except Exception as pdf_e:
                        st.warning(f"⚠️ Could not generate PDF report. Error: {pdf_e}")
                        st.session_state.pdf_buffer = None

            except Exception as e:
                st.error(f"A critical error occurred during analysis: {e}")

    st.markdown("---") 

    if st.session_state.analysis_results is not None and not st.session_state.analysis_results.empty:
        st.subheader("Health Score Results")
        st.dataframe(st.session_state.analysis_results[["Lake", "Health Score", "Rank"]].style.format({"Health Score": "{:.3f}"}))

        st.subheader("Download Center")
        
        if st.session_state.selected_df_for_download is not None:
            csv_data = st.session_state.selected_df_for_download.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv_data,
                file_name=f"filtered_data_{'_'.join(map(str, lake_ids_to_analyze))}.csv",
                mime="text/csv",
                use_container_width=True
            )

        if st.session_state.pdf_buffer is not None:
            st.download_button(
                label="📄 Download PDF Report",
                data=st.session_state.pdf_buffer,
                # ** THE FIX IS HERE **
                file_name=f"comparative_report_{'_'.join(map(str, lake_ids_to_analyze))}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.info("ℹ️ Add lakes to the list above and click 'Analyze Selected Lakes'.")
else:
    st.info("Enter at least one Lake ID above to get started.")
