import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
from io import BytesIO
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

# --- Load CSV in background ---
CSV_PATH = "lake_health_data.csv"  # <-- change this to your actual CSV path

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

# --- Lake Health Score Calculation Function ---
def calculate_lake_health_score(df,
                                vegetation_weight=1/6, barren_weight=1/6, urban_weight=1/6,
                                precipitation_weight=1/6, evaporation_weight=1/6, air_temperature_weight=1/6):
    def norm(x): return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    def rev_norm(x): return 1 - norm(x)

    required_columns = ['Lake', 'Year', 'Vegetation Area', 'Barren Area', 'Urban Area',
                        'Precipitation', 'Evaporation', 'Air Temperature']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    for col in ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    latest_year_data = df[df['Year'] == df['Year'].max()].copy()
    if latest_year_data.empty:
        return pd.DataFrame()

    # Normalize latest values
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
        ('Vegetation Area', 'positive'),
        ('Barren Area', 'negative'),
        ('Urban Area', 'negative'),
        ('Precipitation', 'positive'),
        ('Evaporation', 'negative'),
        ('Air Temperature', 'negative')
    ]:
        slope_col = f"{factor} Trend"
        pval_col = f"{factor} PValue"
        trends[f"{slope_col} Normalized"] = norm(trends[slope_col]) if desirable == 'positive' else rev_norm(trends[slope_col])
        trends[f"{pval_col} Normalized"] = 1 - norm(trends[pval_col])  # lower p-value is better

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
        factor_score('Vegetation Area', vegetation_weight) +
        factor_score('Barren Area', barren_weight) +
        factor_score('Urban Area', urban_weight) +
        factor_score('Precipitation', precipitation_weight) +
        factor_score('Evaporation', evaporation_weight) +
        factor_score('Air Temperature', air_temperature_weight)
    )

    combined_data['Rank'] = combined_data['Health Score'].rank(ascending=False)
    return combined_data.reset_index()
# --- Generate time series plots for each metric ---
def generate_metric_time_series_plots_per_lake(df, lake_ids, metrics):
    images = []
    for lake in lake_ids:
        lake_df = df[df['Lake'].astype(str) == str(lake)].copy()
        if lake_df.empty:
            continue
        for metric in metrics:
            plt.figure(figsize=(10, 5), dpi=150)
            lake_df[metric] = pd.to_numeric(lake_df[metric], errors='coerce')
            if lake_df[metric].dropna().empty:
                continue
            plt.plot(lake_df['Year'], lake_df[metric], marker='o', label=metric)
            plt.title(f"Time Series of {metric} for Lake {lake}")
            plt.xlabel("Year")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            buf = BytesIO()
            plt.savefig(buf, format='PNG')
            plt.close()
            buf.seek(0)
            images.append((f"Lake {lake} - {metric}", buf))
    return images

# --- AI insight generation for all lakes combined ---
def generate_ai_insight_combined(prompt):
    API_KEY = st.secrets["OPENROUTER_API_KEY"]
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(API_URL, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Failed to generate insight."

# Function to generate individual plots per lake for each metric with trendlines
def generate_grouped_plots_by_metric(df, lake_ids, metrics):
    grouped_images = []
    for metric in metrics:
        plt.figure(figsize=(10, 6), dpi=150)
        for lake in lake_ids:
            lake_df = df[df['Lake'].astype(str) == str(lake)].copy()
            if lake_df.empty or metric not in lake_df:
                continue
            lake_df = lake_df.sort_values("Year")
            lake_df[metric] = pd.to_numeric(lake_df[metric], errors='coerce')
            plt.plot(lake_df["Year"], lake_df[metric], marker='o', label=f"Lake {lake}")
            # Trendline
            if lake_df[metric].notna().sum() > 1:
                x = lake_df["Year"]
                y = lake_df[metric]
                slope, intercept, *_ = linregress(x, y)
                plt.plot(x, intercept + slope * x, linestyle='--', alpha=0.6)
        plt.title(f"{metric} Over Time")
        plt.xlabel("Year")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        grouped_images.append((metric, buf))
    return grouped_images

# Final PDF generation function
def generate_comparative_pdf_report(df, results, lake_ids):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 60

    def writeln(text, step=18):
        nonlocal y
        for line in text.split('\n'):
            while len(line) > 100:
                c.drawString(40, y, line[:100])
                y -= step
                line = line[100:]
                if y < 80:
                    c.showPage()
                    y = height - 60
            c.drawString(40, y, line)
            y -= step
            if y < 80:
                c.showPage()
                y = height - 60

    # Title Page
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, y, "Lake Health Comparative Report")
    y -= 40
    c.setFont("Helvetica", 12)
    writeln(f"Lakes compared: {', '.join(lake_ids)}")
    y -= 10
    c.showPage()
    y = height - 50

    # Colored Health Ranking Bars
    writeln("Health Score Rankings (Color-coded):")
    bar_start_x = 80
    bar_height = 16
    for i, row in results.iterrows():
        score = row['Health Score']
        rank = int(row['Rank'])
        color = colors.green if rank == 1 else colors.orange if rank == 2 else colors.red
        c.setFillColor(color)
        bar_width = score * 200  # Scaled bar
        c.rect(bar_start_x, y, bar_width, bar_height, fill=1)
        c.setFillColor(colors.black)
        c.drawString(bar_start_x + bar_width + 10, y + 2, f"Lake {row['Lake']} (Score: {score:.2f}, Rank: {rank})")
        y -= (bar_height + 10)
        if y < 100:
            c.showPage()
            y = height - 50

    # AI Insight
    # from your_ai_module import generate_ai_insight_combined  # Use actual import
    combined_prompt = "Provide a detailed comparative analysis for lakes: " + ", ".join(lake_ids) + ".\n"
    for _, row in results.iterrows():
        combined_prompt += (f"Lake {row['Lake']} has a health score of {row['Health Score']:.2f} and rank {int(row['Rank'])}.\n")
    combined_prompt += "Discuss the values and trends of Vegetation Area, Barren Area, Urban Area, Precipitation, Evaporation, and Air Temperature for these lakes."
    ai_text = generate_ai_insight_combined(combined_prompt)
    writeln("AI-Generated Comparative Analysis:\n" + "-"*40)
    writeln(ai_text)
    c.showPage()

    # Plots grouped by metric (with 2 plots per page in 2x2 layout)
    metrics = ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']
    plots = generate_grouped_plots_by_metric(df, lake_ids, metrics)

    for i in range(0, len(plots), 2):  # 2 plots per page
        y_positions = [height / 2 + 20, 50]
        for j in range(2):
            if i + j >= len(plots): break
            metric, img_buf = plots[i + j]
            c.drawString(40, y_positions[j] + 270, f"{metric}")
            img = ImageReader(img_buf)
            c.drawImage(img, 50, y_positions[j], width=500, height=250, preserveAspectRatio=True)
        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer

# --- Streamlit App ---
st.title("Lake Health Score Dashboard")

# Load data automatically
def load_data():
    return pd.read_csv("lake_health_data.csv")

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select lakes to analyze
lake_ids = st.text_input("Enter Lake IDs separated by commas (e.g. 630,2168266,737797)").replace(" ", "").split(",")
lake_ids = [x for x in lake_ids if x]  # Remove empty strings

if lake_ids:
    # Filter dataframe for selected lakes
    selected_df = df[df["Lake"].astype(str).isin(lake_ids)]
    if selected_df.empty:
        st.error("No data found for the entered Lake IDs.")
    else:
        results = calculate_lake_health_score(selected_df)
        if results.empty:
            st.error("No data available for latest year to calculate scores.")
        else:
            st.subheader("Lake Health Scores")
            st.dataframe(results[["Lake", "Health Score", "Rank"]])

            # Download CSV of selected lakes data
            csv = selected_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Original CSV for Selected Lakes", csv, "selected_lake_data.csv", "text/csv")

            # Download combined PDF report
            st.subheader("Download Combined PDF Report")
            pdf_buffer = generate_comparative_pdf_report(selected_df, results, lake_ids)
            st.download_button(
                label="📄 Download Combined Lake Health Report",
                data=pdf_buffer,
                file_name="combined_lake_health_report.pdf",
                mime="application/pdf"
            )
else:
    st.info("Enter at least one Lake ID above to get started.")
