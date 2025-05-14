import streamlit as st
import pandas as pd
import numpy as np
import requests

# Load data only once
@st.cache_data
def load_data():
    return pd.read_csv("lake_health_data.csv")

def calculate_lake_health_score(df, vegetation_weight=1/6, barren_weight=1/6, urban_weight=1/6,
                                precipitation_weight=1/6, evaporation_weight=1/6, air_temperature_weight=1/6):
    df = df.copy()
    df = df[df['Lake'].isin(df['Lake'].unique())]

    # Latest year data
    latest_year_data = df[df['Year'] == df['Year'].max()].copy()
    if latest_year_data.empty:
        return pd.DataFrame()

    # Normalize level values
    latest_year_data['Vegetation Area Normalized'] = (latest_year_data['Vegetation Area'] - latest_year_data['Vegetation Area'].min()) / (latest_year_data['Vegetation Area'].max() - latest_year_data['Vegetation Area'].min())
    latest_year_data['Barren Area Normalized'] = 1 - (latest_year_data['Barren Area'] - latest_year_data['Barren Area'].min()) / (latest_year_data['Barren Area'].max() - latest_year_data['Barren Area'].min())
    latest_year_data['Urban Area Normalized'] = 1 - (latest_year_data['Urban Area'] - latest_year_data['Urban Area'].min()) / (latest_year_data['Urban Area'].max() - latest_year_data['Urban Area'].min())
    latest_year_data['Precipitation Normalized'] = (latest_year_data['Precipitation'] - latest_year_data['Precipitation'].min()) / (latest_year_data['Precipitation'].max() - latest_year_data['Precipitation'].min())
    latest_year_data['Evaporation Normalized'] = 1 - (latest_year_data['Evaporation'] - latest_year_data['Evaporation'].min()) / (latest_year_data['Evaporation'].max() - latest_year_data['Evaporation'].min())
    latest_year_data['Air Temperature Normalized'] = 1 - (latest_year_data['Air Temperature'] - latest_year_data['Air Temperature'].min()) / (latest_year_data['Air Temperature'].max() - latest_year_data['Air Temperature'].min())

    # Trend values
    trends = df.groupby("Lake").apply(lambda x: pd.Series({
        'Vegetation Area Trend': np.polyfit(x['Year'], x['Vegetation Area'], 1)[0],
        'Barren Area Trend': np.polyfit(x['Year'], x['Barren Area'], 1)[0],
        'Urban Area Trend': np.polyfit(x['Year'], x['Urban Area'], 1)[0],
        'Precipitation Trend': np.polyfit(x['Year'], x['Precipitation'], 1)[0],
        'Evaporation Trend': np.polyfit(x['Year'], x['Evaporation'], 1)[0],
        'Air Temperature Trend': np.polyfit(x['Year'], x['Air Temperature'], 1)[0]
    }))

    # Normalize trend
    for col in trends.columns:
        if 'Barren' in col or 'Urban' in col or 'Evaporation' in col or 'Temperature' in col:
            trends[col + ' Normalized'] = 1 - (trends[col] - trends[col].min()) / (trends[col].max() - trends[col].min())
        else:
            trends[col + ' Normalized'] = (trends[col] - trends[col].min()) / (trends[col].max() - trends[col].min())

    latest_year_data.set_index("Lake", inplace=True)
    combined = latest_year_data.join(trends, how='inner')

    combined['Health Score'] = (
        vegetation_weight * combined['Vegetation Area Normalized'] +
        barren_weight * combined['Barren Area Normalized'] +
        urban_weight * combined['Urban Area Normalized'] +
        precipitation_weight * combined['Precipitation Normalized'] +
        evaporation_weight * combined['Evaporation Normalized'] +
        air_temperature_weight * combined['Air Temperature Normalized'] +
        vegetation_weight * combined['Vegetation Area Trend Normalized'] +
        barren_weight * combined['Barren Area Trend Normalized'] +
        urban_weight * combined['Urban Area Trend Normalized'] +
        precipitation_weight * combined['Precipitation Trend Normalized'] +
        evaporation_weight * combined['Evaporation Trend Normalized'] +
        air_temperature_weight * combined['Air Temperature Trend Normalized']
    )

    combined['Rank'] = combined['Health Score'].rank(ascending=False)
    return combined.reset_index()

def get_insight_from_genai_api(lake_name, lake_data):
    API_KEY = "xxxx"  # Replace with your actual API key
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    prompt = f"""
    Analyze why Lake {lake_name} has a health score of {lake_data['Health Score']:.2f} and is ranked {int(lake_data['Rank'])} among all lakes. 
    Consider these metrics:
    Vegetation Area: {lake_data['Vegetation Area']}, Barren Area: {lake_data['Barren Area']}, Urban Area: {lake_data['Urban Area']}, 
    Precipitation: {lake_data['Precipitation']}, Evaporation: {lake_data['Evaporation']}, Air Temperature: {lake_data['Air Temperature']},
    and their respective trends and normalized values.
    Provide a short reasoning.
    """

    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(API_URL, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Failed to generate insight."

# --- Streamlit UI ---
st.title("Lake Health Scoring and Insights")

data = load_data()

num_lakes = st.number_input("How many lakes do you want to compare?", min_value=1, step=1)
lake_ids = []
for i in range(num_lakes):
    lake_id = st.text_input(f"Enter Lake ID #{i+1}", key=f"lake_{i}")
    if lake_id:
        lake_ids.append(lake_id)

if lake_ids:
    selected_df = data[data['Lake'].isin(lake_ids)]
    results = calculate_lake_health_score(selected_df)

    if not results.empty:
        st.subheader("Lake Health Scores")
        st.dataframe(results[['Lake', 'Health Score', 'Rank']])

        st.subheader("AI Insights")
        for _, row in results.iterrows():
            with st.expander(f"Insight for {row['Lake']}"):
                lake_data = selected_df[selected_df['Lake'] == row['Lake']].iloc[-1].to_dict()
                lake_data.update(row.to_dict())
                insight = get_insight_from_genai_api(row['Lake'], lake_data)
                st.write(insight)

        # CSV Download
        csv = selected_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Selected Lake Data as CSV", data=csv, file_name="selected_lake_data.csv", mime='text/csv')
    else:
        st.warning("No valid health data calculated. Check lake IDs or data availability.")
