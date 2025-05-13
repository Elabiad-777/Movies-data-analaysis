import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from clean_data import clean_movie_data

# Page config - MUST be the first Streamlit command
st.set_page_config(layout="wide", page_title="🎬 Advanced Movies Analysis")

# Load and clean data
@st.cache_data
def load_data():
    return clean_movie_data()

df = load_data()

st.title("🎬 Advanced Movies Analysis Dashboard")



# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    
    # Runtime slider
    min_runtime, max_runtime = st.slider(
        "Runtime (minutes)",
        int(df['runtime'].min()),
        int(df['runtime'].max()),
        (90, 120)
    )


    
    # Multi-select for ratings
    rating_filter = st.multiselect(
        "Select Ratings", 
        options=sorted(df['vote_average'].unique()),
        default=[7.0, 8.0]
    )
    
    revenue_filter = st.multiselect(
    "Revenue Category",
    options=df['revenue_category'].unique().tolist(),
    default=df['revenue_category'].unique().tolist()
)
    
    if 'genres' in df.columns:
        all_genres = sorted(set(g for sublist in df['genres'].str.split('|') for g in sublist))
        genre_filter = st.multiselect(
            "Select Genres",
            options=all_genres,
            default=all_genres[:3]
        )

df_filtered = df[
    (df['runtime'].between(min_runtime, max_runtime)) &
    (df['vote_average'].isin(rating_filter)) &
    (df['revenue_category'].isin(revenue_filter))
]

if 'genres' in df.columns and genre_filter:
    df_filtered = df_filtered[df_filtered['genres'].str.contains('|'.join(genre_filter))]

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Relationships", "🤖 ML Insights", "🔍 Raw Data"])
with tab1:
    
    st.markdown("""
    <style>
        div[data-testid="stMetric"] {
            background-color: #262730;
            padding: 15px;
            border-radius: 10px;
            color: white !important;
            text-align: center;
            border: 1px solid #5c5c5c;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Movies", len(df_filtered))
    with col2:
        st.metric("Avg Budget", f"${df_filtered['budget'].mean():,.0f}")
    with col3:
        st.metric("Avg Revenue", f"${df_filtered['revenue'].mean():,.0f}")

    fig1 = px.histogram(
        df_filtered,
        x="vote_average",
        nbins=20,
        title="Rating Distribution",
        color="revenue_category"
    )

    fig2 = px.box(
        df_filtered,
        y="runtime",
        x="revenue_category",
        title="Runtime by Revenue Category"
    )

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)



with tab2: 
    st.subheader("Budget vs Revenue Analysis")
    fig3 = px.scatter(
        df_filtered, 
        x="budget", 
        y="revenue", 
        color="vote_average",
        size="runtime",
        hover_name="original_title",
        trendline="lowess",
        title="Budget vs Revenue (Colored by Rating)"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("Feature Correlations")
    numeric_df = df_filtered.select_dtypes(include=['float64', 'int64'])
    fig4 = px.imshow(
        numeric_df.corr(),
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig4, use_container_width=True)

with tab3:  
    st.subheader("Machine Learning Predictions")
    
    st.image("heatmap.png", caption="Feature Correlations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("LinearRegression_scatter.png", caption="Linear Regression")
    with col2:
        st.image("DecisionTree_scatter.png", caption="Decision Tree")
    
    st.markdown("""
    ### Model Performance Comparison
    | Model            | MSE          | R² Score   |
    |------------------|--------------|------------|
    | Linear Regression| 2.1e+09      | 0.72       |
    | Ridge            | 2.1e+09      | 0.72       |
    | Lasso            | 2.3e+09      | 0.69       |
    | Decision Tree    | 1.8e+09      | 0.75       |
    """)

with tab4:  # Raw Data tab
    st.subheader("Filtered Movie Data")
    st.dataframe(df_filtered.sort_values('revenue', ascending=False))
    
    st.download_button(
        label="📥 Download Filtered Data",
        data=df_filtered.to_csv(index=False),
        file_name="filtered_movies.csv",
        mime="text/csv"
    )

st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .st-b7 {
        color: #2a9d8f;
    }
</style>
""", unsafe_allow_html=True)

