import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objs as go

@st.cache_data
def load_data():
    # Load the main dataframe
    df = pd.read_csv('anime_dataset.csv')
    
    # Preprocess Genres
    df['Genres'] = df['Genres'].fillna('')  # Fill NaN values with empty string
    
    # Create CountVectorizer for Genres
    cv_genre = CountVectorizer(tokenizer=lambda x: x.split(', '))
    genre_matrix = cv_genre.fit_transform(df['Genres'])
    genre_names = cv_genre.get_feature_names_out()
    
    # Create Genre DataFrame
    genre_df = pd.DataFrame(
        genre_matrix.toarray(), 
        columns=genre_names, 
        index=df['anime_id']
    ).reset_index()
    
    # Create CountVectorizer for Types
    cv_type = CountVectorizer(tokenizer=lambda x: x.split(', '))
    type_matrix = cv_type.fit_transform(df['Type'])
    type_names = cv_type.get_feature_names_out()
    
    # Create Type DataFrame
    type_df = pd.DataFrame(
        type_matrix.toarray(), 
        columns=type_names, 
        index=df['anime_id']
    ).reset_index()
    
    # Merge genre and type dataframes
    genre_type_df = genre_df.merge(type_df, on='index')
    
    # Compute cosine similarity matrix
    genre_type_cosine = genre_type_df.drop(columns='index')
    genre_type_cosine_matrix = cosine_similarity(genre_type_cosine)
    
    return df, genre_type_df, genre_type_cosine_matrix

# Recommendation function
def get_recommendations(anime_name, df, genre_type_df, genre_type_cosine_matrix, selected_type=None):
    # Find target anime
    target = df[df['Name'] == anime_name]
    
    if target.empty:
        return None
    
    target_id = int(target['anime_id'].values[0])
    
    # Find the index of the target anime in genre_type_df
    target_index = genre_type_df[genre_type_df['index'] == target_id].index[0]
    
    # Compute cosine similarity
    cosine_sim = genre_type_cosine_matrix[target_index]
    
    # Create similarity dataframe
    cosine_sim_df = pd.DataFrame({
        'Compatibility Score': cosine_sim,
        'anime_id': genre_type_df['index']
    }).sort_values('Compatibility Score', ascending=False)
    
    # Remove the original anime from recommendations
    cosine_sim_df = cosine_sim_df[cosine_sim_df['anime_id'] != target_id]
    
    # Merge recommendations with original dataframe
    df_recommended = df[df['anime_id'].isin(cosine_sim_df['anime_id'])]
    df_recommended = df_recommended.merge(cosine_sim_df, on='anime_id')
    
    # Apply type filter if selected
    if selected_type:
        df_recommended = df_recommended[df_recommended['Type'] == selected_type]
    
    return df_recommended.sort_values('Compatibility Score', ascending=False).head(20)

# Image Grid Visualization
def create_image_grid(df):
    # Create a grid of images with hover information
    fig = px.layout_grid(
        df.head(12),  # Limit to 12 images
        image_column='Image URL',
        hover_data=['Name', 'Score', 'Type'],
        title='Anime Image Grid'
    )
    
    # Customize layout
    fig.update_layout(
        title_x=0.5,
        height=600,
        width=1000,
    )
    
    return fig

# Streamlit App
def main():
    st.title('🎬 Anime Recommendation System')
    
    # Load data
    df, genre_type_df, genre_type_cosine_matrix = load_data()
    
    # Sidebar for search and filters
    st.sidebar.header('Anime Search')
    
    # Search box with autocomplete
    anime_names = sorted(df['Name'].unique())
    selected_anime = st.sidebar.selectbox(
        'Search for an Anime', 
        options=anime_names
    )
    
    # Type Filter
    unique_types = df['Type'].unique()
    selected_type = st.sidebar.selectbox(
        'Filter by Anime Type',
        options=['All'] + list(unique_types)
    )
    selected_type = None if selected_type == 'All' else selected_type
    
    # Additional filters
    st.sidebar.header('Filters')
    min_score = st.sidebar.slider('Minimum Score', 0.0, 10.0, 6.0, 0.1)
    
    # Visualization Section
    st.header('Anime Visualization')
    
    # Image Grid Visualization
    if st.checkbox('Show Anime Image Grid'):
        # Create and display image grid
        image_grid_fig = create_image_grid(df)
        st.plotly_chart(image_grid_fig)
    
    # Recommendation button
    if st.sidebar.button('Get Recommendations'):
        # Get recommendations
        recommendations = get_recommendations(
            selected_anime, 
            df, 
            genre_type_df, 
            genre_type_cosine_matrix,
            selected_type
        )
        
        if recommendations is not None and not recommendations.empty:
            # Filter by minimum score
            recommendations = recommendations[recommendations['Score'] >= min_score]
            
            st.header(f'Recommendations based on {selected_anime}')
            
            # Create columns for image display
            cols = st.columns(4)
            
            # Display recommendations in a grid
            for i, (index, row) in enumerate(recommendations.iterrows()):
                with cols[i % 4]:
                    st.image(row['Image URL'], caption=row['Name'], width=200)
                    st.write(f"Score: {row['Score']}")
                    st.write(f"Compatibility: {row['Compatibility Score']:.2f}")
                    st.write(f"Type: {row['Type']}")
        else:
            st.error('No recommendations found!')

    # Optional: Genre Distribution Visualization
    st.sidebar.header('Genre Insights')
    if st.sidebar.checkbox('Show Genre Distribution'):
        # Count genre occurrences
        genre_counts = df['Genres'].str.split(', ', expand=True).stack().value_counts()
        
        # Create a bar chart using Plotly
        fig = px.bar(
            x=genre_counts.head(10).index, 
            y=genre_counts.head(10).values,
            labels={'x': 'Genres', 'y': 'Count'},
            title='Top 10 Genres in the Dataset'
        )
        
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()