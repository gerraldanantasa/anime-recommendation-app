import streamlit as st  
import pandas as pd   
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.feature_extraction.text import CountVectorizer
import urllib.request 
from PIL import Image  

@st.cache_data  
@st.cache_resource
def load_data():  
    # Load the main dataframe  
    df = pd.read_csv('anime-gg.csv')[['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes', 'Synopsis', 'Image URL','Studios']]
    
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
    genre_type_df = genre_df.merge(type_df)  
    
    # Compute cosine similarity matrix  
    genre_type_cosine = genre_type_df.drop(columns='anime_id')  
    genre_type_cosine_matrix = cosine_similarity(genre_type_cosine)  
    
    return df, genre_type_df, genre_type_cosine_matrix  

# Recommendation function  
def get_recommendations(anime_name, df, genre_type_df, genre_type_cosine_matrix):  
    # Find target anime  
    target = df[df['Name'] == anime_name]  
    
    if target.empty:  
        return None  
    
    genre_type_cosine = genre_type_df.drop(columns='anime_id')  
    # Target Value  
    target_id = int(target[['anime_id']].values) 

    df_target = genre_type_df[genre_type_df['anime_id']==target_id]

    # Target cosine with others  
    cosine_sim = cosine_similarity(df_target.drop(columns='anime_id'), genre_type_cosine)
    
    # Target cosine with others
    cosine_sim = cosine_similarity(df_target.drop(columns='anime_id'), genre_type_cosine)
    cosine_sim_df = pd.DataFrame(cosine_sim).transpose().sort_values(0, ascending=False).rename(columns = {0:'Compatibility Score'})
    cosine_sim_df['Compatibility Score'] = round(cosine_sim_df['Compatibility Score']*100,2)

    #Getting anime id based off index
    list_recommended_index = cosine_sim_df.index.tolist() # Exclude first one cos its the anime itself

    recc_list = genre_type_df.iloc[list_recommended_index]['anime_id'].values.tolist()
    cosine_sim_df['anime_id'] = recc_list

    df_reccomended = df.iloc[list_recommended_index][df['anime_id']!=target_id]
    df_final = df_reccomended.merge(cosine_sim_df)
    return df_final.sort_values('Compatibility Score', ascending=False)  

# Streamlit App  
def main():  
    st.title('🎬 Anime Recommendation System')
    st.header('Click the little arrow on the top left')
    
    # Load data  
    df, genre_type_df, genre_type_cosine_matrix = load_data()  
    
    # Sidebar for search and filters  
    st.sidebar.header('Anime Search')  
    st.sidebar.write('Data is originated from https://www.kaggle.com/datasets/dsfelix/animes-dataset-2023/data')
    
    # Search box with autocomplete  
    anime_names = sorted(df['Name'].unique())  
    selected_anime = st.sidebar.selectbox(  
        'Type to Search for an Anime',   
        options=anime_names  
    )  
    
    # Recommendation Filters  
    st.sidebar.header('Recommendation Filters')  
    
    # Number of Recommendations Filter  
    recommendation_count = st.sidebar.selectbox(  
        'Number of Recommendations',  
        options=[10, 20, 50]  
    )  
    
    # Type Filter  
    unique_types = ['All'] + list(df['Type'].unique())  
    selected_type = st.sidebar.selectbox(  
        'Filter by Anime Type',  
        options=unique_types  
    )  
    
    # Additional filters  
    st.sidebar.header('Score Filter')  
    min_score = st.sidebar.slider('Minimum Score', 0.0, 10.0, 6.0, 0.1)  
    
    # Recommendation button  
    if st.sidebar.button('Get Recommendations'):  
        # Get recommendations  
        recommendations = get_recommendations(  
            selected_anime,   
            df,   
            genre_type_df,   
            genre_type_cosine_matrix  
        )  
        
        if recommendations is not None and not recommendations.empty:  
            # Filter by minimum score  
            recommendations = recommendations[recommendations['Score'] >= min_score]  
            
            # Filter by type if not 'All'  
            if selected_type != 'All':  
                recommendations = recommendations[recommendations['Type'] == selected_type]  
            
            # Limit number of recommendations  
            recommendations = recommendations.head(recommendation_count)  
            
            st.header(f'Recommendations based on {selected_anime}')
            
            # Display top 5 recommendations with detailed information
            for index, row in recommendations.head(5).iterrows():
                st.subheader(row['Name'])
                
                # Create columns for image and details
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Display image
                    st.image(row['Image URL'], use_column_width=True, caption='Anime Poster')
                
                with col2:
                    # Display details
                    st.write(f"**Type:** {row['Type']}")
                    st.write(f"**Score:** {row['Score']}")
                    st.write(f"**Compatibility Score:** {row['Compatibility Score']}%")
                    st.write(f"**Genres:** {row['Genres']}")
                    st.write(f"**Studios:** {row['Studios']}")
                
                # Display synopsis
                st.write("**Synopsis:**")
                st.write(row['Synopsis'])
                
                # Add a separator
                st.markdown("---")
            
            # Display full recommendations dataframe
            st.subheader('Full Recommendations')
            display_columns = [  
                'Name', 'Type', 'Score', 'Compatibility Score',   
                'Genres', 'Episodes'
            ]  
            st.dataframe(  
                recommendations[display_columns],   
                use_container_width=True,  
                hide_index=True  
            )  
        else:  
            st.error('No recommendations found!')  


if __name__ == '__main__':  
    main()