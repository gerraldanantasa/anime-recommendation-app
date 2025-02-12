import streamlit as st  
import pandas as pd   
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.feature_extraction.text import CountVectorizer
import urllib.request 
from PIL import Image  
import json
import os
import base64
from io import BytesIO
from PIL import Image


# Ensure watchlists directory exists
os.makedirs('watchlists', exist_ok=True)

# Global lists to track anime (for compatibility with existing code)
list_film = []
favorite_list = []


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load the image
logo_image = Image.open('FT_2025-02-12 20_46_37.034.png')  # Or use the hand-drawn image
base64_logo = image_to_base64(logo_image)

# Custom CSS for improved styling
st.set_page_config(
    page_title="Anime Recommender & Tracker",
    page_icon="🎬",
    layout="wide"
)

# In sidebar
st.sidebar.markdown(f"""
<div class="logo-container">
    <img src="data:image/png;base64,{base64_logo}" class="logo-image" alt="Anime GG Logo">
</div>
""", unsafe_allow_html=True)



# Add custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stHeader {
        color: #2c3e50;
    }
    .stSubheader {
        color: #34495e;
    }
    .recommendation-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Watchlist Management Functions
def save_watchlist(username, watchlist):
    """Save watchlist to a JSON file"""
    filename = f'watchlists/{username}_watchlist.json'
    with open(filename, 'w') as f:
        json.dump(watchlist, f, indent=4)

def load_watchlist(username):
    """Load watchlist from a JSON file"""
    filename = f'watchlists/{username}_watchlist.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def add_to_watchlist(username):
    """Add an anime to the user's watchlist from the dataset"""
    # Load existing watchlist
    global list_film
    list_film = load_watchlist(username)
    
    st.sidebar.header("🎥 Add to Watchlist")
    
    df = pd.read_csv('anime-gg.csv')[['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes', 'Synopsis', 'Image URL']]

    # Sort and unique anime names from the dataset
    anime_names = sorted(df['Name'].unique())
    
    # Search box with autocomplete for anime selection
    selected_anime = st.sidebar.selectbox(
        "Select Anime to Add",
        options=anime_names,
        index=None,
        placeholder="Type to search..."
    )
    
    # If an anime is selected
    if selected_anime:
        # Find the anime in the dataset
        anime_data = df[df['Name'] == selected_anime].iloc[0]
        
        # Prepare additional input fields
        episodes_watched = st.sidebar.number_input(
            "Episodes Watched", 
            min_value=0, 
            max_value=int(anime_data['Episodes']) if pd.notna(anime_data['Episodes']) else 1000,
            value=0
        )
        
        status_options = ['Not Started', 'Watching', 'Finished']
        status = st.sidebar.selectbox("Watch Status", status_options)
        
        # Optional: Allow user to add initial score
        score = st.sidebar.number_input(
            "Your Score", 
            min_value=0.0, 
            max_value=10.0,  
            value=0.0
        )
        
        if st.sidebar.button("Add to Watchlist"):
            # Check for duplicates
            if not any(str(film['Name']).lower() == str(selected_anime).lower() for film in list_film):
                new_entry = {
                    'Name': selected_anime,
                    'Genres': anime_data['Genres'].split(',') if pd.notna(anime_data['Genres']) else [],
                    'Type': anime_data['Type'] if pd.notna(anime_data['Type']) else 'Anime',
                    'Episodes': int(anime_data['Episodes']) if pd.notna(anime_data['Episodes']) else 0,
                    'Episodes Watched': episodes_watched,
                    'Status': status,
                    'Score': score,
                    'Synopsis': str(anime_data['Synopsis']) if pd.notna(anime_data['Synopsis']) else '',
                    'Image URL': str(anime_data['Image URL']) if pd.notna(anime_data['Image URL']) else ''
                }
                
                list_film.append(new_entry)
                
                # Save updated watchlist
                save_watchlist(username, list_film)
                st.sidebar.success(f"'{selected_anime}' added to watchlist!")
            else:
                st.sidebar.warning(f"'{selected_anime}' already exists in the list.")

def display_list_film():
    """Display the current list_film across all pages"""
    if list_film:
        # Create a DataFrame for display
        df = pd.DataFrame(list_film)
        
        # Columns to display
        display_columns = ['Name', 'Genres', 'Type', 'Episodes', 'Episodes Watched', 'Status', 'Score']
        
        # Expander to show full watchlist
        with st.expander("📺 Current Watchlist"):
            # Display the watchlist
            st.dataframe(
                df[display_columns], 
                use_container_width=True, 
                hide_index=True
            )
            
            # Additional statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anime", len(df))
            with col2:
                completed = len(df[df['Status'] == 'Finished'])
                st.metric("Completed", completed)
            with col3:
                total_episodes = df['Episodes Watched'].sum()
                st.metric("Total Episodes Watched", total_episodes)

def display_watchlist(username):
    """Display the user's watchlist"""
    global list_film
    list_film = load_watchlist(username)
    
    st.header("📺 Detailed Watchlist")
    
    if list_film:
        # Create a DataFrame for display
        df = pd.DataFrame(list_film)
        
        # Columns to display
        display_columns = ['Name', 'Genres', 'Type', 'Episodes', 'Episodes Watched', 'Status', 'Score']
        
        # Display the watchlist with more detailed view
        st.dataframe(
            df[display_columns], 
            use_container_width=True, 
            hide_index=True
        )
        
        # Detailed view with charts or additional information
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution
            status_counts = df['Status'].value_counts()
            st.subheader("Watchlist Status")
            st.bar_chart(status_counts)
        
        with col2:
            # Progress visualization
            st.subheader("Watching Progress")
            df['Progress'] = df['Episodes Watched'] / df['Episodes'] * 100
            st.bar_chart(df.set_index('Name')['Progress'])
        
        # Optional: Delete functionality
        delete_anime = st.selectbox("Select Anime to Delete", [''] + df['Name'].tolist())
        
        if delete_anime:
            if st.button("Confirm Delete"):
                # Remove the selected anime
                list_film = [film for film in list_film if film['Name'] != delete_anime]
                
                # Save updated watchlist
                save_watchlist(username, list_film)
                st.success(f"'{delete_anime}' removed from watchlist!")
                st.rerun()
    else:
        st.info("Your watchlist is empty. Add some anime!")

def update_watchlist(username):
    """Update an existing anime in the user's watchlist"""
    global list_film
    list_film = load_watchlist(username)
    
    st.sidebar.header("✏️ Update Watchlist")
    
    if list_film:
        # Create a list of anime names for selection
        anime_names = [film['Name'] for film in list_film]
        selected_anime = st.sidebar.selectbox("Select Anime to Update", anime_names)
        
        # Find the selected anime
        selected_film = next((film for film in list_film if film['Name'] == selected_anime), None)
        
        if selected_film:
            # Update fields
            episodes_watched = st.sidebar.number_input(
                "Episodes Watched", 
                min_value=0, 
                max_value=selected_film['Episodes'], 
                value=selected_film['Episodes Watched']
            )
            
            status_options = ['Not Started', 'Watching', 'Finished']
            status = st.sidebar.selectbox(
                "Watch Status", 
                status_options, 
                index=status_options.index(selected_film['Status'])
            )
            
            score = st.sidebar.number_input(
            "Your Score", 
            min_value=0.0, 
            max_value=10.0,  
            value=0.0
        )
            
            if st.sidebar.button("Update"):
                # Update the film entry
                selected_film['Episodes Watched'] = episodes_watched
                selected_film['Status'] = status
                selected_film['Score'] = score
                
                # Save updated watchlist
                save_watchlist(username, list_film)
                st.sidebar.success(f"'{selected_anime}' updated successfully!")
    else:
        st.sidebar.info("Add anime to your watchlist first!")

# Recommendation function  
def get_recommendations(anime_name, df, genre_type_df, genre_type_cosine_matrix):  
    # Find target anime  
    target = df[df['Name'] == anime_name]  
    
    if target.empty:  
        return None  
    
    # Ensure genre_type_df has the required columns
    if 'anime_id' not in genre_type_df.columns:
        genre_type_df['anime_id'] = df['anime_id']
    
    # Prepare cosine similarity matrix
    genre_columns = [col for col in genre_type_df.columns if col not in ['Name', 'anime_id']]
    genre_type_cosine = genre_type_df[genre_columns]
    
    # Target Value  
    try:
        target_id = int(target['anime_id'].values[0])
    except (IndexError, ValueError):
        return None

    # Find the index of the target anime in the genre_type_df
    target_index = genre_type_df[genre_type_df['anime_id'] == target_id].index

    if len(target_index) == 0:
        return None

    # Compute cosine similarity
    try:
        cosine_sim = cosine_similarity(
            genre_type_df.loc[target_index, genre_columns], 
            genre_type_cosine
        )[0]
    except Exception as e:
        st.error(f"Error in cosine similarity calculation: {e}")
        return None

    # Create similarity DataFrame
    cosine_sim_df = pd.DataFrame({
        'anime_id': genre_type_df['anime_id'],
        'Compatibility Score': cosine_sim
    })
    
    # Sort and process recommendations
    cosine_sim_df = cosine_sim_df[cosine_sim_df['anime_id'] != target_id]
    cosine_sim_df['Compatibility Score'] = round(cosine_sim_df['Compatibility Score'] * 100, 2)
    cosine_sim_df = cosine_sim_df.sort_values('Compatibility Score', ascending=False)

    # Merge with original dataframe
    try:
        recommendations = df[df['anime_id'].isin(cosine_sim_df['anime_id'])].copy()
        recommendations = recommendations.merge(cosine_sim_df, on='anime_id')
        recommendations = recommendations.sort_values('Compatibility Score', ascending=False)
        
        return recommendations
    except Exception as e:
        st.error(f"Error in recommendation processing: {e}")
        return None  

def load_data():  
    # Load the main dataframe  
    df = pd.read_csv('anime-gg.csv')[['anime_id', 'Name', 'Score', 'Genres', 'Type', 'Episodes', 'Synopsis', 'Image URL']]
    
    # Combine with user's list_film
    if list_film:
        user_df = pd.DataFrame(list_film)
        df = pd.concat([df, user_df], ignore_index=True)
    
    # Preprocess Genres  
    df['Genres'] = df['Genres'].fillna('').astype(str)
    
    # Ensure Genres are strings and lowercase
    df['Genres'] = df['Genres'].apply(lambda x: x.lower() if isinstance(x, str) else str(x).lower())
    
    # Create CountVectorizer for Genres  
    cv_genre = CountVectorizer(
        tokenizer=lambda x: x.split(','),
        lowercase=True
    )  
    
    # Handle potential empty list
    genres = df['Genres'].tolist()
    if not genres:
        genres = ['']
    
    genre_matrix = cv_genre.fit_transform(genres)  
    genre_names = cv_genre.get_feature_names_out()  
    
    # Create Genre DataFrame  
    genre_df = pd.DataFrame(  
        genre_matrix.toarray(),   
        columns=genre_names,   
        index=df['Name']  
    ).reset_index()  
    
    # Create CountVectorizer for Types  
    cv_type = CountVectorizer(
        tokenizer=lambda x: x.split(','),
        lowercase=True
    )  
    type_matrix = cv_type.fit_transform(df['Type'].fillna('').astype(str))  
    type_names = cv_type.get_feature_names_out()  
    
    # Create Type DataFrame  
    type_df = pd.DataFrame(  
        type_matrix.toarray(),   
        columns=type_names,   
        index=df['Name']  
    ).reset_index()  
    
    # Merge genre and type dataframes  
    genre_type_df = genre_df.merge(type_df)  
    
    # Compute cosine similarity matrix  
    genre_type_cosine = genre_type_df.drop(columns='Name')  
    genre_type_cosine_matrix = cosine_similarity(genre_type_cosine)  
    
    return df, genre_type_df, genre_type_cosine_matrix  

def main():
    # Simulate a username (in a real app, this would come from authentication)
    username = "default_user"
    
    # Sidebar Navigation
    menu = st.sidebar.radio("Navigation", 
        ["Anime Recommender", "My Watchlist", "Update Watchlist", "Add to Watchlist"]
    )

    global list_film
    list_film = load_watchlist(username)
    
    # Display current watchlist on all pages
    display_list_film()
    
    if menu == "Anime Recommender":
        # Existing recommendation system code
        st.title('🎬 Anime Recommendation System')
        st.markdown("""
        ### Discover Your Next Favorite Anime!
        Click the small arrow on the top left to search for an anime and get personalized recommendations based on genres and type.
        """)
        
        # Load data  
        df, genre_type_df, genre_type_cosine_matrix = load_data()  
        
       # Sidebar Configuration
        st.sidebar.header('🔍 Anime Search', divider='rainbow')
        st.sidebar.markdown("""
        *Data sourced from [Kaggle Anime Dataset 2023](https://www.kaggle.com/datasets/dsfelix/animes-dataset-2023/data)*
        """)
        
        # Search box with autocomplete  
        anime_names = sorted(df['Name'].unique())  
        selected_anime = st.sidebar.selectbox(  
            '**Select an Anime**',   
            options=anime_names,
            index=None,
            placeholder="Type to search..."
        )  
        
        # Recommendation Filters  
        st.sidebar.header('🎯 Recommendation Filters', divider='blue')  
        
        # Number of Recommendations Filter  
        recommendation_count = st.sidebar.selectbox(  
            '**Number of Recommendations**',  
            options=[10, 15, 20, 25, 50],
            help="Choose how many anime recommendations you want to see"
        )  
        
        # Type Filter  
        unique_types = ['All'] + list(df['Type'].unique())  
        selected_type = st.sidebar.selectbox(  
            '**Filter by Anime Type**',  
            options=unique_types,
            help="Filter recommendations by anime type"
        )  
        
        # Additional filters  
        st.sidebar.header('⭐ Score Filter', divider='green')  
        min_score = st.sidebar.slider(
            '**Minimum Score**', 
            0.0, 10.0, 6.0, 0.1,
            help="Filter recommendations by minimum rating"
        )  
        
        # Recommendation button  
        if st.sidebar.button('🚀 Get Recommendations', type='primary'):  
            # Validate anime selection
            if selected_anime is None:
                st.error("❌ Please select an anime first!")
            else:
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
                    
                    st.header(f'🌟 Recommendations for {selected_anime}')
                    
                    # Display top recommendations with detailed information
                    for index, row in recommendations.iterrows():
                        st.markdown(f"""
                        <div class='recommendation-card'>
                            <h3>{row['Name']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create columns for image and details
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            # Display image
                            st.image(row['Image URL'], use_container_width=True, caption='Anime Poster')
                        
                        with col2:
                            # Display details with improved formatting
                            st.markdown(f"""
                            **📺 Type:** {row['Type']}
                            
                            **⭐ MAL Score:** {row['Score']}/10
                            
                            **🔍 Compatibility Score:** {row['Compatibility Score']}%
                            
                            **🏷️ Genres:** {row['Genres']}
                            """)
                        
                        # Display synopsis
                        st.markdown(f"""
                        **📝 Synopsis:**
                        *{row['Synopsis']}*
                        """)
                        
                        # Add a separator
                        st.markdown("---")
                    
                    # Display full recommendations dataframe
                    st.subheader('📋 Full Recommendations')
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
                    st.error('🤷‍♀️ No recommendations found!')
    
    elif menu == "My Watchlist":
        display_watchlist(username)
    
    elif menu == "Update Watchlist":
        update_watchlist(username)
    
    elif menu == "Add to Watchlist":
        add_to_watchlist(username)

if __name__ == '__main__':  
    main()