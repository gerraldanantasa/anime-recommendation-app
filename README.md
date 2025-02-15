# Anime Recommendation System 🎬

This is a Streamlit web application that provides personalized anime recommendations based on user preferences. It allows users to search for an anime, get recommendations, manage their watchlist, and track their watching progress.

## Features ✨

- Search for an anime from a comprehensive dataset
- Get personalized recommendations based on the selected anime's genres and type
- Filter recommendations by minimum score and anime type
- Add anime to a personal watchlist
- Update watchlist entries with episodes watched, status, and personal score
- View detailed information about each recommended anime, including synopsis and poster image
- Track watching progress and view watchlist statistics

## Dataset 📊

The anime dataset used in this application is sourced from [Kaggle Anime Dataset 2023](https://www.kaggle.com/datasets/dsfelix/animes-dataset-2023/data). It contains information about various anime, including their names, genres, types, scores, and synopses.

## Installation 🛠️

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/anime-recommendation-system.git
   cd anime-recommendation-system

2. Install the required dependencies:
    ```bash
    Copypip install -r requirements.txt

3. Run the Streamlit application:
  ``bash
  Copystreamlit run app.py

4. Open the application in your web browser at http://localhost:8501.

## Usage 🚀

1. On the sidebar, select an anime from the dropdown menu or type to search for a specific anime.
2. Adjust the recommendation filters, such as the number of recommendations, minimum score, and anime type, according to your preferences.
3. Click the "Get Recommendations" button to generate personalized recommendations based on the selected anime.
4. Explore the recommended anime, view their details, and add them to your watchlist if interested.
5. Navigate to the "My Watchlist" page to view your watchlist, update anime entries, and track your watching progress.
6. Use the "Update Watchlist" and "Add to Watchlist" pages to manage your watchlist entries.

## Technologies Used 🖥️

- Python
- Streamlit
- Pandas
- scikit-learn
- PIL (Python Imaging Library)

## Contributing 🤝
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgements 🙏

The anime dataset used in this application is sourced from [Kaggle Anime Dataset 2023](https://www.kaggle.com/datasets/dsfelix/animes-dataset-2023/data)
Special thanks to the open-source community for their invaluable contributions.
