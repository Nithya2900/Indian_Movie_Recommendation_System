Movie Mate - Indian Movie Recommendation System
Table of Contents
Introduction
Objectives
Features
Technologies Used
Dataset
Usage
How It Works
Future Enhancements
Contributors
License

Introduction
Movie Mate is a movie recommendation system that leverages a dataset of Indian movies to provide personalized recommendations based on user input. This project implements content-based filtering techniques to recommend movies similar to those a user has previously watched or selected.
The project aims to make it easier for users to discover Indian cinema, spanning various genres, directors, and actors. It provides a tailored movie-watching experience by recommending films that match the user's preferences.

Objectives
Build a recommendation system focusing on Indian cinema.
Implement content-based filtering to recommend movies based on features like genre, director, and actors.
Provide a simple and user-friendly interface for users to input their movie preferences and receive relevant recommendations.

Features
Personalized Recommendations: Based on content features like genre, director, actors, etc.
Extensive Movie Database: Indian movie dataset with detailed information.
User-Friendly Interface: Simple form to input the movie title and receive recommendations instantly.

Technologies Used
Python: Core language for development.
Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
Scikit-learn: For cosine similarity computation.
Flask: For building the web application (if applicable).
CSV: For storing the dataset (recommendations.csv).

Dataset
The movie dataset used for this project was sourced from Kaggle. It contains Indian movies with the following features:
  Movie Title
  Genre
  Director
  Actors
  Writer
  Reviews
  Year of Release
  IMDB Ratings
  The dataset is stored in a CSV file, recommendations.csv.

Usage
Run the Recommendation System:
Input:
Enter the title of a movie you enjoyed.
The system will provide a list of 10 recommended movies based on your input.
Output:
A list of similar movies will be displayed in the terminal or web interface, based on the content features (genre, director, etc.).

How It Works
Data Preprocessing: The dataset is loaded, cleaned, and preprocessed. Features such as genre, director, actors, and description are split into lists and converted to lowercase for uniformity.
Cosine Similarity: The model calculates the similarity between movies using the cosine similarity technique based on features like genre and director. When a user inputs a movie title, the system searches for the most similar movies based on these attributes.
Movie Recommendations: After calculating the similarity, the system outputs the top 10 most similar movies to the user's input.

Future Enhancements
Add collaborative filtering methods for more personalized recommendations.
Incorporate additional features such as user reviews and ratings.
Improve the user interface with more interactive elements.
Expand the dataset to include movies from more Indian languages and regions.

Contributors
Nithya - Project Developer
Prashuna - Project Developer

License
This project is licensed under the MIT License - see the LICENSE file for details.












