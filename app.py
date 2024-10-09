import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, render_template

# Load your data from the specified path
data_recsys = pd.read_csv("C:/Users/hp15s/Desktop/movierec/data/movies.csv")

# Download stopwords if not already downloaded
import nltk
nltk.download('stopwords')

# Initialize stopwords and stemmer
listStopwords = set(stopwords.words('english'))
ps = PorterStemmer()

# Preprocess function
def preprocess_data(data_recsys):
    data_recsys['description'] = data_recsys['description'].fillna('').astype('str').str.lower()
    data_recsys['description'] = data_recsys['description'].str.translate(str.maketrans('', '', string.punctuation))
    
    filtered_descriptions = []

    for desc in data_recsys['description']:
        filtered = []
        for word in desc.split():
            if word not in listStopwords:
                word_stemmed = ps.stem(word)
                filtered.append(word_stemmed)
        filtered_descriptions.append(' '.join(filtered))

    data_recsys['description'] = filtered_descriptions

# Call the function to preprocess data
preprocess_data(data_recsys)

# Set index for the DataFrame
data_recsys.set_index('original_title', inplace=True)

# Prepare the final content for similarity calculation
data_recsys['final_content'] = ''
for i, text in data_recsys.iterrows():
    words = ''
    for col in data_recsys.columns:
        if isinstance(text[col], str):
            words += text[col] + ' '
    data_recsys.loc[i, 'final_content'] = words.strip()  # Use .loc to avoid SettingWithCopyWarning

# Vectorize the final content
count = CountVectorizer()
count_matrix = count.fit_transform(data_recsys['final_content']).astype(np.uint8)

# Calculate cosine similarity
def similarity_cosine(start, end):
    if end > count_matrix.shape[0]:
        end = count_matrix.shape[0]
    return cosine_similarity(X=count_matrix[start:end], Y=count_matrix)

# Create a Series for movie indices
index_movies = pd.Series(data_recsys.index)

# Movies Recommendation function
def get_movies(title, cosine_sim):
    recommended_movies = []
    index_movie_input = index_movies[index_movies == title].index[0]
    score_movies = pd.Series(cosine_sim[index_movie_input]).sort_values(ascending=False)
    top_10_index_movies = list(score_movies.iloc[1:11].index)
    for i in top_10_index_movies:
        recommended_movies.append(data_recsys.index[i] + ' (' + str(data_recsys['year'].iloc[i]) + ')')
    return recommended_movies

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    cosine_sim = cosine_similarity(count_matrix)
    recommendations = get_movies(movie_title, cosine_sim)
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)




