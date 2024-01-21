import streamlit as st
import pandas as pd
import numpy as np
import math  
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('filtered_dataset.csv')


def smooth_user_preference(x):
    return math.log(1 + x, 2)

ratings_full_df = df.groupby(['Book-Title', 'User-ID'])['Book-Rating'].sum().apply(smooth_user_preference).reset_index()

# Train-test split
ratings_train_df, ratings_test_df = train_test_split(ratings_full_df, test_size=0.2, stratify=ratings_full_df['User-ID'], random_state=42)

ratings_train_df['Book-Title'] = le.transform(ratings_train_df['Book-Title'])
ratings_test_df['Book-Title'] = le.transform(ratings_test_df['Book-Title'])

# SVD Matrix Factorization
NUMBER_OF_FACTORS_MF = 15
U, sigma, Vt = np.linalg.svd(ratings_full_df.pivot(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0), full_matrices=False)

# Create a recommendation DataFrame
cf_preds_df = pd.DataFrame(U.dot(np.diag(sigma)).dot(Vt), columns=ratings_full_df['Book-Title'].unique(), index=ratings_full_df['User-ID'].unique())

class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        sorted_user_predictions = self.cf_predictions_df.loc[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['Book-Title'].isin(items_to_ignore)].head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left', on='Book-Title')[['recStrength', 'Book-Title']]

        return recommendations_df

# Create a CFRecommender object
cf_recommender_model = CFRecommender(cf_preds_df, df)

# Streamlit app
st.title('Book Recommendation System')

# User input for Book Title
book_title = st.text_input('Enter a book title to get recommendations:', 'The Da Vinci Code')

# Get recommendations for the entered book title
if st.button('Get Recommendations'):
    try:

        # Get recommendations
        recommendations = cf_recommender_model.recommend_items(user_id=1, items_to_ignore=[], topn=5, verbose=True)

        # Display recommendations
        st.subheader('Top 5 Recommendations:')
        for i, row in recommendations.iterrows():
            st.write(f"{i + 1}. {row['Book-Title']} (Predicted Rating: {row['recStrength']:.2f})")

    except ValueError:
        st.error('Invalid Book Title. Please enter a valid title.')
