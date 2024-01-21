#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

df = pd.read_csv('filtered_dataset.csv')

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'Book-Rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

st.title('Book Recommendation System')



# User input for Book ISBN
book_isbn = st.text_input('Enter a book ISBN to get recommendations:', '0316666343')

# Get recommendations for the entered book ISBN
if st.button('Get Recommendations'):
    try:
        # Get inner id for the entered book ISBN
        book_inner_id = model.trainset.to_inner_iid(book_isbn)

        # Get top N recommendations
        recommendations = model.get_neighbors(book_inner_id, k=5)

        # Convert inner ids back to ISBNs
        recommended_books = [model.trainset.to_raw_iid(inner_id) for inner_id in recommendations]

        # Display recommendations
        st.subheader('Top 5 Recommendations:')
        for i, book in enumerate(recommended_books):
            st.write(f"{i+1}. {df[df['ISBN'] == book]['Book-Title'].values[0]} by {df[df['ISBN'] == book]['Book-Author'].values[0]}")

    except ValueError:
        st.error('Invalid Book ISBN. Please enter a valid ISBN.')

