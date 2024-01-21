#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import streamlit as st


# In[2]:


df = pd.read_csv('filtered_dataset.csv')


# In[3]:


reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'Book-Rating']], reader)


# In[4]:


trainset = data.build_full_trainset()


# In[5]:


svd_model = SVD()
svd_model.fit(trainset)


# In[6]:


st.title("Book Recommendation App")


# In[7]:


input_book_title = st.text_input("Enter a book title:", "The Da Vinci Code")  
book_ratings = df[df['Book-Title'] == input_book_title][['User-ID', 'ISBN', 'Book-Rating']]


# In[9]:


if not book_ratings.empty:
    svd_model.fit(trainset)

    user_ids = book_ratings['User-ID'].tolist()
    other_books = df[df['User-ID'].isin(user_ids) & (df['Book-Title'] != input_book_title)]['ISBN'].unique()
    predictions = [svd_model.predict(user_id, book) for user_id in user_ids for book in other_books]

    sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    st.subheader("Top 5 Recommended Books:")

    for i, prediction in enumerate(sorted_predictions[:5]):
        book_info = df[df['ISBN'] == prediction.iid][['Book-Title', 'Book-Author', 'Image-URL-S']].iloc[0]
        st.write(f"{i + 1}. {book_info['Book-Title']} by {book_info['Book-Author']} (Rating: {prediction.est:.2f})")
else:
    st.warning("Book not found. Please enter a valid book title.")

