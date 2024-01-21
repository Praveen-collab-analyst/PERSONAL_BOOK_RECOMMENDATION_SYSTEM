#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


# In[3]:


df = pd.read_csv('filtered_dataset.csv')


# In[4]:


reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'Book-Rating']], reader)


# In[5]:


trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# In[6]:


model = SVD()
model.fit(trainset)


# In[7]:


predictions = model.test(testset)
rmse = accuracy.rmse(predictions)


# In[13]:


st.title('Book Recommendation System')


# In[14]:


book_isbn = st.text_input('Enter a book ISBN to get recommendations:', '316666343')


# In[12]:


if st.button('Get Recommendations'):
    try:
        
        book_inner_id = model.trainset.to_inner_iid(book_isbn)
        recommendations = model.get_neighbors(book_inner_id, k=5)
        recommended_books = [model.trainset.to_raw_iid(inner_id) for inner_id in recommendations]
        st.subheader('Top 5 Recommendations:')
        for i, book in enumerate(recommended_books):
            st.write(f"{i+1}. {df[df['ISBN'] == book]['Book-Title'].values[0]} by {df[df['ISBN'] == book]['Book-Author'].values[0]}")
    except ValueError:
        st.error('Invalid Book ISBN. Please enter a valid ISBN.')

st.sidebar.subheader('Model Evaluation')
st.sidebar.text(f'RMSE: {rmse:.4f}')

st.sidebar.subheader('Unique Book Titles')
unique_titles = df['Book-Title'].unique()
st.sidebar.write(unique_titles[:10])  


# In[ ]:




