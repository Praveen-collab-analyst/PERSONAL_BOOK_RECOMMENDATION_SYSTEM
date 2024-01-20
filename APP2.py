#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split


# In[2]:


df = pd.read_csv('filtered_dataset.csv')


# In[3]:


reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['User-ID', 'Book-Title', 'Book-Rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# In[4]:


svd = SVD()
svd.fit(trainset)


# In[5]:


st.title('Book Recommendation App')


# In[6]:


book_title = st.text_input('Enter Book Title', 'The Catcher in the Rye')


# In[9]:


if st.button('Get Recommendations'):
    predictions = []
    for user_id in df['User-ID'].unique():
        pred = svd.predict(user_id, book_title)
        predictions.append({'User-ID': user_id, 'Prediction': pred.est})

    recommendations = pd.DataFrame(predictions).sort_values('Prediction', ascending=False)
    st.write(recommendations.head(10))

