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


# In[8]:


input_book_isbn = df[df['Book-Title'] == input_book_title]['ISBN'].values


# In[9]:


if len(input_book_isbn) > 0:
    input_book_ratings = [(trainset.to_inner_iid(isbn), svd_model.predict(trainset.to_inner_uid(6242), trainset.to_inner_iid(isbn)).est)
                          for isbn in input_book_isbn]

    input_book_ratings = sorted(input_book_ratings, key=lambda x: x[1], reverse=True)

    st.subheader("Top 5 Recommended Books:")

    # Display top 5 recommended books
    for i in range(min(5, len(input_book_ratings))):
        book_id, predicted_rating = input_book_ratings[i]
        book_title = df[df['ISBN'] == trainset.to_raw_iid(book_id)]['Book-Title'].values[0]
        book_author = df[df['ISBN'] == trainset.to_raw_iid(book_id)]['Book-Author'].values[0]
        st.write(f"{i + 1}. {book_title} by {book_author} (Predicted Rating: {predicted_rating:.2f})")

else:
    st.warning("Book not found. Please enter a valid book title.")


# In[ ]:





# In[ ]:




