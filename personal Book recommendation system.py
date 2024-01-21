#!/usr/bin/env python
# coding: utf-8

# In[188]:


import pandas as pd # for manipulation of  tabular data
import numpy as np # for numeric python 
from collections import Counter

# For data visualization
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(8,5),'figure.dpi':100}) # for setting the figure size.
import seaborn as sns # for visualization 
import random # to get random sample or data

# For Model building
import scipy
import math
from sklearn.metrics.pairwise import cosine_similarity # importing consine_similarity score from metrics module of seaborn lib.
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors # importing NearestNeighbors form neighbors module.
from sklearn.model_selection import train_test_split # importing train_test_split from model_preprocessing from sklearn module.
from scipy.sparse.linalg import svds 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing # for preprocessing

# Ignoring stopwords (words with no semantics) from English
import nltk
from nltk.corpus import stopwords # for handling stopwords in dataset.
from sklearn.preprocessing import normalize
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer # importing TfidfVectorizer from feature extraction
from sklearn.model_selection import train_test_split

# This is to supress the warning messages (if any) generated in our code
import warnings
warnings.filterwarnings('ignore') # for ignoring the warnings


# In[4]:


book_data = pd.read_csv("/Users/nick/Desktop/Dataset (2)/Books.csv")
book_data.head() 


# In[5]:


print("Columns: ", list(book_data.columns))


# In[6]:


book_data.info()


# In[7]:


book_data.shape


# In[8]:


book_data.isnull().sum()


# In[9]:


book_data.drop(['Image-URL-L'], axis= 1, inplace= True)


# In[10]:


book_data.isnull().sum()


# In[11]:


book_data.loc[(book_data['Book-Author'].isnull()),: ]


# In[12]:


book_data.loc[(book_data['Publisher'].isnull()),: ]


# In[13]:


book_data.loc[(book_data['ISBN'] == '193169656X'),'Publisher'] = 'No Mention'
book_data.loc[(book_data['ISBN'] == '1931696993'),'Publisher'] = 'No Mention'


# In[14]:


book_data[book_data['Publisher'] == 'No Mention']


# In[15]:


book_data['Year-Of-Publication'].unique()


# In[16]:


def replace_df_value(df, idx, col_name, val):
    df.loc[idx, col_name] = val
    return df


# In[17]:


replace_df_value(book_data, 209538, 'Book-Author', 'Michael Teitelbaum')
replace_df_value(book_data, 209538, 'Year-Of-Publication', 2000)
replace_df_value(book_data, 221678, 'Publisher', 'DK Publishing Inc')

replace_df_value(book_data, 221678, 'Book-Author', 'James Buckley')
replace_df_value(book_data, 221678, 'Year-Of-Publication', 2000)
replace_df_value(book_data, 221678, 'Publisher', 'DK Publishing Inc')

replace_df_value(book_data, 220731, 'Book-Author', 'Jean-Marie Gustave Le ClÃ?Â©zio')
replace_df_value(book_data, 220731, 'Year-Of-Publication', 2003)
replace_df_value(book_data, 220731, 'Publisher', 'Gallimard')


# In[18]:


book_data.loc[221678]


# In[19]:


book_data.loc[209538]


# In[20]:


book_data.loc[220731]


# In[21]:


book_data['Year-Of-Publication'].unique()


# In[22]:


book_data.isnull().sum()


# In[23]:


book_data.loc[(book_data['Book-Author'].isnull()),: ]


# In[24]:


book_data.loc[187689]


# In[25]:


book_data.loc[(book_data['ISBN'] == '9627982032'),'Book-Author'] = 'David Tait'


# In[26]:


book_data.loc[187689]


# In[27]:


book_data.isnull().sum()


# In[30]:


users_data = pd.read_csv("/Users/nick/Desktop/Dataset (2)/Users.csv")
users_data.head()


# In[31]:


users_data.isnull().sum()


# In[32]:


users_data = pd.read_csv("/Users/nick/Desktop/Dataset (2)/Users.csv")
country_data = pd.DataFrame(users_data)
country_data['Country'] = country_data['Location'].str.extract(r', ([^,]+)$')
print(country_data)


# In[33]:


users_data.isnull().sum()


# In[34]:


users_data['Country'].unique()


# In[35]:


users_data['Country'] = users_data['Country'].fillna('Unknown')


# In[36]:


unique_value = users_data['Country']
counts_value = dict(Counter(unique_value))
counts_list = pd.DataFrame(list(counts_value.items()), columns=['Unique_Value', 'Count'])
print(counts_list)


# In[37]:


users_data.isnull().sum()


# In[38]:


users_data['Age'].unique()


# In[39]:


users_data['Age'].isnull().sum()


# In[40]:


users_data.isnull().sum()


# In[41]:


ratings_data = pd.read_csv("/Users/nick/Desktop/Dataset (2)/Ratings.csv")
ratings_data


# In[42]:


ratings_data.isnull().sum()


# In[43]:


ratings_data['Book-Rating'].unique()


# In[44]:


unique_ratings = pd.merge(book_data, ratings_data, on='ISBN', how='inner')


# In[45]:


print(ratings_data.shape)
print(unique_ratings.shape)


# In[46]:


unique_ratings['Book-Rating'].unique()


# In[47]:


# Merging the data frames

merged_data1=pd.merge(users_data,ratings_data,on='User-ID') # merging df_users with df_ratings based on User-ID
merged_dataset=pd.merge(merged_data1,book_data,on='ISBN') # merging  merged_df with df_books based on ISBN


# In[48]:


merged_dataset.head() # showing top 5 records of final dataframe


# In[49]:


merged_dataset.columns


# In[50]:


merged_dataset.info() # basic information about the final datafram after merging


# In[51]:


# Size of the merged dataset
merged_dataset.shape


# In[52]:


# Total duplicates present in the data

merged_dataset.duplicated().sum()


# In[53]:


# Check for missing values

merged_dataset.isnull().sum()


# In[54]:


merged_dataset['Year-Of-Publication'] = pd.to_numeric(merged_dataset['Year-Of-Publication'], errors='coerce')


# # Exploratory data analysis

# In[55]:


# Box plot for age

sns.boxplot(merged_dataset['Age']);


# It can be clearly seen that a lot of outliers are present in age column.

# In[56]:


# Outlier data became NaN

merged_dataset.loc[(merged_dataset.Age > 100) | (merged_dataset.Age < 5), 'Age'] = np.nan


# In[57]:


# Null values in age column

nulls = sum(merged_dataset['Age'].isnull()) # checking the missing value in Age
print(nulls)


# In[58]:


# Imputing null values
median = merged_dataset['Age'].median() # finding the median of Age column
std = merged_dataset['Age'].std() # Standard Deviation of Age
print(median)
print(std)


# In[59]:


merged_dataset['Age'].fillna(median, inplace=True)
print()


# In[60]:


# Check for missing values

merged_dataset.isnull().sum()


# In[63]:


merged_dataset.shape # checking shape of final dataframe.


# In[64]:


merged_dataset.to_csv('merged_dataset.csv', index=False)


# In[58]:


# Distribution of age after removing outliers and fixing missing values

x = merged_dataset.Age.value_counts().sort_index() # counting the values of Age
sns.histplot(merged_dataset['Age'], bins=10, kde=True, color='skyblue')
plt.xlabel('Age')
plt.ylabel('x')
plt.title('Distribution of Age')
plt.show()


# It's observable that maximum number of users were of the age in between 20 to 60.

# In[59]:


# showing the distribution of Year of Publication.

sns.distplot(merged_dataset[merged_dataset['Year-Of-Publication']>1800]['Year-Of-Publication'],color='purple',bins=50);


# There was an exponential increase in book publication after the year 1950.

# In[60]:


# ploatting the count of top 30 books using coutplot.

sns.countplot(y='Book-Author',data=book_data,order=pd.value_counts(book_data['Book-Author']).iloc[:30].index, palette='pastel')
plt.title("Authors with Most Number of Books", fontweight='bold');


# In[61]:


# Counting the top the publisher using countplot of seaborn 

sns.countplot(y='Publisher',data=book_data,order=pd.value_counts(book_data['Publisher']).iloc[:30].index)
plt.title('Top 30 Publishers', fontweight='bold');


# Publisher with highest number of books published was Harlequin followed by Solhoutte and Pocket.

# In[62]:


# Pie Graph of top five countires.

palette_color = sns.color_palette('pastel')
explode = (0.1, 0, 0, 0, 0)
merged_dataset.Country.value_counts().iloc[:5].plot(kind='pie', colors=palette_color, autopct='%.0f%%', explode=explode, shadow=True)
plt.title('Top 5 countries', fontweight='bold');


# In[63]:


# Average Book ratings with respect to top 30 books using catplot

book_rating = merged_dataset.groupby(['Book-Title','Book-Author'])['Book-Rating'].agg(['count','mean']).sort_values(by='mean', ascending=False).reset_index()
sns.catplot(x='mean', y='Book-Title', data=book_rating[book_rating['count']>500][:30], kind='bar', palette = 'Paired',hue='Book-Author' )
plt.xlabel('Average Ratings')
plt.ylabel('Books')
plt.title('Most Famous Books', fontweight='bold');


# Harry Potter authored by J K Rowling had got the best average ratings followed by To Kill a Mockingbird and The Da Vinci Code.

# In[64]:


# barplot of book_rating with respect to its index

sns.barplot(x = merged_dataset['Book-Rating'].value_counts().index,y = merged_dataset['Book-Rating'].value_counts().values,
            palette = 'magma').set(title="Ratings Distribution", xlabel = "Rating",ylabel = 'Number of books')
plt.show();


# As we can see that more than 6 lakh have 0 rating

# In[65]:


# coutplot of book_ratings

sns.countplot(x="Book-Rating", palette = 'Paired', data = merged_dataset)
plt.title("Ratings", fontweight='bold');


# As we can see that 8 is the most rated book if we exclude the rating o

# In[66]:


# Top 15 highst readers from countries
# Count the number of users from each country
country_counts = merged_dataset['Country'].value_counts()

# Select the top 15 countries
top_countries = country_counts.head(15)

# Plotting a bar chart
plt.figure(figsize=(12, 6))
top_countries.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Top 15 Countries with the Highest Number of Readers')
plt.xlabel('Country')
plt.ylabel('Number of Readers')
plt.xticks(rotation=45, ha='right')  # Rotate country names for better readability
plt.show()


# Countplot of explicit ratings indicates that higher ratings are more common amongst users and rating 8 has been rated highest number of times.

# # Collaborative filtering models

# ### **Item Based**

# Collaborative filtering methods Collaborative methods for recommender systems are methods that are based solely on the past interactions recorded between users and items in order to produce new recommendations. These interactions are stored in the so-called “user-item interactions matrix”.

# * Every user's rating at facevalue can't be considered because if the user is a **novice reader** with only an experience of reading a couple of books, his/her ratings might not be much relevant for finding similarity among books.
# * Therefore as a general rule of thumb let's consider only those Users who have rated atleast **50** books and only those books which have got atleast **50** ratings.

# In[68]:


# Checking the shape of merged dataframe
merged_dataset.shape


# In[69]:


merged_dataset.columns


# In[84]:


x = merged_dataset.groupby('User-ID').count()['Book-Rating'] > 50


# In[85]:


x[x]


# In[87]:


merged_dataset['User-ID'].isin(x[x].index)


# In[88]:


# taking explitcit rating_df means Taking where book rating is not equal to zero
merged_dataset = merged_dataset[merged_dataset['Book-Rating']!=0]


# In[89]:


print("Shape of merged dataframe Now : ",merged_dataset.shape)


# In[90]:


# Applying constraint on user id using it's count 

x = merged_dataset.groupby('User-ID').count()['Book-Rating'] >50

filtered_dataset = merged_dataset[merged_dataset['User-ID'].isin(x[x].index)]


# In[91]:


# Applying constraint on number of ratings

y = merged_dataset.groupby('Book-Title').count()['Book-Rating'] >50
filtered_dataset = filtered_dataset[filtered_dataset['Book-Title'].isin(y[y].index)]


# In[92]:


filtered_dataset.shape


# In[93]:


# head of filtered dataframe
filtered_dataset.head()


# In[94]:


y1 = filtered_dataset.groupby('Book-Title').count()['Book-Rating']>= 10
famous_books = y1[y1].index


# In[95]:


famous_books


# In[96]:


filtered_dataset = filtered_dataset[filtered_dataset['Book-Title'].isin(famous_books) ]
filtered_dataset


# In[97]:


filtered_dataset['User-ID'].nunique()


# In[99]:


filtered_dataset.to_csv('filtered_dataset.csv', index=False)


# In[100]:


pt = filtered_dataset.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating').fillna(0) # filling nan with 0


# In[101]:


pt # showing the Pivot tabel


# In[103]:


# Create an object of cosine similarity

similarity_scores = cosine_similarity(pt)


# In[104]:


# Matrix size 

similarity_scores.shape


# In[105]:


for i,j in enumerate([1,2,3]):
  print(f"Index : {i} value {j}")


# In[109]:


def recommend_book(book_name):
  """
  Description: It takes a book name and return data frame with similarity score 
  Function: recommend-book
  Argument: book-name
  Return type : dataframe
  """
  index = np.where(pt.index == book_name)[0][0] # finding index of same book
  similar_books = sorted(list(enumerate(similarity_scores[index])), key = lambda x:x[1], reverse = True)[1:11] # creating the list tuple of index with respect to similarity score
  
  # print(similar_books)
  
  print("\n----------------Recommended books-----------------\n")
  for i in similar_books:
    print(pt.index[i[0]]) 
  print("\n.....................................................\n")  
  return find_similarity_score(similar_books,pt) 


# In[110]:


def find_similarity_score(similarity_scores,pivot_table):

  """
  Description: It takes similarity_Score and pivot table and return dataframe.
  function : find_similarity_Score
  Output : dataframe
  Argument  similarity_score and pivot table
  """
  list_book = []
  list_sim = []
  for i in similarity_scores:
    index_ = i[0]
    sim_ = i[1]
    list_sim.append(sim_)
    # list_book.append(pivot_table[pivot_table.index == index_]['Book-Title'][index_])
    list_book.append(pivot_table.iloc[index_,:].name)
    
    df = pd.DataFrame(list(zip(list_book, list_sim)),
               columns =['Book', 'Similarity'])
  # df =pd.DataFrame([list_book, list_sim], columns = ["Book",'Similarity_Score'])
  return df


# In[111]:


recommend_book('The Notebook')


# USER-BASED FILTERING

# In[117]:


filtered_dataset


# In[119]:


pt2 = filtered_dataset.pivot_table(index='User-ID',columns='Book-Title',values='Book-Rating')


# In[120]:


pt2


# In[125]:


pd.DataFrame(pairwise_distances(pt2, metric='cosine'))


# In[126]:


sim2 = 1- pairwise_distances(pt2, metric='cosine')
pd.DataFrame(sim2)


# In[131]:


similar_user = pd.DataFrame(sim2)


# In[132]:


similar_user.index = filtered_dataset['User-ID'].unique()
similar_user.columns = filtered_dataset['User-ID'].unique()


# In[133]:


similar_user


# In[134]:


similar_user.idxmax()


# In[136]:


filtered_dataset[(filtered_dataset['User-ID'] == 72352) | (filtered_dataset['User-ID'] == 132492)]


# In[138]:


similar_user_score = cosine_similarity(pt2)


# In[139]:


similar_user_score[0]


# In[142]:


def recommendations_for_user(user_id):
    print('\n Recommended Books for User_id',(user_id),':\n')
    recom = list(similarity_user.sort_values([user_id], ascending= False).head().index)[1:11]
    books_list = []
    for i in recom:
        books_list = books_list + list(filtered_dataset[filtered_dataset['User-ID']==i]['Book-Title'])
    return set(books_list)-set(filtered_dataset[filtered_dataset['User-ID']==user_id]['Book-Title'])


# In[143]:


recommendations_for_user(6242)


# In[210]:


from sklearn import model_selection
train_data, test_data = model_selection.train_test_split(filtered_dataset, test_size=0.20)


# In[211]:


print(f'Training set lengths: {len(train_data)}')
print(f'Testing set lengths: {len(test_data)}')
print(f'Test set is {(len(test_data)/(len(train_data)+len(test_data))*100):.0f}% of the full dataset.')


# In[212]:


u_unique_train = train_data['User-ID'].unique()
train_data_user2idx = {o:i for i, o in enumerate(u_unique_train)}

# Get int mapping for isbn in train dataset
i_unique_train = train_data['ISBN'].unique()
train_data_book2idx = {o:i for i, o in enumerate(i_unique_train)}


# In[213]:


u_unique_test = test_data['User-ID'].unique()
test_data_user2idx = {o:i for i, o in enumerate(u_unique_test)}

# Get int mapping for isbn in test dataset
i_unique_test = test_data['ISBN'].unique()
test_data_book2idx = {o:i for i, o in enumerate(i_unique_test)}


# In[214]:


train_data['u_unique'] = train_data['User-ID'].map(train_data_user2idx)
train_data['i_unique'] = train_data['ISBN'].map(train_data_book2idx)

# testing set
test_data['u_unique'] = test_data['User-ID'].map(test_data_user2idx)
test_data['i_unique'] = test_data['ISBN'].map(test_data_book2idx)

# Convert back to three feature of dataframe
train_data = train_data[['u_unique', 'i_unique', 'Book-Rating']]
test_data = test_data[['u_unique', 'i_unique', 'Book-Rating']]


# In[215]:


train_data.sample(2)


# In[216]:


test_data.sample(2)


# In[217]:


# first I'll create an empty matrix of users books and then I'll add the appropriate values to the matrix by extracting them from the dataset
n_users = train_data['u_unique'].nunique()
n_books = train_data['i_unique'].nunique()

train_matrix = np.zeros((n_users, n_books))

for entry in train_data.itertuples():
    train_matrix[entry[1]-1, entry[2]-1] = entry[3]


# In[218]:


train_matrix.shape


# In[219]:


n_users = test_data['u_unique'].nunique()
n_books = test_data['i_unique'].nunique()

test_matrix = np.zeros((n_users, n_books))

for entry in test_data.itertuples():
    test_matrix[entry[1]-1, entry[2]-1] = entry[3]


# In[220]:


test_matrix.shape


# # Cosine Similarity Based Recommendation System

# In[221]:


train_matrix_small = train_matrix[:1000, :1000]
test_matrix_small = test_matrix[:1000, :1000]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_matrix_small, metric='cosine')
item_similarity = pairwise_distances(train_matrix_small.T, metric='cosine')


# In[222]:


def predict_books(ratings, similarity, type='user'): # default type is 'user'
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)

        # Use np.newaxis so that mean_user_rating has the same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# In[223]:


item_prediction = predict_books(train_matrix_small, item_similarity , type='item')
user_prediction = predict_books(train_matrix_small, user_similarity , type='user')


# In[224]:


# Evaluation metric by mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, test_matrix):
    prediction = prediction[test_matrix.nonzero()].flatten()
    test_matrix = test_matrix[test_matrix.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, test_matrix))

print(f'Item-based CF RMSE: {rmse(item_prediction, test_matrix_small)}')
print(f'User-based CF RMSE: {rmse(user_prediction, test_matrix_small)}')


# In[225]:


filtered_dataset


# In[226]:


get_ipython().system('pip install scikit-surprise')


# In[227]:


from surprise import Reader, Dataset


# In[229]:


reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(filtered_dataset[['User-ID','Book-Title','Book-Rating']], reader)


# In[230]:


from surprise import SVD, model_selection, accuracy
model = SVD()

# Train on books dataset
get_ipython().run_line_magic('time', "model_selection.cross_validate(model, data, measures=['RMSE'], cv=5, verbose=True)")


# In[231]:


# to test result let's take an user-id and item-id to test our model.
uid = 276744
iid = '038550120X'
pred = model.predict(uid, iid, verbose=True)


# In[233]:


print(f'The estimated rating for the book with ISBN code {pred.iid} from user #{pred.uid} is {pred.est:.2f}.\n')
actual_rtg= ratings_data[(ratings_data['User-ID']==pred.uid) &
                             (ratings_data['ISBN']==pred.iid)]['Book-Rating'].values[0]
print(f'The real rating given for this was {actual_rtg:.2f}.')


