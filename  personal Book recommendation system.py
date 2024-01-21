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


# In[234]:


# Create an object of csr matrix
# Sparse Matrix Representations 

df_matrix = csr_matrix(pt.values)


# In[235]:


# Building a KNN model with Cosine Similarity as the target metric for calculating the distances.

knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors=5)
knn.fit(pt) # training knn 


# In[236]:


def recommend(book, n_values=11):
  
  distances, indices = knn.kneighbors(pt.loc[book,:].values.reshape(1, -1), n_neighbors = n_values)
  dist = distances.flatten().tolist()
  books = []
  for i in range(1, len(indices.flatten())):
    books.append(pt.index[indices.flatten()[i]])
  
  data = list(zip(books,dist))
  df = pd.DataFrame(data,columns=['book','Distance'])
  return df


# In[237]:


recommend("Harry Potter and the Sorcerer's Stone (Book 1)")


# Collaborative filtering methods
# Collaborative methods for recommender systems are methods that are based solely on the past interactions recorded between users and items in order to produce new recommendations. These interactions are stored in the so-called “user-item interactions matrix”.

# **Implementing KNN**

# **Books which are rated by atleast 10 users**

# In[148]:


#Rating data with exclusion of Books with rating 0
ratings_data= ratings_data[ratings_data['Book-Rating'] != 0]


# In[149]:


#Merging dataframe rating and books on ISBN
df=pd.merge(ratings_data,book_data, on='ISBN')


# In[150]:


# Books interactionn count
books_interactions_count_df = df.groupby(['ISBN', 'User-ID']).size().groupby('ISBN').size()
print('# of books: %d' % len(books_interactions_count_df))

# Books with enough interactions
books_with_enough_interactions_df = books_interactions_count_df[books_interactions_count_df >= 10].reset_index()[['ISBN']]
print('# of books with at least 10 interactions: %d' % len(books_with_enough_interactions_df))
print(books_with_enough_interactions_df.head(5))


# **Users which have rated atleast 25 different books**

# In[151]:


# Users interactionn count
users_interactions_count_df = df.groupby(['User-ID', 'ISBN']).size().groupby('User-ID').size()
print('# of users: %d' % len(users_interactions_count_df))

# Users with enough interactions
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 25].reset_index()[['User-ID']]
print('# of users with at least 25 interactions: %d' % len(users_with_enough_interactions_df))
print(users_with_enough_interactions_df.head(5))


# In[152]:


#Users with enough interactions
print('# of interactions: %d' % len(df))
interactions_from_selected_users_df = df.merge(users_with_enough_interactions_df, 
               how = 'right',
               on = 'User-ID')
print('# of interactions from users with at least 25 interactions: %d' % len(interactions_from_selected_users_df))


# **Dataframe of Users and Books with enough interactions**

# In[153]:


#Users and Books with enough interactions
print('# of interactions: %d' % len(df))
interactions_from_selected_books_and_users_df= interactions_from_selected_users_df.merge(books_with_enough_interactions_df, on = 'ISBN')
print('# of interactions from users with at least 25 interactions and books with at least 10 interactions: %d' % len(interactions_from_selected_books_and_users_df))


# In[154]:


#interactions from selected books and users dataframe
interactions_from_selected_books_and_users_df.head(5)


# In[155]:


#Shape of interactions from selected books and users dataframe
interactions_from_selected_books_and_users_df.shape


# In[156]:


#aggregating all the interactions of users and applying log transformation to rating
import math
def smooth_user_preference(x):
    return math.log(1+x, 2)

interactions_full_df1 = interactions_from_selected_books_and_users_df.groupby(['User-ID', 'Book-Title'])['Book-Rating'].sum().apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df1))
interactions_full_df = interactions_from_selected_books_and_users_df.groupby(['User-ID', 'ISBN'])['Book-Rating'].sum().apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(5)


# In[157]:


#Creating a sparse pivot table.

df_user_item_matrix = interactions_full_df.pivot(index='ISBN',columns='User-ID',values='Book-Rating').fillna(0)
user_item_matrix_sparse = csr_matrix(df_user_item_matrix.values)
df_user_item_matrix1 = interactions_full_df1.pivot(index='User-ID',columns='Book-Title',values='Book-Rating').fillna(0)
df_user_item_matrix1=df_user_item_matrix1.transpose()
user_item_matrix_sparse1 = csr_matrix(df_user_item_matrix1.values)
user_item_matrix_sparse1=csr_matrix(df_user_item_matrix1.values)


# **Model Building**

# In[158]:


#Fitting Model
model = NearestNeighbors(n_neighbors=30, metric='cosine', algorithm='brute', n_jobs=-1)
 
model.fit(user_item_matrix_sparse1)


# **Recommendations for randomly selected book**

# In[159]:


#Recommendations for randomly selected book

query_index = np.random.choice(df_user_item_matrix1.shape[0])
distances, indices = model.kneighbors(df_user_item_matrix1.iloc[query_index, :].values.reshape((1, -1)), n_neighbors = 16)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for Book {0}:\n'.format(df_user_item_matrix1.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, df_user_item_matrix1.index[indices.flatten()[i]], distances.flatten()[i]))


# **Model building and recommendation for perticular book**

# In[160]:


#Model building and recommendation for perticular book
model = NearestNeighbors(n_neighbors=30, metric='cosine', algorithm='brute', n_jobs=-1)
 
model.fit(user_item_matrix_sparse)

index_to_book = dict()
 
df_titles_book = df.set_index('ISBN').loc[df_user_item_matrix.index]
 
count = 0
 
for index, row in df_titles_book.iterrows():
 
    index_to_book[count]=row['Book-Title']
 
    count +=1
 
 
def recommender(model, user_item_matrix_sparse, df_book, number_of_recommendations, book_index):
 
    main_title = index_to_book[book_index]
 
    dist, ind = model.kneighbors(user_item_matrix_sparse[book_index], n_neighbors=number_of_recommendations+1)
 
    dist = dist[0].tolist()
 
    ind = ind[0].tolist()
 
    titles = []
 
    for index in ind:
 
        titles.append(index_to_book[index])
 
    recommendations = list(zip(titles,dist))    
 
    # sort recommendations

    recommendations_sorted = sorted(recommendations, key = lambda x:x[1])
 
    # reverse recommendations, leaving out the first element 
 
    recommendations_sorted.reverse()
 
    recommendations_sorted = recommendations_sorted[:-1]
 
    print("Recommendations for Book {}: ".format(main_title))
 
    count = 0
 
    for (title, distance) in recommendations_sorted:
 
        count += 1
 
        print('{}. {}, recommendation score = {}'.format(count, title, round(distance,5)))
 
recommender(model, user_item_matrix_sparse, df, 10, 10)


# **SVD**

# In[161]:


from surprise import accuracy, Dataset, SVD
from surprise import Reader
from surprise import Dataset
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

import matplotlib.pyplot as plt


# In[162]:


minimum_rating = min(interactions_full_df['Book-Rating'].values)
 
maximum_rating = max(interactions_full_df['Book-Rating'].values)


# In[163]:


reader = Reader(rating_scale=(minimum_rating,maximum_rating))
 
data = Dataset.load_from_df(interactions_full_df[['User-ID', 'ISBN', 'Book-Rating']], reader) 


# **Train Test Split And Model Building**

# In[164]:


# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

# We'll use the famous SVD algorithm
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# Then compute MAE
accuracy.mae(predictions)


# In[165]:


interactions_full_df.head(1)


# In[166]:


user_id = '254'

isbn = '0060934700'

prediction = algo.predict(uid=user_id, iid=isbn)

print("Predicted rating of user with id {} for movie with id {}: {}".format(user_id, isbn, round(prediction.est,3)))


# Log transforamtion is applied to ratings

# In[167]:


# Predictions- actual and estimated
predictions


# **SVDpp**

# **Train Test Split And Model Building**

# In[168]:


# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

# We'll use the famous SVD algorithm
algo = SVDpp()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# Then compute MAE
accuracy.mae(predictions)


# In[169]:


interactions_full_df.head(1)


# In[170]:


user_id = '254'

isbn = '0060934700'

prediction = algo.predict(uid=user_id, iid=isbn)

print("Predicted rating of user with id {} for movie with id {}: {}".format(user_id, isbn, round(prediction.est,3)))


# In[171]:


# Predictions- actual and estimated
predictions


# **NMF**

# **Train Test Split And Model Building**

# In[172]:


# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

# We'll use the famous SVD algorithm
algo = NMF()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# Then compute MAE
accuracy.mae(predictions)


# In[173]:


interactions_full_df.head(1)


# In[174]:


user_id = '254'

isbn = '0060934700'

prediction = algo.predict(uid=user_id, iid=isbn)

print("Predicted rating of user with id {} for movie with id {}: {}".format(user_id, isbn, round(prediction.est,3)))


# In[175]:


# Predictions- actual and estimated
predictions


# **SlopeOne**

# **Train Test Split And Model Building**

# In[176]:


# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

# We'll use the famous SVD algorithm
algo = SlopeOne()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# Then compute MAE
accuracy.mae(predictions)


# In[177]:


interactions_full_df.head(1)


# In[178]:


user_id = '254'

isbn = '0060934700'

prediction = algo.predict(uid=user_id, iid=isbn)

print("Predicted rating of user with id {} for movie with id {}: {}".format(user_id, isbn, round(prediction.est,3)))


# In[179]:


# Predictions- actual and estimated
predictions


# **All Models Evaluation   (Cross Validation Used)**

# In[180]:


svd = cross_validate(SVD(), data, cv=5, n_jobs=-1, verbose=False)

svdpp = cross_validate(SVDpp(), data, cv=5, n_jobs=-1, verbose=False)

nmf = cross_validate(NMF(), data, cv=5, n_jobs=-1, verbose=False)

slope = cross_validate(SlopeOne(), data, cv=5, n_jobs=-1, verbose=False)


df_results = pd.DataFrame(columns=['Method', 'RMSE', 'MAE'])

df_results.loc[len(df_results)]=['SVD', round(svd['test_rmse'].mean(),5),round(svd['test_mae'].mean(),5)]

df_results.loc[len(df_results)]=['SVD++', round(svdpp['test_rmse'].mean(),5),round(svdpp['test_mae'].mean(),5)]

df_results.loc[len(df_results)]=['NMF', round(nmf['test_rmse'].mean(),5),round(nmf['test_mae'].mean(),5)]

df_results.loc[len(df_results)]=['SlopeOne', round(slope['test_rmse'].mean(),5),round(slope['test_mae'].mean(),5)]

display(df_results)


ax = df_results[['RMSE','MAE']].plot(kind='bar', figsize=(15,8))

ax.set_xticklabels(df_results['Method'].values)

ax.set_title('RMSE and MAE of different collaborative filtering algorithms')

plt.xticks(rotation=45)


plt.show();


# **Conclusion**

# *   Wild Animus is the best-selling book
# 
# 
# *   Author Agatha Christie, William Shakespeare and Stephen King wrote most of the books 
# 
# 
# 
# *   Harlequin publication published the most books
# 
# 
# *   More than 50% readers are from USA
# *   Book-Ratings are negatively distributed with median rating of 8.
# 
# * Root mean squared error of model **SVD** is 0.31 and mean absolute error is 0.21
# 
# * Root mean squared error of model **NMF** is 0.34 and mean absolute error is 0.24
# 
# * Root mean squared error of model **SlopeOne** is 0.39 and mean absolute error is 0.27
# 
# * **SVD++** is the **best recommendation model** with root mean squared error of 0.30 and mean absolute error of 0.20
# 
# 

# ### **User Based**

# Recommender systems have a problem known as user **cold-start**, in which it is hard to provide **personalized** recommendations for users with none or a very few number of consumed items, due to the **lack** of information to model their preferences. For this reason, we are keeping in the dataset only users with at least **100** interactions.

# In[181]:


users_ratings_count_df = merged_dataset.groupby(['Book-Title', 'User-ID']).size().groupby('User-ID').size()
print('Number of users: %d' % len(users_ratings_count_df))


users_with_enough_ratings_df = users_ratings_count_df[users_ratings_count_df >50].reset_index()[['User-ID']] # Users who rated more than 50 books
print('Number of users with at least 10 ratings: %d' % len(users_with_enough_ratings_df))


# In[182]:


print('Number of ratings : %d' % len(merged_dataset))
ratings_from_selected_users_df = merged_dataset.merge(users_with_enough_ratings_df, 
               how = 'right',
               left_on = "User-ID",
               right_on = "User-ID"
               
               )
print('Number of ratings from users with at least 100 interactions: %d' % len(ratings_from_selected_users_df))


# In[183]:


ratings_from_selected_users_df.head()


# In[184]:


# Create an object of label encoder

le = preprocessing.LabelEncoder()
le.fit(merged_dataset['Book-Title'].unique()) 


# In[185]:


def smooth_user_preference(x):
    return math.log(1+x, 2)
    
ratings_full_df = ratings_from_selected_users_df.groupby(['Book-Title','User-ID'])['Book-Rating'].sum().apply(smooth_user_preference).reset_index()
print('Number of unique user/item interactions: %d' % len(ratings_full_df))
ratings_full_df.head()


# In[189]:


ratings_train_df, ratings_test_df = train_test_split(ratings_full_df, 
                                   test_size=0.20,
                                   stratify=ratings_full_df['User-ID'],
                                   random_state=42)

print('Number of ratings on Train set: %d' % len(ratings_train_df))
print('Number of ratings on Test set: %d' % len(ratings_test_df))


# In[190]:


ratings_train_df['Book-Title'] = le.transform(ratings_train_df['Book-Title'])
ratings_test_df['Book-Title'] = le.transform(ratings_test_df['Book-Title'])

ratings_train_df.head()


# In[191]:


ratings_train_df.duplicated().sum()


# In[192]:


# Creating a sparse pivot table with users in rows and items in columns

users_items_pivot_matrix_df = ratings_train_df.pivot(index='User-ID', columns='Book-Title', values= 'Book-Rating').fillna(0)

users_items_pivot_matrix_df.head()


# In[193]:


users_items_pivot_matrix = users_items_pivot_matrix_df.values
users_items_pivot_matrix[:10]


# In[194]:


users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]


# #### Singular Value Decomposition 

# In[195]:


# The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15

# Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)


# In[196]:


users_items_pivot_matrix.shape


# In[197]:


# shape of U
U.shape


# In[198]:


sigma = np.diag(sigma)
sigma.shape


# In[199]:


# shape of Vt
Vt.shape 


# After the factorization, we try to to reconstruct the original matrix by multiplying its factors. The resulting matrix is not sparse any more. It was generated predictions for items the user have not yet interaction, which we will exploit for recommendations.

# In[200]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_user_predicted_ratings


# In[201]:


all_user_predicted_ratings.shape


# In[202]:


# Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head()


# In[203]:


# Number of users
len(cf_preds_df.columns)


# In[204]:


class CFRecommender:
    '''
    Class_Name : CFRecommender
    Description : This class is used to recommend book using SVD(Singular value decomposition)
  
    '''
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None): # constructor of class
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self): # to get the model name
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False): # to recommend the items
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating content that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['Book-Title'].isin(items_to_ignore)].sort_values('recStrength', ascending = False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'Book-Title', 
                                                          right_on = 'Book-Title')[['recStrength', 'Book-Title']]


        return recommendations_df # returning dataframe of recommended book
    
cf_recommender_model = CFRecommender(cf_preds_df, filtered_dataset) # making object of CFRecommender


# #### **Evaluation**

# In Recommender Systems, there are a set metrics commonly used for evaluation. We choose to work with **Top-N accuracy metrics**, which evaluates the accuracy of the top recommendations provided to a user, comparing to the items the user has actually interacted in test set.
# 
# This evaluation method works as follows:
# 
# * For each user
#     * For each item the user has interacted in test set
#         * Sample 100 other items the user has never interacted.   
#         * Ask the recommender model to produce a ranked list of recommended items, from a set composed of one interacted item and the 100 non-interacted items
#         * Compute the Top-N accuracy metrics for this user and interacted item from the recommendations ranked list
# * Aggregate the global Top-N accuracy metrics

# In[205]:


#Indexing by personId to speed up the searches during evaluation
ratings_full_indexed_df = ratings_full_df.set_index('User-ID') # set the index as User-ID
ratings_train_indexed_df = ratings_train_df.set_index('User-ID')
ratings_test_indexed_df = ratings_test_df.set_index('User-ID')


# The Top-N accuracy metric choosen was **Recall@N** which evaluates whether the interacted item is among the top N items (hit) in the ranked list of 101 recommendations for a user.

# In[206]:


def get_items_rated(person_id, ratings_df):
    rated_items = ratings_df.loc[person_id]['Book-Title']
    return set(rated_items if type(rated_items) == pd.Series else [rated_items])


# In[207]:


#Top-N accuracy metrics consts

EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:

    def get_not_rated_items_sample(self, person_id, sample_size, seed=42):

            rated_items = get_items_rated(person_id, ratings_full_indexed_df)
            all_items = set(filtered_dataset['Book-Title'])
            non_rated_items = list(all_items - rated_items)

            random.seed(seed) 
            non_rated_items_sample = random.sample(non_rated_items, sample_size)
            return set(non_rated_items_sample)

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index
    
    # Function to evaluate the performance of model for each user

    def evaluate_model_for_user(self, model, person_id):
        
        # Getting the items in test set
        rated_values_testset = ratings_test_indexed_df.loc[person_id]
        
        if type(rated_values_testset['Book-Title']) == pd.Series:
            person_rated_items_testset = set(rated_values_testset['Book-Title'])
        else:
            person_rated_items_testset = set([int(rated_values_testset['Book-Title'])])
            
        rated_items_count_testset = len(person_rated_items_testset) 

        # Getting a ranked recommendation list from the model for a given user
        person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_rated(person_id, ratings_train_indexed_df),topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        
        # For each item the user has interacted in test set
        for item_id in person_rated_items_testset:
            
            # Getting a random sample of 100 items the user has not interacted with
            non_rated_items_sample = self.get_not_rated_items_sample(person_id, sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, seed=item_id%(2**32))

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_rated_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['Book-Title'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['Book-Title'].values
            
            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items
        recall_at_5 = hits_at_5_count / float(rated_items_count_testset)
        recall_at_10 = hits_at_10_count / float(rated_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'rated_count': rated_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    
    # Function to evaluate the performance of model at overall level
    def evaluate_model(self, model): # taking model and self which is an object of respective class
        
        people_metrics = []
        
        for idx, person_id in enumerate(list(ratings_test_indexed_df.index.unique().values)):    
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
            
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('rated_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['rated_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['rated_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()  


# In[208]:


print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)

print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)


# **<u>Conclusions</u>**
# 
# 
# * First We got insights that the majority of users did not rate the books. Also majority of the books were rated <b>8/10</b>.
# 
# * The majority of the readers were in the age group of <b>20-45</b>.
# 
# *  We saw an exponential increase in the publication of books after the year <b>1950</b>.
# 
# * <i>Agatha christie, and William Shakespeare</i> wrote the maximum no. of books.
# 
# * And the maximum books were from the publication house <b>Harlequin and Silhouette</b>.
# 
# * <b>Harry Potter authored by J K Rowling</b> had got the best average ratings followed by To Kill a Mockingbird and The Da Vinci Code.
# 
# * Finally, we evaluated our models using recall at @5 and recall @10 where we got the value to be <b>81%</b>.

# In[ ]:




