#!/usr/bin/env python
# coding: utf-8

# # Netflix EDA
# - Performing Explratory Data Analysis to Understand The dataset

# # Define the problem statement
# - problem_statement = "Analyzing the Netflix dataset to generate insights that could help Netflix in deciding which type of shows/movies to produce and how to grow the business in different countries."

# # Tasks
# - Understand the date, types and missing values
# - Clean the dataset and handle the missing the values
# - Perform data Visuailzation
# - Crete the summary report

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/000/940/original/netflix.csv")  # Replace "file.csv" with the actual file name or data path

print(df.head())


# ** Defining Problem Statement and Analysing basic metrics **

# In[ ]:


# Define the problem statement
problem_statement = "Analyzing the Netflix dataset to generate insights that could help Netflix in deciding which type of shows/movies to produce and how to grow the business in different countries."


# # Basic **metrics**

# In[ ]:



# num_rows, num_columns
print(df.shape)


# In[ ]:


# data_types
print(df.dtypes)


# In[ ]:


#missing_values
print(df.isnull().sum())


# In[ ]:


# replacing null value
df['director']=df['director'].fillna('not_available')
df['country']=df['country'].fillna('not_available')
df['date_added']=df['date_added'].fillna(df['date_added'].mode())
df['rating']=df['rating'].fillna(df['rating'].mode())
df['cast']=df['cast'].fillna('not_available')


# In[ ]:


#missing_values
print(df.isnull().sum())


# In[ ]:


print("Problem Statement:")
print(problem_statement)
print()


# In[ ]:


num_rows = df.shape[0]  # Number of rows
num_columns = df.shape[1]  # Number of columns

# Print the results
print("Number of Rows:", num_rows)
print("Number of Columns:", num_columns)


# In[ ]:


# Data Types
data_types = df.dtypes
print("Data Types:")
print(data_types)
print()


# In[ ]:


# Missing Values
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)
print()


# In[ ]:





# In[ ]:


# Missing Values
missing_values = df.isnull().sum()*100/df.shape[0]
print("Missing Values:")
print(missing_values)
print()


# 2.   **Observations on the shape of data, data types of all the attributes, conversion of categorical attributes to 'category' (If required), missing value detection, statistical summary**

# In[ ]:


# shape of data
print(df.shape)


# In[ ]:


print("Data Types of Attributes:")
print(df.dtypes)
print()


# In[ ]:



# categorical attributes
print(df.info())


# In[ ]:


# Missing value detection
print(df.isnull().sum())


# In[ ]:


# Statistical summary
print(df.describe())


# **3. Non-Graphical Analysis: Value counts and unique attributes **

# In[ ]:


# Value counts for 'country' column
print(df['country'].value_counts())


# In[ ]:


# Unique attributes for 'country' column
print(df['country'].unique())


# **4. Visual Analysis - Univariate, Bivariate after pre-processing of the **

# In[ ]:


#convert the columns to strings
df['cast'] = df['cast'].astype(str) #actor = cast
df['director'] = df['director'].astype(str)
df['country'] = df['country'].astype(str)

# Print the DataFrame
print(df)


# In[ ]:


# For unnesting the 'cast' column, we can split the actors into separate rows
df['cast'] = df['cast'].str.split(', ')
df_unnested = df.explode('cast')
print(df)


# In[ ]:


# Similarly, unnest the 'director' column
df_unnested['director'] = df_unnested['director'].str.split(', ')
df_unnested = df_unnested.explode('director')
print(df)


# In[ ]:


# Similarly, unnest the 'country' column
df_unnested['country'] = df_unnested['country'].str.split(', ')
df_unnested = df_unnested.explode('country')
print(df)


# In[ ]:


# Univariate Analysis
# Let's visualize the distribution of content types (TV shows vs. movies)
plt.figure(figsize=(6, 4))
sns.countplot(x='type', data=df_unnested)
plt.title('Distribution of Content Types (TV Shows vs. Movies)')
plt.xlabel('Content Type')
plt.ylabel('Count')
plt.show()


# In[ ]:


# Bivariate Analysis                                                 top 10 country
# Let's visualize the number of movies and TV shows in each country
plt.figure(figsize=(14, 8))
sns.countplot(x='country', hue='type', data=df_unnested)
plt.title('Number of Movies and TV Shows in Each Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Content Type')
plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Check and convert columns to strings before splitting
df['cast'] = df['cast'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
df['director'] = df['director'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
df['country'] = df['country'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

# Get the top 10 countries by count of content
top_countries = df['country'].explode().value_counts().head(10).index

# Filter the dataframe for the top 10 countries
df_top_countries = df[df['country'].explode().isin(top_countries)]

# Create a countplot for the top 10 countries and content type
plt.figure(figsize=(14, 8))
sns.countplot(x='country', hue='type', data=df_top_countries)
plt.title('Number of Movies and TV Shows in Top 10 Countries')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Content Type')
plt.show()


# In[ ]:


# Separate movies and TV shows
movies = df[df['type'] == 'Movie']
tv_shows = df[df['type'] == 'TV Show']


# In[ ]:


# Univariate Analysis for continuous variable 'duration' (after converting it to numeric)
df_unnested['duration'] = pd.to_numeric(df_unnested['duration'], errors='coerce')

print(df)


# **4.1 For continuous variable(s): Distplot, countplot, histogram for univariate analysis **

# In[ ]:


# Distplot for release_year distribution
plt.figure(figsize=(12, 6))
sns.distplot(df['release_year'], bins=30, kde=True, hist=True)
plt.xlabel('Release Year')
plt.ylabel('Density')
plt.title('Release Year Distribution')
plt.show()


# In[ ]:


# Histogram for release_year distribution
plt.figure(figsize=(16, 12))
plt.hist(df['release_year'], bins=30, edgecolor='black')
plt.xlabel('Release Year')
plt.ylabel('Frequency')
plt.title('Release Year Distribution')
plt.show()


# **`4.2 For categorical variable(s): Boxplot`**

# In[ ]:


# Boxplot for 'rating' distribution against 'release_year'
plt.figure(figsize=(14, 6))
sns.boxplot(x='rating', y='release_year', data=df)
plt.xlabel('Rating')
plt.ylabel('Release Year')
plt.title('Boxplot of Release Year for Each Rating')
plt.xticks(rotation=90)
plt.show()


# **4.3 For correlation: Heatmaps, Pairplots**

# **5. Missing Value & Outlier check (Treatment optional)**

# In[ ]:


df.columns


# In[ ]:


# Outlier Check (for numerical columns)
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=df[col])
    plt.xlabel(col)
    plt.title(f'Boxplot of {col}')
    plt.show()


# **6. Insights based on Non-Graphical and Visual Analysis**

# **6.1 Comments on the range of attributes**

# In[ ]:


df['duration'].value_counts()


# In[ ]:


movie = df[df['type']=='Movie']
shows = df[df['type']=='TV Show']
shows.columns


# In[ ]:


shows['duration'].value_counts()


# In[ ]:





# **6.2 Comments on the distribution of the variables and relationship between them**

# In[ ]:


# Distribution of TV Shows vs. Movies
plt.figure(figsize=(6, 4))
sns.countplot(x='type', data=df)
plt.title('Distribution of TV Shows vs. Movies')
plt.xlabel('Content Type')
plt.ylabel('Count')
plt.show()


# In[ ]:


movie['country']


# In[ ]:


sns.countplot(data=shows, x='duration')
plt.title('Distribution of Content Across Countries')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# Distribution of Content Across Countries
plt.figure(figsize=(12, 6))
sns.countplot(x='country', data=movie, order=movie['country'].value_counts().index[:10])
plt.title('Distribution of Content Across Top 10 Countries')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# Distribution of Content Across Countries
plt.figure(figsize=(12, 6))
sns.countplot(x='country', data=df, order=df['country'].value_counts().index[:10])
plt.title('Distribution of Content Across Top 10 Countries')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# Release Trends Over the Years
plt.figure(figsize=(10, 6))
sns.distplot(df['release_year'], bins=30, kde=True)
plt.title('Release Trends Over the Years')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.show()


# In[ ]:


# TV Show Launch Time Analysis
df['date_added'] = pd.to_datetime(df['date_added'])
df['month_added'] = df['date_added'].dt.month

plt.figure(figsize=(8, 5))
sns.countplot(x='month_added', data=df, hue='type')
plt.title('TV Show Launch Time Analysis')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title='Content Type', labels=['Movie', 'TV Show'])
plt.show()


# In[ ]:


# Content Duration Distribution
# Sample DataFrame
data = {'duration': ['90 min', '120 min', '150 min', '75 min', '105 min']}
df = pd.DataFrame(data)

# Preprocess 'duration' column to extract numerical values
df['duration'] = df['duration'].str.extract('(\d+)').astype(float)

plt.figure(figsize=(10, 6))
sns.distplot(df['duration'], bins=30, kde=True)
plt.title('Content Duration Distribution')
plt.xlabel('Duration')
plt.ylabel('Density')
plt.show()


# In[ ]:


# TV Rating Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='rating', data=df, order=df['rating'].value_counts().index)
plt.title('TV Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# In[ ]:


# Genre Popularity Analysis
plt.figure(figsize=(12, 6))
sns.countplot(y='listed_in', data=df, order=df['listed_in'].value_counts().index[:10])
plt.title('Genre Popularity')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()


# In[ ]:


# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# **6.3 Comments for each univariate and bivariate plot**

# **# Univariate Plots**

# In[ ]:


# 1. Distribution of TV Shows vs. Movies
plt.figure(figsize=(6, 4))
sns.countplot(x='type', data=df)
plt.title('Distribution of TV Shows vs. Movies')
plt.xlabel('Content Type')
plt.ylabel('Count')
plt.show()


# In[ ]:


# 2. Distribution of Content Across Countries
plt.figure(figsize=(12, 6))
sns.countplot(x='country', data=df, order=df['country'].value_counts().index[:10])
plt.title('Distribution of Content Across Top 10 Countries')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# 3. Release Trends Over the Years
plt.figure(figsize=(10, 6))
sns.distplot(df['release_year'], bins=30, kde=True)
plt.title('Release Trends Over the Years')
plt.xlabel('Release Year')
plt.ylabel('Density')
plt.show()


# In[ ]:


# 4. TV Show Launch Time Analysis
df['date_added'] = pd.to_datetime(df['date_added'])
df['month_added'] = df['date_added'].dt.month

plt.figure(figsize=(8, 5))
sns.countplot(x='month_added', data=df, hue='type')
plt.title('TV Show Launch Time Analysis')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title='Content Type', labels=['Movie', 'TV Show'])
plt.show()


# In[ ]:


# 5. Content Duration Distribution
# Assume 'duration' column contains both minutes and season information
df['duration_min'] = df['duration'].str.extract('(\d+)').astype(float)

plt.figure(figsize=(10, 6))
sns.distplot(df['duration_min'].dropna(), bins=30, kde=True)
plt.title('Content Duration Distribution')
plt.xlabel('Duration (minutes)')
plt.ylabel('Density')
plt.show()


# In[ ]:


# 6. TV Rating Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='rating', data=df, order=df['rating'].value_counts().index)
plt.title('TV Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# In[ ]:


# 7. Genre Popularity Analysis
plt.figure(figsize=(12, 6))
sns.countplot(y='listed_in', data=df, order=df['listed_in'].value_counts().index[:10])
plt.title('Genre Popularity')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()


# **# Bivariate Plot**

# In[ ]:


# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:


df = pd.read_csv("https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/000/940/original/netflix.csv")
df['cast']=df['cast'].str.split(',')


# In[ ]:



df['cast']=df['cast'].str.split(',')


# In[ ]:


df.explode('cast')


# # Summary
# 
# so, far we had perform lots of operation over the dataset to dig out same very useful informantion form it. we have to conclude the dataset in few line. than we can say that.
# 
# - Netflix has more movies than TV Shows
# - Most number of Movies and Tv shows are produced by United Stats by following by India who has produced the second most number of movies on Netflix
# - Most of Content on Netflix(Movies and TV Shows Combined) is for Mature Audience
# - 2018 is the year in which Netflix realese alot more contect as compared to other years
# - International Movie and Dramas are the most popular Genres on Netflix

# In[ ]:




