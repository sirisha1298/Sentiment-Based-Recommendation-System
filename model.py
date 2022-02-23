# Import the necessary libraries
import pandas as pd
import numpy as np
from numpy import * 
from collections import defaultdict
from collections import Counter
import csv
import re 
import string

# Visualization libraries
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# To show all the columns
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 300)

# Avoid warnings
import warnings
warnings.filterwarnings("ignore")

# NLTK libraries
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import en_core_web_sm
nlp = en_core_web_sm.load()

# Modelling
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Instantiate two Global variables 
stop_words = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()

# Helper functions 

# Data Preprocessing 
def preprocess_df(df):
    # Drop the columns which are not useful and have high missing values
    df = df.drop(columns=['reviews_userCity', 'reviews_userProvince'])
    df = df.drop(columns=['reviews_didPurchase', 'reviews_doRecommend'])

    # Drop all those rows where manufacturer is null
    df = df[df['manufacturer'].notna()]
    # Drop all those rows where reviews_date is null
    df = df[df['reviews_date'].notna()]
    # Drop all those rows where reviews_title is null
    df = df[df['reviews_title'].notna()]
    # Drop all those rows where reviews_username is null
    df = df[df['reviews_username'].notna()]
    # Drop all those rows where user_sentiment is null
    df = df[df['user_sentiment'].notna()]

    # Remove the prefix word reviews_ from the column names
    df.columns = list(map((lambda x : x.lstrip("reviews_") if x.startswith("reviews_") else x), list(df.columns)))
    # Correct the column name to rating which got stripped during lstrip
    df = df.rename(columns={'ating':'rating'})
    print(df.info()) 

    # Let's convert target variable to a numerical binary value for the classification modeling purpose
    df['user_sentiment'] = df['user_sentiment'].apply(lambda x : 1 if x=="Positive" else 0)
    print(df['user_sentiment'].value_counts())

    # Convert the username, title and text to lowercase incase of any ambiguity
    df['username'] = df['username'].apply(lambda x : x.lower())
    df['text'] = df['text'].apply(lambda x : x.lower())
    df['title'] = df['title'].apply(lambda x : x.lower())

    # Use strip to remove leading and trailing white spaces
    df['username'] = df['username'].apply(lambda x : x.strip())
    df['text'] = df['text'].apply(lambda x : x.strip())
    df['title'] = df['title'].apply(lambda x : x.strip())

    return df

# Data Analysis on Users and Items
def analyse_df(df):
    # Get the number of unique users and items 
    print("Number of unique users:\n")
    print(df['username'].value_counts())

    print("Number of unique items:\n")
    print(df['name'].value_counts())

# Check for hyperlinks
def check_hyperlink(x):
    if x.find("http:") == -1 or x.find("https:") == -1:
        return 0
    else:
        return 1

# Helper function to remove stop words
def remove_stopwords(text):
    words = [i for i in text.split() if i not in stop_words]
    return ' '.join(words)

# Helper function to lemmatize the sentence
def lemmatize_sentence(text):
    sent = [wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(text)]
    return " ".join(sent)

# Text processing - Tokenization and Lemmatization
def text_processing(df):
    df['review'] = df['text']

    # Remove all the punctuation marks from the review column
    df['review'] = df['review'].str.replace('[^\w\s]','')
    # Remove stop words from the df using above function
    df['review'] = df['review'].apply(remove_stopwords)

    # Lemmatize every word in a sentence using the above defined function
    df['review'] = df['review'].apply(lemmatize_sentence)
    df['review'] = df['review'].str.strip()

    return df

# Visualize the dataframe 
def visualize_df(df):
    # Visualize the review length against the number of reviews
    plt.figure(figsize=(10,6))
    doc_lens = [len(d) for d in df['review']]
    plt.hist(doc_lens, bins = 10)
    plt.xlabel('Review Length')
    plt.ylabel('Number of Reviews')
    plt.show()

    # Visualize the percentage of positive and negative reviews
    plt.hist(df['user_sentiment'])
    plt.xlabel('Category')
    plt.ylabel('Number of Reviews')
    plt.show()

# Define a function which does all the modeling, displaying metrics,
#  roc-auc curves based on the model which we are going to send as the input.
def model_fit(X_train_tf, Y_train, X_test_tf, Y_test, ml_model, coef_show):
    
    model = ml_model.fit(X_train_tf, Y_train)
    model_train_pred = model.predict(X_train_tf)
    model_test_pred = model.predict(X_test_tf)
    accuracy_train = model.score(X_train_tf, Y_train)
    accuracy_test = model.score(X_test_tf, Y_test)
    model_performance = classification_report(Y_test, model_test_pred)
    validation_pred_proba_grad = model.predict_proba(X_test_tf)
    roc_auc = roc_auc_score(Y_test, validation_pred_proba_grad[:,1])
    
    print("***** Accuracy of the Train model: ", accuracy_train, " *******")
    print("***** Accuracy of the Test model: ", accuracy_test, " *******")
    print('')
    print(model_performance)
    print('')
    print("***** ROC_AUC score: ", roc_auc, " *******")
    print("*************************************************")
    
    if coef_show == True:
        featureNames = tfidf.get_feature_names()
        coef = model.coef_.tolist()[0]
        coeff_df = pd.DataFrame({'Word' : featureNames, 'Coefficient' : coef})
        coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
        print('')
        print("************ Top 10 positive features (variables) ************")
        print(coeff_df.head(20).to_string(index=False))
        print('')
        print("************ Top 10 negative features (variables) ************")        
        print(coeff_df.tail(20).to_string(index=False))
    
    print("Confusion matrix for train and test set")

    plt.figure(figsize=(10,5))

    c_train = confusion_matrix(Y_train, model_train_pred)
    c_test = confusion_matrix(Y_test, model_test_pred)
    
    plt.subplot(1,2,1)
    sns.heatmap(c_train/np.sum(c_train), annot=True , fmt = ' .2%', cmap="Blues")

    plt.subplot(1,2,2)
    sns.heatmap(c_test/np.sum(c_test), annot=True , fmt = ' .2%', cmap="Blues")

    plt.show()
    
    # Calculate Sensitivity and Specificity
    true_neg = c_test[0, 0]
    false_pos = c_test[0, 1]
    false_neg = c_test[1, 0]
    true_pos = c_test[1, 1]
    
    sensitivity = true_pos/(false_neg+true_pos)
    print("Sensitivity is : ",sensitivity)
    
    specificity = true_neg/(false_pos+true_neg)
    print("Specificity is : ",specificity)

    
    return model

def recommendation_system(train_df, test_df, train_dummy):
    # Create a pivot table for the user-item matrix
    train_df_pt = train_df.pivot_table(index='username', columns='name', values='rating')
    print(train_df_pt.shape)

    # Normalising the rating of each user around 0 mean 
    # This does the mean for each row as the axis is 1
    mean = np.nanmean(train_df_pt, axis=1)

    print(mean)
    print(sum(mean))
    print(len(mean))

    # subtract the mean from the train df and transpose
    train_df_sub = (train_df_pt.T - mean).T

    # finding cosine similarity with NaN
    user_correlation_sub = 1 - pairwise_distances(train_df_sub.fillna(0), metric='cosine')
    user_correlation_sub[np.isnan(user_correlation_sub)] = 0
    print(user_correlation_sub)

    # User-User prediction 
    # Doing the prediction for the users which are positively related with other users, 
    # and not the users which are negatively related as we are interested in the users which are more
    # similar to the current users. So, ignoring the correlation for values less than 0.

    # Remove negative correlations among the users
    user_correlation_sub[user_correlation_sub<0]=0
    #Check if there are legitimate values in the correlation matrix
    print(sum(sum(user_correlation_sub)))

    # predict the user rating
    user_pred_rating = np.dot(user_correlation_sub, train_df_pt.fillna(0))

    #Since we are interested only in the items not rated by the user, 
    # we will ignore the items rated by the user by making it zero.
    user_final_rating = np.multiply(user_pred_rating,train_dummy)

    return user_final_rating, user_correlation_sub, train_df_sub

# Helper function to normalize the range of the rating
def normalize_range(common_user_pred_rating):
    x  = common_user_pred_rating.copy() 
    x = x[x>0]

    scaler = MinMaxScaler(feature_range=(1, 5))
    print(scaler.fit(x))
    y = (scaler.transform(x))
    return y


def evaluation_rs(train_df, test_df, train_dummy, user_final_rating, user_correlation_sub, train_df_sub):
    # Get the common users in both train_df and test_df
    common_users = test_df[test_df['username'].isin(train_df['username'])]
    print(common_users.shape)

    # Convert the common users df to pivot table 
    common_users_pt = common_users.pivot_table(index='username', columns='name', values='rating')
    # Convert the user_correlation_sub to a dataframe for easy access
    user_corr_df = pd.DataFrame(user_correlation_sub)

    # Change the index of the user_corr_df to usernames
    user_corr_df['username'] = train_df_sub.index
    user_corr_df.set_index('username',inplace=True)

    # Convert the usernames to list
    usernames_list = common_users['username'].tolist()
    user_corr_df.columns = train_df_sub.index.tolist()
    user_corr_df_1 =  user_corr_df[user_corr_df.index.isin(usernames_list)]
    user_corr_df_2 = user_corr_df_1.T[user_corr_df_1.T.index.isin(usernames_list)]
    user_corr_df_3 = user_corr_df_2.T
    # Mark the negative correlations to 0 
    user_corr_df_3[user_corr_df_3<0]=0

    # predict the user rating 
    common_user_pred_rating = np.dot(user_corr_df_3, common_users_pt.fillna(0))

    # Create a dummy test dataframe 
    test_dummy = common_users.copy()
    # make the rating to 1 if the user has given a rating, else 0
    test_dummy['rating'] = test_dummy['rating'].apply(lambda x : 1 if x>=1 else 0)
    # Create a pivot table for the dummy test dataframe
    test_dummy_pt = test_dummy.pivot_table(index='username', columns='name', values='rating').fillna(0)

    # predict the final testing 
    common_user_pred_rating = np.multiply(common_user_pred_rating, test_dummy_pt)
    
    # normalize the range of the rating
    y = normalize_range(common_user_pred_rating)

    # Creating a pivot table of common users
    common_users_1 = common_users.pivot_table(index='username', columns='name', values='rating')
    # Finding total non-NaN value
    total_non_nan = np.count_nonzero(~np.isnan(y))
    # calculate rmse 
    rmse = (sum(sum((common_users_1 - y )**2))/total_non_nan)**0.5

    return rmse

# Recommend top 5 products for a user
def top5_recommend(user_name):
    # top20 ratings for a particular user
    top20_ratings = user_final_rating.loc[user_name].sort_values(ascending=False)[0:20]
    # using lr_model predict the sentiment score of a product based on the review.

    products = []
    sentiments = []
    for item in top20_ratings.index.tolist():
        # get the item name
        item_name = item
        # split the review into tokens to send it as features to the model
        item_review = df[df['name']==item_name]['text'].tolist()
    
        # get the features of the review
        tfidf_features = tfidf.transform(item_review)
        # get the name and the sentiment of the review
        products.append(item_name)
        sentiments.append(lr_model.predict(tfidf_features).mean())
    
    # Create a dataframe of the products and the sentiments
    products_df = {'Product' : products, 'Sentiment' : sentiments}
    products_df = pd.DataFrame(products_df)
    # Sort the dataframe based on the sentiment score in the descending order
    products_df = products_df.sort_values(by=['Sentiment'], ascending=False)

     # top 5 recommendations will be the following
    top5_products = products_df['Product'][0:5].tolist()
    return top5_products
    


#####################################################################################################

# Read the dataset into a dataframe
df = pd.read_csv("sample30.csv")
print(df.shape)
print(df.info())
print(list(df.columns))
print(df.head())

# Print the % of missing values for each column again 
print("Percentage of missing values : \n")
print(round(100*((df.isnull().sum())/len(df)),5))

# Preprocess the data using the above function 
df = preprocess_df(df)
# Analyse dataframe
analyse_df(df)
# Text processing
df = text_processing(df)
# visualize data 
visualize_df(df)

################################################################################################

# Assign the X and Y variables
X = df['review']
Y = df['user_sentiment']

# Train Test split for the feature extraction and modeling purpose 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)

# print the shapes of x_train, x_test, y_train and y_test
print('X_train', X_train.shape)
print('Y_train', Y_train.shape)
print('X_test', X_test.shape)
print('Y_test', Y_test.shape)

# Instantiate a TfIdfVectorizer Object

tfidf = TfidfVectorizer(ngram_range=(1,3), lowercase=True, analyzer='word',stop_words= 'english',
                        token_pattern=r'\w{1,}')

# fit the model
tfidf.fit(X_train)

# extract the train features from the fitted train dataset using transform
train_features = tfidf.fit_transform(X)

## transforming the train and test datasets
X_train_tf = tfidf.transform(X_train.tolist())
X_test_tf = tfidf.transform(X_test.tolist())


# Since we saw huge class imbalance in the previous outputs, 
# its better we rectify it first before training the model
# Instantiate the Smote class object
smote = SMOTE()

# Apply smote resampling on the train and test sets.
X_train_tf, Y_train = smote.fit_resample(X_train_tf, Y_train)
X_test_tf, Y_test = smote.fit_resample(X_test_tf, Y_test)

# Print the shape of each dataset.
print('X_train_tf', X_train_tf.shape)
print('Y_train', Y_train.shape)
print('X_test_tf', X_test_tf.shape)
print('Y_test', Y_test.shape)

# Apply the best model we obtained in the previous analysis 

# Instantiate a logistic regression object
logistic_regression = LogisticRegression()

# Use the above defined function to create the Logistic regression model
lr_model = model_fit(X_train_tf, Y_train, X_test_tf, Y_test, logistic_regression, True)

#################################################################################################

# Copy the original df to a new dataframe 
reviews = df

# Diving the dataset into train and test 
train_df, test_df = train_test_split(reviews, train_size=0.7, test_size=0.3, random_state=100)

print(train_df.shape)
print(test_df.shape)

# Create a temporary pivot table with index as username 
# and the items as the columns to which the user has given a rating for.
user_pt = reviews.pivot_table(index='username', columns='name', values='rating').fillna(0)
user_df = pd.DataFrame(user_pt)
print(user_df.shape)

# Create a dummy_train dataframe such that this 
# df will be used for prediction of the items which the user has not rated yet.
train_dummy = train_df.copy()

# As we know we need to mark the items which have not been rated by the user as 1
train_dummy.rating = train_dummy.rating.apply(lambda x : 0 if x>=1 else 1)

# Convert the dummy train dataset into matrix format.
train_dummy = train_dummy.pivot_table(index='username', columns='name', values='rating').fillna(1)


# Cosine Similarity is a measurement that quantifies the similarity between two vectors

# Adjusted cosine similarity is a modified version of vector-based similarity where we incorporate
#  the fact that different users have different ratings schemes. In other words, 
# some users might rate items highly in general, and others might give items lower ratings as a preference.
#  To handle this nature from rating given by user , we subtract average ratings for each user from each
#  user's rating for different movies.

# User based recommendation system 

# Get the final ratings
user_final_rating, user_correlation_sub, train_df_sub = recommendation_system(train_df, 
                                                                test_df, train_dummy)

# Evaluation of the user based recommendation system
rmse = evaluation_rs(train_df, test_df, train_dummy, user_final_rating, user_correlation_sub, train_df_sub)
print(rmse)

#######################################################################################################

# Saving the models in .pkl formats

# dump the logistic regression to lr_model.pkl file
pickle.dump(lr_model,open('models/lr_model.pkl', 'wb'))
# load the lr_model  pickle object
lr_model_obj =  pickle.load(open('models/lr_model.pkl', 'rb'))
# dump the tfidf vectorizer to tfidf_model file 
pickle.dump(tfidf,open('models/tfidf.pkl','wb'))
# load the tfidf pickle object
tfidf_obj = pickle.load(open('models/tfidf.pkl','rb'))
# dump the pickle file for user_final_rating 
pickle.dump(user_final_rating, open('models/final_ratings.pkl', 'wb'))
# load the pickle object for user_final_rating
final_ratings = pickle.load(open('models/final_ratings.pkl', 'rb'))


# top 5 recommendations for a user
user_name = input("Enter a user name to recommend the products : ")
top5_products = top5_recommend(user_name)
print(top5_products)

########################################################################################################

# Save the final dataframe to a csv files 
# so that we can use this dataframe in our flask app
df.to_csv("ratings_df.csv",index=False)


### Flask code is available in "app.py" 
### HTML code is available in "templates/index.html"
### App is deployed in Heroku and is available at : https://sbprs-sirisha.herokuapp.com/ 
### Github link is available at :  https://github.com/sirisha1298/Sentiment-Based-Recommendation-System 

#############################################################################################



















