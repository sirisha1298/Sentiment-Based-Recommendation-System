# Sentiment-Based-Recommendation-System

The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings. 

In order to do this, you planned to build a sentiment-based product recommendation system, which includes the following tasks.

1. Data sourcing and sentiment analysis
2. Building a recommendation system
3. Improving the recommendations using the sentiment analysis model
4. Deploying the end-to-end project with a user interface

Files Used : 
1. SBPRS_Sirisha.ipynb - This is the whole code which consists of all the ML models used for prediction of the sentiment. The models we have used are - Logistic Regression model with SMOTE (to correct the class imbalance), Multinomial Naive Bayes with SMOTE, XGBoost with SMOTE and RandomForest Classifier with SMOTE. 
2. model.py - This consists of end-to-end code used by flask application for the heroku deployment. This consists of a single best ML model - Logistic Regression and single best recommendation system - User based model to get the top 5 recommendations for a user. Pickle files are generated and loaded so that they can be used directly in the app.py file. 
3. app.py - This file consists of code which connects the backend model to the frontend html page. 
4. models/* - This folder consists of all the pickle files generated during the model creation and the recommendation system creation. These pickle files are used by our app.py file to get the recommendations. 
5. templates/* - This folder contains index.html file which is a front-end HTML page for submitting the user input and displaying the top 5 recommendations.
6. ratings_df.csv - The final dataframe file which has bee converted to a csv file after the whole cleaning, analysis and processing is done. This file is used in the app.py to read few items from the dataframe. 
7. Procfile - This file is used by the Heroku application
8. requirements.txt - This file is used by the Heroku application to install the reqired libraries and packages on the Heroku cloud application.
9. sample30.csv - This is the original data file which we converted to dataframe in the initial step of the data reading.

### App is deployed in Heroku and is available at : https://sbprs-sirisha.herokuapp.com/ 
