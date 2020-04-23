# Exploratory Data Analysis and Machine Learning 


This section focusses on Statistical Exploratory Data Analysis and Machine Learning Tasks to solve some really intersting problems Namely :

1) Imbalanced Classification :  Weighing Oversampling Techniques with Undersampling Techniques + Feature Engineering to boost model skill
2) Unsupervised Sentiment Analysis (In Progress ):  Using a dot product of tf-idf score and polarity to determine the sentiment of a sentence and use that as a feature in modelling
3) For a banking company using a Targetted Ad Campaign, Varying model decision threshold to Boost Precision and Recall values of a model, to efficiently identify customers with high probability of conversion while ensuring minimization of ad costs
4) Explore what factors lead to conversion of a customer across different regions and providing a recommendation of strategizing the approach for each age group of customers in light of the medium to approach them.


This section has the following projects : 

### 1) Product_Reviews : 

A detailed study on a highly imbalanced dataset having customer Feedback, Ratings, Recommend Status and Many Engineered Features on Amazon Kindle

The notebook aims at applying several Machine Learning Techniques to identify those customers which didn't recommend the product. Keeping Exploratory data analysis at heart, this notebook aims at creating a pipeline and tries several techniques to come up with features which could aid in identify customers who wouldn't recomment the product.

The work is still in progress as I'm applying `Unsupervised Sentiment Analysis` on customer reviews to generate a score per review and use that as an additional feature along with other Engieenered Features in model building


### 2) BankMarketing :

For a banking compaign running an Advertisement campaign, this project aims at identifying the customers most likely to convert and identofy an approach to minimize cost of advertiment while ensuring the company squeezes out the maximum money from the customer base. The project applies Ensemble Learning Methods to solve the problem, but can further be expanded to apply Neural Net further an alternative approach if cost per ad varies.


### 3) Customer Conversion Rate :

Keeping Exploratory Data Analysis at heart, this project aims at seeking answers to conversion of visitors to a website to potential customers in different regions and the means of targetting. 

The projects explores and finds answers to the following areas : 

a) The Conversion Distribution and advertisment impact by Region

b) The Conversion Distribution by Age among different Regions

c) Does reaching out to old users lead to higher conversion as compared to new users

d) Which source of advertiment leads to higher conversion across different Regions

NOTE : Due to the class imbalance problem, I am builiding the model for this and the first project simultaneouly. to present a clean and organized work, The modelling part with the list of changing hyperparameters are performed in separate notebooks and the final model which beats the bechmakr in either case would be part of these notebooks


