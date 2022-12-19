# Week 12 Capstone project

# Flight price prediction dataset

This dataset was found in Kaggle [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

## Problem description

The final target of this flight price prediction model is to analyse the flight booking dataset obtained from “Ease My Trip” website and to perform various machine learning models in order to get meaningful information from it, especially if you are paying a right price for a flight ticket. 'Ease My Trip' is an internet platform for booking flight tickets, and hence a platform that potential passengers use to buy tickets. A thorough study of the data will aid in the discovery of valuable insights that will be of enormous value to passengers. The idea of this project is to train some ML models as Linear Regression, since the goal is to predict a continuous value, in this case the price, considering the following features:

- Airline: The name of the airline company is stored in the airline column. It is a categorical feature having 6 different airlines.
- Flight: Flight stores information regarding the plane's flight code. It is a categorical feature.
- Source City: City from which the flight takes off. It is a categorical feature having 6 unique cities.
- Departure Time: This is a derived categorical feature obtained created by grouping time periods into bins. It stores information about the departure time and have 6 unique time labels.
- Stops: A categorical feature with 3 distinct values that stores the number of stops between the source and destination cities.
- Arrival Time: This is a derived categorical feature created by grouping time intervals into bins. It has six distinct time labels and keeps information about the arrival time.
- Destination City: City where the flight will land. It is a categorical feature having 6 unique cities.
- Class: A categorical feature that contains information on seat class; it has two distinct values: Business and Economy.
- Duration: A continuous feature that displays the overall amount of time it takes to travel between cities in hours.
- Days Left: This is a derived characteristic that is calculated by subtracting the trip date by the booking date.

Finally the target value is:

- Price: Target variable stores information of the ticket price.

## Problem approach

Since there is a regression problem where needs to predict a continuous value, in this case the price, an approach of Linear Regression, Decision Tree Regressor, Random Forest Regressor and XGBoost is proposed.

## Instructions on how to run the project

### Data

The data is also founded in the [Data](https://raw.githubusercontent.com/FranciscoOrtizTena/ML_Zoomcamp/main/12_week_capstone_project/flight_price_prediction.csv)

### Files

### Training

A) The first file is the [notebook.ipynb](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/tree/main/8_week) where the data preparation, data cleaning, exploratory data analysis, feature importances, model selection process and parameter tuning is done.

0. Loaded libraries: Numpy and pandas for numericla processing; matplotlib and seaborn for making graphs; LinearRegression, Ridge, Lasso, DecisitonTreeRegressor, RandomForestRegressor and XGBoost are the models proposed to predict the price of the flight ticket; train_test_split, DictVectorizer, cross_val_predict for preprocess the data; mean_squared_error, r2_score and mutual_info_score as metrics to evaluate the different models and finally tqdm for take time of the loops.
1. After that, the data is downloaded from the specific link.
2. The data preparation and data cleaning is performed, the string values in the rows are transformed to lowercase and without a blank space. For this problem the feature flight is useless, since it just a number to identify a flight number, finally I identified if there are some outliers, null values or missing data.
3. Exploratory Data Analysis (EDA): I started checking information about the categorical data, which in this case was the airline, the source city, the departure time, the stops, the arrival time, the destination city and the class. Vistara has the most flights, the most frequent source city is Delhi, the most frequent departure time is at morning, the most frequent flights are bought with one stop, the most frequent arrival time is at night, the most frequent destination city is Mumbai and the most travel class is economy. For the numerical values some graphs were also done, showing that the average duration of the flights is 11 hours, the tickets are bought around 27 days in advanced and the mean price is 15,000. Correlating the prices, Vistara is the costlier airline, while Airasia is chepaer, for the source_city, the costlier is chennai and the cheaper is delhi, for the departure_time, the costlier is at night and the cheaper is late_night, for the stops, the costlier is with one and the cheaper is with zero, for the arrival_time, the costlier is at evening and the cheaper is at late night, for the destination_city, the costlier is Kolkata and the cheaper is dehli and is costlier traveling in business than economy.
4. Model selectiong process and parameter tuning: here I trained several models (LinearRegression, DecisionTreeRegressor, RandomForestRegressor and XGBoost) to try to find the best performance using mean_squared_error and r2_score. First the validation framework was created, where the data was splitted into train, validation and test and the target value was passed to another variable dropped from the original data, then the preprocessing was done using the DictVectorizer to make a one-hot-encode for the categorical features. Finally the four models were trained and tuned in the following parameters
4.1.- For the LinearRegression the Ridge and Lasso regularization was tuned, the best model was the LinearRegression without regulatization, with a r2_score = 91.08.
4.2.- For the DecisionTreeRegressor the max_depth and min_sample_leaf were tuned, it was found that the best model fiths with max_depth = 30 and a min_sample_leaf =5, with a r2_score = 98.27
4.3.- For the RandomForestRegresor the n_estimators, max_depth and min_sample_leaf were tunned with the following best parameters n_estimators=75, max_depth=25, min_sample_leaf=5, with r2_score=98.57.
4.4.- For the XGBoost the eta, max_depth and min_child_weight was tunned with the following best parameters eta=1.0, max_depth=10 and min_child_weight=1 with a r2_score=98.41.

5. The models were compared, where RandomForestRegressor was the one with the best r2_score.
6. The final model was trained with the full data train and compared with the test data, the r2_score was better with a 98.57%.

## Deploymeny locally
B) The second file is the [build_bento_model_price.ipynb](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/12_week_capstone_project/build_bento_model_price.ipynb) file, here the best model, which was the Random Forest Regressor model, is trained again and saved into a model to deploy it, then
