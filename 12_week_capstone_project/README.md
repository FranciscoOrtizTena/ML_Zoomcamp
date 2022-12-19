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

## Deployment locally
B) The second file is the [build_bento_model_price.ipynb](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/12_week_capstone_project/build_bento_model_price.ipynb) file, here the best model, which was the Random Forest Regressor model, is trained again and saved into a model to deploy it, then [train.py](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/12_week_capstone_project/train.py) is elaborated with the script to deploy it locally using the bento interface.

The you can run it locally using the following command in the terminal

```bash
bentoml serve train.py:svc --production
```

But be careful to specify the correctly tag of the model in the script, since if you run it from your computer the tag may change.

You can visit the [local host](http://localhost:3000/) to make predictions

Another option to run it locally is to export the model using the following methodology

```bash
bentoml models export flight_price_prediction:wqqm6cd7xcjtzzc6
```
And will create a bentomodel file, you can then pass the file and then use the following command in your terminal to import it

```bash
bentoml models import flight_price_prediction:wqqm6cd7xcjtzzc6
```

C) Finally, you can use the third file [predict.ipynb](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/12_week_capstone_project/predict.ipynb) to load the bentomodel and predict the price of the flight. Remember you need to pass a dictionary as follows:

```python
{"airline": str,
"source_city": str,
"departure_time": str,
"stops": str,
"arrival_time": str,
"destination_city": str,
"class": str,
"duration": float,
"days_left": int}
```

Here is an example:

```python
{"airline": "vistara",
"source_city": "delhi",
"departure_time": "early_morning",
"stops": "two_or_more",
"arrival_time": "evening",
"destination_city": "chennai",
"class": "economy",
"duration": 12.42,
"days_left": 16}
```

Following is a pics on locally deployment

![deployment locally](https://user-images.githubusercontent.com/108430575/208488022-ee700c13-90f8-46a5-831b-56c6768e952d.PNG)

## Deployment using Docker
Once you create you bento model in the script [build_bento_model_price.ipynb](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/12_week_capstone_project/build_bento_model_price.ipynb), you need to create a [bentofile.yaml](), specifying the service, labels programming language and the different packages to use.

Then you need to build your bento with.

```bash
bentoml build
```

To deploy your model using the Docker images, you need first to containerize the previous model into a Docker image, using the bentomodel file, type the following on your terminal to containerize it.

```bash
bentoml containerize flight_price_prediction:wqqm6cd7xcjtzzc6
```

Once it's containerize it, you can build the image using the following command on your terminal, remember to check the tag number for containerize it

```bash
docker run -it --rm -p 3000:3000 flight_price_prediction:wqqm6cd7xcjtzzc6 serve --production
```

Another way is to download the docker image from the repository in the [docker hub]()

First you need to download the docker image with the following command in the terminal

```bash
docker franciscoortiztena/flight_price_prediction
```

And then run the following command

```bash
docker run -it --rm -p 3000:3000 franciscoortiztena/flight_price_prediction serve --production
```

As the one in deploying, you can visit the [local host](http://localhost:3000/) to make the predictions

## Deployment using AWS
Finally if you want to deploy the model in the web, like AWS, follow the steps.
1. Check your [train.py](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/12_week_capstone_project/predict.ipynb) to contain the information that your model want to do.
2. Build your model with the following command on your terminal

```bash
bentoml build
```

3. Then you need to containerize it with the following command

```bash
bentoml containerize flight_price_prediction: --platform=linux/amd64
```
Remember to specify the platform, since it will made thing easy when dploying in AWS

4. Then on your [AWS](https://aws.amazon.com/es/) account you need to create first an Elastic Container Registry. Once is created you can use the push commands proposed in the AWS, remember to be logged in your terminal with your AWS credentials.
- Retrieve an authentication token and authenticate your Docker client to your registry
- Tag your image so you can push the image to this repository
- Push this images to your newly created AWS repository
5. Then you need to create an Elastic Container Service, creating a cluster using the networking only with AWS Fargate.
6. After that you need to create a new task definition in Fargate, choosing a task memory of 0.5GB and a task CPU of 0.25. Add the container, specifying the URI, a soft limit memory of 256MiB, mapping the port to 3000.
7. After that, return to the cluster and "Run new Task", Launching in Fargate, Operating System Family Linux, select the created task definition, in Cluster VPC select the default, Subnets the one with us-east-1a, in a security group configure your Custom TCP port 3000, and after crate the task will starting to run.
8. Into the task you will find the public IP, remember to specify the 3000 port at the end of the IP
9. Finally here is a video of how the model was dployed using AWS
