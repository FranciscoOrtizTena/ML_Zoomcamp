# Week 8 Midterm Project

# Machine Predictive Maintenance Classification Dataset

This dataset was found on [kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

## Problem description

Predictive maintenance technique are designed to help determine the condition of in-service equipment in order to estimate when maintenance should be performed. This approach promises cost savings over routine or time based preventive maintenance, because tasks are performed only when warranted. The idea with this project is to established a machine learning classfication algorithm to predict whether a machine is going to fail, considering the following features:
- UID: unique identifier ranging from 1 to 10,000.
- productID and type: consisting of a letter L, M or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specifc serial number.
- air temperature (K) generated using a random walk process later normalized to a standard deviation of 2 K around 300 K.
- process temperature (K) generated using a random walk process normalized to a standard deviation of 1 K, added to ther air temperatura plus 10 K.
- rotational speed (rpm) calculated from power of 2860 W, overlaid with a normally distributed noise.
- torque (Nm) torque values are normally distributed around 40 Nm with and lf= 10Nm and no negative values.
- tool wear (min) The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process and a machine failure label that indicates, whether the machine gas failed in this particular data point for any of the following failure modes are True.

Finally the target values are:
- Target: Failure or Not.
- Failure type: Type of failure.

## Problem approach

Since there is a classification problem where needs to predict if a machine is going to fail or not, an approach of Linear Regression, Decision Tree, Random Forest and XGBoost is proposed.

## Instructions on how to run the project

### Data

The data is also founded in the [data](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/8_week/predictive_maintenance.csv)

### Files

#### Training

A) The first file is the [notebook](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/8_week/notebook.ipynb) where the data preparation, data cleaning, exploratory data analysis, feature importances, model selection process and parameter tuning is done.

0. Loaded libraries: Numpy and pandas for numerical processing; matplotlib and seaborn for making graphs; LogisticRegression, DecisionTreeClassifier, RandomForest and XGBoost are the models proposed to predict if a machine is propense to fail; train_test_split, DictVectorizer, cross_val_predict and KFold for prepoced data; consufion_matrix, accuracy_score, roc_auc_score, mutual_info_score, precision_score, recall_score, precision_recall_curve and roc_curve as metrics to evaluate the different models and finally tqdm for take time of the loops.

1. After that, the data is downloaded from the specific link.

2. The data preparation and data cleaning is performed, the names in the columns are transformed to lowercase and without a blank space. Here I identify that two columns are useless: the udi and product_id, since those numbers are use it to identify the particular machine and also identified if there are some outliers or nulls values in the data.

3. Exploratory Data Analysis: I started checking information about the categorical data, which in this case was just the type of the machine and for the numerical data I made some histograms and scatter plots to try to find some relations between the features. Here I found that the most common correlation was the torque with the rotational speed, when a machine had low torque and high rotational speed, the machine was more propensed to fail and viceverse, when there was a high torque with low rotational speed the machine was failing.

4. Model selection process and parameter tuning: here I trained several models (LogisticRegression, DecisionTree, RandomForest and XGBoost) to try to find the best performance using the roc_auc_curve value. First the validation framework was created, where the data was splitted into train, validation and test and the target value was passed to another variable a dropped from the original data, then the preprocessing was done using the DictVectorizer to make a one-hot encode for the categorical variable of type. Finally the four models were trained and tuned in the following parameters:
4.1.- For the LogisticRegression the C parameter of regularization was tuned, it was found with a C=15 but the roc_auc_score was only 89.36
4.2.- For the DecisionTree the max_depth and min_sample_leaf were tuned, it was found that the best model fits with max_depth = 8 and a min_sample_leaf = 50, with roc_auc_score = 97%.
4.3.- For the RandomForest the n_estimators, max_depth and min_samples_leaf were tunned with the following best parameters n_estimators=200, max_depth=10, min_samples_leaf=25, with roc_auc_score = 96.43
4.4.- For the XGBoost the eta, max_depth and min_child_weight was tunned with the following best parameters eta= 0.3, max_depth = 3 and min_child_weight = 10 with a roc_auc_score = 97.33%

5.- The models were compared, where XGboost was the one with the best roc_auc_score

6.- The final model was trained with the full data train and compared with the test data, the roc_auc_score was better with a 98.24% of roc_auc_score.

#### Deployment locally

B) The second file is the [build_bento_model_maintenance.ipynb](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/8_week/build_bento_model_maintenance.ipynb) file, here the best model, which was the XGBoost model, is trained again and saved into a bento model to deploy it.

Once the Bentomodel is created, the [train.py](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/8_week/train.py) is elaborated whit the script to deploy it locally using the bento interface. You can run it locally using the following command in the terminal

```bash
bentoml serve train.py:svc --production
```

But be careful to specify the correctl tag of the model in the script, since if you run it from your computer the tag may change.

You can visit the [local host](http://0.0.0.0:3000/) to make predictions

Another option to run it locally is to export the model using the following methodology.

```bash
bentoml models export maintenance_predict_model:asw4gns4q2vxgjv5
```

And will create the following [bentomodel](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/8_week/maintenance_predict_model-asw4gns4q2vxgjv5.bentomodel?raw=true) file, you can download this file and the use the following command in your terminal to import it.

```bash
bentoml models import maintenance_predict_model-asw4gns4q2vxgjv5.bentomodel
```

c) Finally, you can use the third file [predict.ipynb](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/8_week/predict.ipynb) to load the bentomodel and predict if a machine is prone to fail or not. Remember you need to pass a dictionary as follows:

{"type": str,
 "air_temperature_[k]": float,
 "process_temperature_[k]": float,
 "rotational_speed_[rpm]": int,
 "torque_[nm]": float,
 "tool_wear_[min]": int}
 
 Here is an example
 
 {"type": "L",
  "air_temperature_[k]": 298.0,
  "process_temperature_[k]": 308.7,
  "rotational_speed_[rpm]": 1268,
  "torque_[nm]": 69.4,
  "tool_wear_[min]": 189}
  
#### Deployment using Docker
  
To deploy your model using the Docker images, you need first to containerize the previous model, if you use the bentomodel file, type the following on your terminal to containerize it.


```bash
bentoml containerize maintenance_predict_classifier:fjxpm3s4soefsjv5
```

Once it's containerize it you can build the image using the following command on your terminal.

```bash
docker run -it --rm -p 3000:3000 maintenance_predict_classifier:fjxpm3s4soefsjv5 serve --production
```

As the one in deploying you can visit the [local host](http://0.0.0.0:3000/) to make the predictions
