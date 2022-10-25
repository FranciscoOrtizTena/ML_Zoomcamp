# Week 8 Midterm Project

# Machine Predictive Maintenance Classification Dataset

This dataset was found on [kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

## Problem description

Predictive maintenance technique are designed to help determine the condition of in-service equipment in order to estimate when maintenance should be performed. This approach promises cost savings over routine or time based preventive maintenance, because tasks are performed only when warranted. The idea with this project is to established a machine learning classfication algorithm to predict whether a machine is going to fail, considering the following features:
- UID: unique identifier ranging from 1 to 10,000.
- productID: consisting of a letter L, M or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specifc serial number.
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

### Exploratory Data Analysis
