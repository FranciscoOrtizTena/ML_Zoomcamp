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
