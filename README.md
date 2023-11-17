# 8663_Churning_Customers

https://studio.youtube.com/video/MvInaRVPVB8/edit 
The above is the link to a video that displays how the application is used

This project is based on developing a model to predict the likelihood that a customer would churn given various factors. 

The project begins by loading the dataset
Afterwards I run the dataset to see all the columns and to check for null values and to see if there would be a need to encode the data
My data pre-processing steps included:
Feature importance to get a hold of the relevant columns that i would be using to develop my model
Encoding the data to make sure all features are numerical, thus my model prediction will be unbiased
Futhermore, exploratory data analysis was used to display the relationship between the independent variable and the dependent variables

After pre-processing the data i went on to start training and testing my model
I made use of the multi-layer perceptron model with the help of functional API's which were imported from the keras libraries and with cross validation and GridSearchCV
The model is configured with specified layers, activations, and dropout rates.

The model was initially evaluated using the accuracy score
Then after training i went on to test the data, over here i utilised the AUC score and the accuracy score after making changes to the model to get a better evaluation

Finally i saved the best model and used that for my deployment.
