# Comparison-of-ML-model-s-stability-on-medical-data
This repository is used for project about ML model’s stability in medical data. The leave-one-out concept is used to analyze the behavior for 5 models(Logistic Regression, Decision Tree, Random Forest, Bagging,CatBoost) towards removal of certain percent of data from the dataset. Stability of the model is observed by deriving generalization error. 

Results are named according to sections.
4.1	Stability of models
4.1.1	Diabetes data
4.1.2	Heart disease or attack data
4.2	Parameters optimization influence on stability
4.3	Random state change influence on stability
4.3.1	Diabetes data
4.3.2	Heart disease or attack data
4.4	Cross validation change influence on stability
4.4.1	Diabetes data
4.4.2	Heart disease or attack data
5. Stability test on Wisconsin breast cancer dataset

The main method, which is used in all sections except 4.3:
At first, we find the Best model for the dataset by putting hyperparameters and using GridSearch. Here we use 0.2 of sample for train and 5-fold cross validation for each model. Then in each step we delete 0.07 percent of dataset randomly. Randomness is mandatory to be sure that there are no logical connections between the removed observations are independent. In the second step we delete 0.07 percent from the main dataset and evaluating accuracy with already chosen best model. Then, without changing the main dataset we delete 1.14 percent from it and try the best model on that, this loop is working until 91% of the data is deleted. After this we collect all the information about the removed observations and accuracy in one dataset. In 4.2 section we also use optimization in each step and finding best model in each step.

The method, which is used in section 4.3:
In each step we will remove the same percent of data, but we are not using the same random. For the same amount of percent we remove observations 10 times, but using different random seeds according to steps number. Here we will have 10 times more points for each step and all of them will represent different accuracy. 
