# Predicting of House Prices    


The project solves a regression problem - predicting house prices. 

It is built using pandas, numpy, scikit-learn and xgboost libraries.    

The data was taken from the Kaggle database [data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)  


### Goal of this project

The purpose of this project is educational. Specifically, I aimed to improve skills in areas such as: EDA, model selection and tuning, scripts writing, creative thinking in problem solving.       

### Table of Contents<a name = 'content'></a>     

- [Development](#developmnt)    
- [Usage](#usage)      
- [Project Navigation](#navigation)    
   
    
### Development<a name = 'development'></a>   
[Table of Contents](#content)     

To install all required dependencies, run the `requirements.txt` file from the root directory of the project:    

> pip install -r requirements.txt

To build a project run the following command while in the root directory:    

> python Scripts/make_project.py


### Usage<a name = 'usage'></a>     
[Table of Contents](#content)     

If you are that lucky guy who has real estate in Ames, Iowa, and you want to find out its value, you can do it as follows:    

1. In the MakeNewPrediction folder, write your property information into a *make_prediction* file similar to *make_prediction_example*. You can take inspiration for feature values from the *Domuntetion/data_description.txt* file.

2. In the same folder, run the command:
> python make_new_prediction

3. Look up the value of your properties in the **price_of_your_property** file that will appear afterwards.


### Project Navigation<a name = 'navigation'></a>      
[Table of Contents](#content)     

The easiest to grasp part of the project is in the **Notebooks** folder. This folder contains Jupyter Notebooks with a lot of comments, deliberations and code explanations.    

The first notebook in chronology is *Notebooks/Exploratory/ExploratoryAnalysis.ipynb*. This contains all the preprocessing of the data. (loading data, handling missing values, selecting features, etc.) This is also where I save a few variations of the data to the Data folder.     
*The script for this notebook: Scripts/preprocessing.py*    

The next notebook is *Notebooks/Modeling/FittingAndEvaluation.ipynb*. In this notebook, I find the best hyperparameters for different models and save the results.     
*The script for this notebook: Scripts/models_tuning_and_evaluation.py*

The last notebook is *Notebooks/Modeling/TryDifferentData.ipynb*. In this notebook, I train the models on different data and get real prediction scores using Kaggle submissions.     
*The script for this notebook: Scripts/models_prediction_on_different_data.py*

A description of the original data can be found in the **Documentation** folder.



Enjoy coding!
