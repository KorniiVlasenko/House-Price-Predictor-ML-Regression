# This script contains:  
#   - models.pkl loading,
#   - processed data loading,
#   - prediction on data processed in EDA notebook,
#   - prediction on data processed in EDA notebook with different number of features,
#   - normalized data loading,
#   - prediction on normalized data
#   - prediction on scaled data
#   - creating all prediction files at once (general function)


import pandas as pd
import numpy as np
import pickle

# Load Models
def load_models():
    with open('models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    return models


# Load processed data (train and test)
def load_processed_data(train_processed_path, test_processed_path):
    # load and split train data
    train_data = pd.read_csv(train_processed_path, index_col = 'Id')
    X_processed = train_data.copy()
    y = X_processed['SalePrice']
    X_processed = X_processed.drop(['SalePrice'], axis = 1)

    # load test data
    test_data_processed = pd.read_csv(test_processed_path, index_col = 'Id')

    return X_processed, y, test_data_processed


# Get predictions on test data processed in EDA notebook
def get_prediction_processed(X_processed, y, test_data_processed, models):
    for model in models:
        # fit / predict
        model.fit(X_processed, y)
        prediction = model.predict(test_data_processed)
        
        # Get submission name
        cut_model_name_after = str(model).find('(')
        model_name = str(model)[:cut_model_name_after]
        submission_name = model_name + '_processed.csv'
        
        # Fit prediction to the submission format and save .csv
        submission_dataframe = pd.DataFrame({'Id': test_data_processed.index, 'SalePrice': prediction})
        submission_dataframe.to_csv(f"prediction_files/{submission_name}", index = False)


def get_prediction_diff_num_of_features(X_processed, y, test_data_processed, models):
    # Get a list of columns
    columns_list = list(X_processed.columns)

    # Create a list with best number of features for every model
    cut_list = [36, 55, 65, 41, 36, 36]


    for i in range(len(models)):
        # Get next model and corresponding num of features from cut_list
        model = models[i]
        num_features = cut_list[i] 
        
        # Shrinking the training data and fitting the model
        X_short = X_processed.copy()
        X_short = X_short[columns_list[:num_features]]
        model.fit(X_short, y)
        
        # Making prediction on cutted test data
        prediction_short = model.predict(test_data_processed[columns_list[:num_features]])
        
        # Get submission name
        cut_model_name_after = str(model).find('(')
        model_name = str(model)[:cut_model_name_after]
        submission_name = model_name + '_shorted.csv'
        
        # Fit prediction to the submission format and save .csv
        submission_dataframe = pd.DataFrame({'Id': test_data_processed.index, 'SalePrice': prediction_short})
        submission_dataframe.to_csv(f"prediction_files/{submission_name}", index = False)
        

# Load normalized data
def load_data_normalized(train_normalized_path, test_normalized_path):
    train_data_norm = pd.read_csv(train_normalized_path, index_col = 'Id')
    test_data_norm = pd.read_csv(test_normalized_path, index_col = 'Id')

    # Prepare training data
    X_norm = train_data_norm.copy()
    y_norm = X_norm['SalePrice']
    X_norm = X_norm.drop(['SalePrice'], axis = 1)

    return X_norm, y_norm, test_data_norm


# Prediction on normalized test data 
def get_prediction_normalized(X_norm, y_norm, test_data_norm, models):    
    # Only linear models will be used. They can benefit the best from normalized data
    linear_models = models[:3]

    columns_list = list(X_norm.columns)

    # We will use best number of features (see TryDifferentData notebook)
    cut_list_norm = [36, 55, 65]

    # Create prediction file for every model
    for i in range(len(linear_models)):
        # Get next model and corresponding num of features from cut_list_norm
        model = linear_models[i]
        num_features = cut_list_norm[i] 
        
        # Shrink normalized train data and fit the model
        X_norm_short = X_norm.copy()
        X_norm_short = X_norm_short[columns_list[:num_features]]
        model.fit(X_norm_short, y_norm)
        
        # The target value in the training data is also normalized, 
        # so it is necessary to take the exponent to get the correct prediction
        prediction_norm = np.exp(model.predict(test_data_norm[columns_list[:num_features]]))
        
        # Get submission name
        cut_model_name_after = str(model).find('(')
        model_name = str(model)[:cut_model_name_after]
        submission_name = model_name + '_norm.csv'
        
        # Fit prediction to the submission format and save .csv
        submission_dataframe = pd.DataFrame({'Id': test_data_norm.index, 'SalePrice': prediction_norm})
        submission_dataframe.to_csv(f"prediction_files/{submission_name}", index = False)


# Prediction on scaled test data 
def get_prediction_scaled(X_processed, y, test_data_processed, models):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # Scale data
    X_train_scaled = scaler.fit_transform(X_processed)
    test_data_scaled = scaler.transform(test_data_processed)

    for i in range(len(models)):
        model = models[i]
        model.fit(X_train_scaled, y)
        
        prediction_scaled = model.predict(test_data_scaled)
        
        # Get submission name
        cut_model_name_after = str(model).find('(')
        model_name = str(model)[:cut_model_name_after]
        submission_name = model_name + '_scaled.csv'
        
        # Fit prediction to the submission format and save .csv
        submission_dataframe = pd.DataFrame({'Id': test_data_processed.index, 'SalePrice': prediction_scaled})
        submission_dataframe.to_csv(f"prediction_files/{submission_name}", index = False)


# Get all prediction files by calling a single function
def get_all_predictions(train_processed_path, test_processed_path, train_normalized_path, test_normalized_path):
    models = load_models()
    X_processed, y, test_data_processed = load_processed_data(train_processed_path, test_processed_path)
    get_prediction_processed(X_processed, y, test_data_processed, models)
    get_prediction_diff_num_of_features(X_processed, y, test_data_processed, models)
    X_norm, y_norm, test_data_norm = load_data_normalized(train_normalized_path, test_normalized_path)
    get_prediction_normalized(X_norm, y_norm, test_data_norm, models)
    get_prediction_scaled(X_processed, y, test_data_processed, models)