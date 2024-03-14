# This script contains:  
#   - loading best model from models.pkl,
#   - loading data to fit best model and make prediction,
#   - make a file with best prediction
#   - general function that run all previous functions at once

# For why this particular model trained on this particular data is the best, see the TryDifferentData notebook


import pandas as pd
import numpy as np
import pickle

# Load best model from models.pkl and write best_model.pkl
def load_best_model():
    # Load all models
    with open('models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    # Get best model from models list
    best_model = models[2]

    # Save best model as a separate pkl file
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    return best_model


# Load processed data (train and test)
def load_and_scale_processed_data(train_processed_path = '../Data/train_data_processed.csv', 
                                  test_processed_path = '../Data/test_data_processed.csv'):
    from sklearn.preprocessing import StandardScaler

    # load and split train data
    train_data = pd.read_csv(train_processed_path, index_col = 'Id')
    X_processed = train_data.copy()
    y = X_processed['SalePrice']
    X_processed = X_processed.drop(['SalePrice'], axis = 1)

    # load test data
    test_data_processed = pd.read_csv(test_processed_path, index_col = 'Id')

    # scale train and test data
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_processed)
    test_data_scaled = scaler.transform(test_data_processed)

    return X_scaled, y, test_data_scaled


# Get predictions on test data processed in EDA notebook
def best_model_fit_predict(X_scaled, y, test_data_scaled, best_model,
                           test_processed_path = '../Data/test_data_processed.csv'):
    # Get test data to use it's indexes
    test_data_processed = pd.read_csv(test_processed_path)

    # fit / predict
    best_model.fit(X_scaled, y)
    prediction = best_model.predict(test_data_scaled)
        
    # Get submission name
    submission_name = 'best_prediction.csv'
        
    # Fit prediction to the submission format and save .csv
    submission_dataframe = pd.DataFrame({'Id': test_data_processed.index, 'SalePrice': prediction})
    submission_dataframe.to_csv(f"prediction_files/best_prediction/{submission_name}", index = False)


# General function to do all this at once
def make_best_prediction():
    best_model = load_best_model()
    X_scaled, y, test_data_scaled = load_and_scale_processed_data()
    best_model_fit_predict(X_scaled, y, test_data_scaled, best_model)
