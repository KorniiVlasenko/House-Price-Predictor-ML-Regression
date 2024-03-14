# This script contains:  
#   - loading best model from models.pkl,
#   - loading data to fit best model and make prediction,
#   - make a file with the prediction

# If you are that lucky guy who has real estate in Ames, Iowa, 
# and you want to know its value, you can use this program))))))




import pandas as pd
import numpy as np
import pickle

# Load best model from models.pkl and write best_model.pkl
def load_best_model():
    # Load all models
    with open('../Scripts/models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    # Get best model from models list
    best_model = models[2]

    return best_model


# Load and processed data (train and new)
def load_and_process_data(train_origin_path = '../Data/train_origin.csv', 
                            new_data_path = 'make_prediction.csv'):
    from sklearn.preprocessing import StandardScaler
    import preprocessing as pp

    # load and split train data
    train_data = pd.read_csv(train_origin_path, index_col = 'Id')
    X = train_data.copy()
    y = X['SalePrice']
    X = X.drop(['SalePrice'], axis = 1)

    # load new data
    new_data = pd.read_csv(new_data_path, index_col = 'Id')

    # Make features selection
    X_selected, new_data_selected = pp.features_selection(X, new_data)

    # Missing Values Processing
    new_data_no_missing = pp.missing_processing(new_data_selected)

    # Outliers excluding
    X_no_outliers, y = pp.outliers_excluding(X_selected, y)

    # Categorical encoding
    X_encoded, new_data_encoded = pp.categorical_encoding(X_no_outliers, new_data_no_missing)

    # scale train and new data
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_encoded)
    new_data_scaled = scaler.transform(new_data_encoded)

    return X_scaled, y, new_data_scaled


# Get predictions on test data processed in EDA notebook
def new_prediction(X_scaled, y, new_data_scaled, best_model):
    # fit / predict
    best_model.fit(X_scaled, y)
    prediction = best_model.predict(new_data_scaled)
    
    # Round prediction to two decimal places
    rounded_prediction = np.round(prediction, 2)
    
    # Add dollar sign to rounded prediction
    formatted_prediction = ['$' + str(p) for p in rounded_prediction]

    # Get prediction name
    prediction_name = 'price_of_your_property.csv'
        
    # Fit prediction to the submission format and save .csv
    prediction_dataframe = pd.DataFrame({'Id': [i for i in range(1, len(new_data_scaled) + 1)], 
                                         'SalePrice': formatted_prediction})
    prediction_dataframe.to_csv(f"{prediction_name}", index = False)


# Make prediction file using functions written above
best_model = load_best_model()
X_scaled, y, new_data_scaled = load_and_process_data()
new_prediction(X_scaled, y, new_data_scaled, best_model)
