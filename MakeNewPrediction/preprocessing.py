# This script contains:  
#   - data loading,
#   - preprocessing of train and test data (see more in the EDA notebook):
#                                           * features selection,
#                                           * missing values processing,
#                                           * outliers excluding,
#                                           * categorical variables encoding.
#   - saving different data variants for model selection (see more in the TryDifferentData notebook)



# Set aliases
import pandas as pd
import numpy as np

# Data Loading
def data_loading(train_data_path = '../Data/train_origin.csv', 
                 test_data_path = '../Data/test_origin.csv'):
    # load data fron .csv files
    train_data = pd.read_csv(train_data_path, index_col = 'Id')
    test_data = pd.read_csv(test_data_path, index_col = 'Id')

    # separate target variable from other features
    X = train_data.copy()
    y = X['SalePrice']
    X = X.drop(['SalePrice'], axis = 1)

    return X, y, test_data


# Features Selection
def features_selection(X, test_data):
    # These features were selected based on data visualization and mutual information score (see EDA notebook)
    features_to_keep = ['OverallQual', 'Neighborhood', 'GrLivArea', 'YearBuilt', 'GarageArea', 'TotalBsmtSF', 'FullBath',
                         'YearRemodAdd', '2ndFlrSF', 'Foundation', 'Exterior2nd', 'LotArea', 'Fireplaces', 'OpenPorchSF']
    X = X[features_to_keep]
    test_data = test_data[features_to_keep]
    return X, test_data


# Missing Values Processing
def missing_processing(test_data):    # Only test data will be processing, because train data doesn't have missing data in selected rows
    # The specifics of my test data and its evaluation do not allow me to throw out rows, 
    # so any missing data will be filled in. (see more in EDA notebook)

    from sklearn.impute import SimpleImputer

    # Get numerical and categorical features into separate variables
    num_cols = ['TotalBsmtSF', 'GarageArea', '2ndFlrSF', 'TotalBsmtSF', 'LotArea', 'OpenPorchSF']

    cat_cols = ['OverallQual', 'Neighborhood', 'YearBuilt', 'FullBath',
                             'YearRemodAdd', 'Foundation', 'Exterior2nd', 'Fireplaces']

    # Create imputer objects
    num_imputer = SimpleImputer(strategy = 'median')
    cat_imputer = SimpleImputer(strategy = 'most_frequent')

    # Fill in the missing data
    test_data[num_cols] = num_imputer.fit_transform(test_data[num_cols])
    test_data[cat_cols] = cat_imputer.fit_transform(test_data[cat_cols])
    return test_data


# Ouliers Excluding
def outliers_excluding(X, y):
    # Outliers in numerical columns
    # The outliers I exclude next were found using visualization in the EDA notebook
    
    # Exclude biggest values
    indices_to_remove = X['GrLivArea'].nlargest(4).index
    X = X.drop(indices_to_remove)
    indices_to_remove = X['GarageArea'].nlargest(3).index
    X = X.drop(indices_to_remove)
    indices_to_remove = X['LotArea'].nlargest(16).index
    X = X.drop(indices_to_remove)
    indices_to_remove = X['OpenPorchSF'].nlargest(6).index
    X = X.drop(indices_to_remove)

    # Make sure the indices match
    y = y[X.index]


    # Outliers in categorical columns. I will do this relying on 1.5 IQR rule

    # Concat features with SalePrice 
    X_y = pd.concat([X, y], axis=1)

    # Create a list where outliers indices will be stored
    outliers_indices = []

    # Create a list of categorical variables
    cat_cols = ['OverallQual', 'Neighborhood', 'YearBuilt', 'FullBath',
                'YearRemodAdd', 'Foundation', 'Exterior2nd', 'Fireplaces']

    # Run through every categoty of every categorical feature
    for col in cat_cols:
        for category in X[col].unique():
            
            # Calculate the Interquartile Range for every category
            quartile1, quartile3 = np.percentile(X_y[X_y[col] == category]['SalePrice'], [25, 75])
            iqr = quartile3 - quartile1

            # Define the boundaries beyond which the value will be considered an outlier.
            low_border = quartile1 - 1.5 * iqr
            high_border = quartile3 + 1.5 * iqr
            
            # Get outliers indices of a particular category
            outliers_low_price = list(X_y[(X_y[col] == category) & (X_y['SalePrice'] <= low_border)].index)
            outliers_high_price = list(X_y[(X_y[col] == category) & (X_y['SalePrice'] >= high_border)].index)
            
            # Add indices to the outliers_indices
            outliers_indices += outliers_low_price + outliers_high_price
    
    if outliers_indices:
        # Only one appearance of every outlier index is necessary
        outliers_indices = set(outliers_indices)
        
        # Check if any outliers are already present in X or y
        outliers_indices = [idx for idx in outliers_indices if idx in X.index or idx in y.index]
        
        # Drop the outliers from X
        X = X.drop(index=outliers_indices)
                
        # Make sure than X and y indices match
        y = y[X.index]   
    
    
    return X, y



# Categorical Encoding
def categorical_encoding(X, test_data):
    from sklearn.preprocessing import OneHotEncoder

    # List of columns that have to be encoded
    cat_cols_to_encode = ['Neighborhood', 'Foundation', 'Exterior2nd']

    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X[cat_cols_to_encode]))
    OH_cols_test = pd.DataFrame(OH_encoder.transform(test_data[cat_cols_to_encode]))

    # Get feature names after one-hot encoding
    OH_feature_names = OH_encoder.get_feature_names_out(cat_cols_to_encode)

    # Set column names for one-hot encoded columns
    OH_cols_train.columns = OH_feature_names
    OH_cols_test.columns = OH_feature_names

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X.index
    OH_cols_test.index = test_data.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X.drop(cat_cols_to_encode, axis=1)
    num_X_test = test_data.drop(cat_cols_to_encode, axis=1)

    # Add one-hot encoded columns to numerical features
    X_encoded = pd.concat([num_X_train, OH_cols_train], axis=1)
    test_data_encoded = pd.concat([num_X_test, OH_cols_test], axis=1)

    # Ensure all columns have string type
    X_encoded.columns = X_encoded.columns.astype(str)
    test_data_encoded.columns = test_data_encoded.columns.astype(str)
    
    return X_encoded, test_data_encoded


# Save train / test data 
def save_processed_data(X, y, test_data):
    # Save train data
    train_data_processed = pd.concat([X, y.rename('SalePrice')], axis = 1)
    train_data_processed.to_csv('../Data/train_data_processed.csv')

    # Save test data
    test_data.to_csv('../Data/test_data_processed.csv')


# Save normalized data
def save_normalized_data(X, y, test_data):
    # Concat X and y
    train_data_stats = pd.concat([X, y.rename('SalePrice')], axis = 1)

    # Normilize columns using np.log function. Normalizing just these columns is justified in the EDA notebook.
    train_data_stats['SalePrice'] = np.log(train_data_stats['SalePrice'])
    train_data_stats['GrLivArea'] = np.log(train_data_stats['GrLivArea'])
    train_data_stats.loc[train_data_stats['TotalBsmtSF'] > 0, 'TotalBsmtSF'] = np.log(train_data_stats['TotalBsmtSF'])

    # Do the same for test data.
    test_data['GrLivArea'] = np.log(test_data['GrLivArea'])
    test_data.loc[test_data['TotalBsmtSF'] > 0, 'TotalBsmtSF'] = np.log(test_data['TotalBsmtSF'])

    # Saving normalized data
    train_data_stats.to_csv('../Data/train_data_stats.csv')
    test_data.to_csv('../Data/test_data_stats.csv')


def full_preprocessing(train_data_path = '../Data/train_origin.csv', 
                       test_data_path = '../Data/test_origin.csv'):
    X, y, test_data = data_loading(train_data_path, test_data_path)
    X, test_data = features_selection(X, test_data)
    test_data = missing_processing(test_data)
    X, y = outliers_excluding(X, y)
    X, test_data = categorical_encoding(X, test_data)
    save_processed_data(X, y, test_data)
    save_normalized_data(X, y, test_data)




