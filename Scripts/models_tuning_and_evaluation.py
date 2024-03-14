# This script contains:  
#   - prepared data loading,
#   - defining custom scoring metric,
#   - training and tuning of hyper parameters of such models as (see more in the FittingAndEvaluation notebook):
#                                           * Linear Regression,
#                                           * Ridge Regression,
#                                           * Lasso Regression,
#                                           * Elastic Net,
#                                           * Decision Tree Regressor,
#                                           * Random Forest Regressor, 
#                                           * XGBoost Regressor.
#   - saving models trained on data processed in EDA notebook

# Set aliases
import pandas as pd
import numpy as np

# Load the processed data
def load_processed_data(processed_data_path = '../Data/train_data_processed.csv'):
    data = pd.read_csv(processed_data_path, index_col = 'Id')
    X = data.copy()  
    y = X['SalePrice']
    X = X.drop(['SalePrice'], axis = 1)
    return X, y


# Define a custom scoring method. (See FittingAndEvaluation notebook why this scorer was choosen)

# My metric for cross validation
def rmse_log(model, X, y):
    from sklearn.metrics import mean_squared_error

    y_pred = model.predict(X)
    return np.sqrt(mean_squared_error(np.log1p(y), np.log1p(y_pred)))

# My metric for GridSearch
def rmse_log_for_gridsearch(y_true, y_pred):
    from sklearn.metrics import mean_squared_error

    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

from sklearn.metrics import make_scorer
custom_scorer = make_scorer(rmse_log_for_gridsearch, greater_is_better=False)


# Linear Regression
def linear_regression_fit(X, y):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    linear_regression = LinearRegression()

    # Get cross-validation score
    linear_regression_scores = cross_val_score(linear_regression,
                            X,
                            y,
                            cv = 5,
                            scoring = rmse_log)

    print('Linear Regression score: ', linear_regression_scores.mean())
    
    return linear_regression


# Ridge Regression
# All default parameter values were found by experimentation in the FittingAndEvaluation notebook
def ridge_regression_tune_fit(X, y, alpha = [2.641]):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV

    # Create a model sample
    ridge_sample = Ridge()

    # Set the search area for GridSearch
    ridge_hyper_params = {'alpha': alpha, 'random_state': [0]}

    # Create a GridSearch sample and fit the model
    ridge_regression = GridSearchCV(ridge_sample, ridge_hyper_params, scoring = custom_scorer, cv = 5)
    ridge_regression.fit(X, y)

    # Print best parameters and best score
    print('Ridge Regression, best value of λ: ', ridge_regression.best_params_)
    print('Best score: ', -1 * ridge_regression.best_score_)

    # Give model best parameters
    ridge_best_params = ridge_regression.best_params_
    ridge_regression = Ridge(**ridge_best_params)

    return ridge_regression



# Lasso Regression 
# All default parameter values were found by experimentation in the FittingAndEvaluation notebook
def lasso_regression_tune_fit(X, y, alpha = [41]):
    from sklearn.linear_model import Lasso
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import GridSearchCV

    # I don't want to overload the output of the Lasso regression
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Create a model sample
    lasso_sample = Lasso()

    # Set the search area for GridSearch
    lasso_hyper_params = {'alpha': alpha, 'random_state': [0]}
    
    # Create a GridSearch sample and fit the model
    lasso_regression = GridSearchCV(lasso_sample, lasso_hyper_params, scoring = custom_scorer, cv = 5)
    lasso_regression.fit(X, y)

    # Print best parameters and best score
    print('Lasso Regression, best value of alpha: ', lasso_regression.best_params_)
    print('Best score: ', -1 * lasso_regression.best_score_)

    # Give model best parameters
    lasso_best_params = lasso_regression.best_params_
    lasso_regression = Lasso(**lasso_best_params)

    return lasso_regression


# Elastic Net  
# All default parameter values were found by experimentation in the FittingAndEvaluation notebook
def elastic_net_tune_fit(X, y, alpha = [41], l1_ratio = [1]):
    from sklearn.linear_model import ElasticNet
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import GridSearchCV

    # I don't want to overload the output of the Elastic Net
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    # Create a model sample
    elastic_net_sample = ElasticNet()

    # Set the search area for GridSearch
    elnet_hyper_params = {'alpha': alpha, 'l1_ratio': l1_ratio, 'random_state': [0]}

    # Create a GridSearch sample and fit the model
    elastic_net = GridSearchCV(elastic_net_sample, elnet_hyper_params, scoring = custom_scorer, cv = 5)
    elastic_net.fit(X, y)

    # Print best parameters and best score
    print('Elastic Net, best alpha and l1_ratio: ', elastic_net.best_params_)
    print('Best score: ', -1 * elastic_net.best_score_)

    # Give model best parameters
    elnet_best_params = elastic_net.best_params_
    elastic_net = ElasticNet(**elnet_best_params)

    return elastic_net


# DecisionTree   
# All default parameter values were found by experimentation in the FittingAndEvaluation notebook
def decision_tree_tune_fit(X, y, max_depth = [6], min_samples_split = [2], min_samples_leaf = [6], max_features = [35], 
                           min_impurity_decrease = [0], ccp_alpha = [0]):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV

    # Create a model sample
    decision_tree_sample = DecisionTreeRegressor()

    # Set the search area for GridSearch
    decision_tree_hyper_params = {'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'max_features': max_features,
                                'random_state': [0],
                                'min_impurity_decrease': min_impurity_decrease,
                                'ccp_alpha': ccp_alpha
                                }
    
    # Create a GridSearch sample and fit the model
    decision_tree_regressor = GridSearchCV(decision_tree_sample, decision_tree_hyper_params, 
                                        scoring = custom_scorer, cv = 5)
    decision_tree_regressor.fit(X, y)

    # Print best parameters and best score
    print('Decision Tree, best parameters: ', decision_tree_regressor.best_params_)
    print('Best score: ', -1 * decision_tree_regressor.best_score_)

    # Give model best parameters
    decision_tree_best_params = decision_tree_regressor.best_params_
    decision_tree_regressor = DecisionTreeRegressor(**decision_tree_best_params)

    return decision_tree_regressor


# Random Forest  
# All default parameter values were found by experimentation in the FittingAndEvaluation notebook 
def random_forest_tune_fit(X, y, n_estimators = [1150], max_depth = [27], min_samples_split = [3], 
                           min_samples_leaf = [1], max_features = [12]):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    # Create a model sample
    random_forest_sample = RandomForestRegressor()

    # Set the search area for GridSearch
    random_forest_hyper_params = {'n_estimators': n_estimators,
                                'max_depth': max_depth, 
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'max_features': max_features,
                                'random_state': [0],
                                'n_jobs': [-1]
                                }
    
    # Create a GridSearch sample and fit the model
    random_forest_regressor = GridSearchCV(random_forest_sample, random_forest_hyper_params, 
                                        scoring = custom_scorer, cv = 5)
    random_forest_regressor.fit(X, y)

    # Print best parameters and best score
    print('Random Forest, best parameters: ', random_forest_regressor.best_params_)
    print('Best score: ', -1 * random_forest_regressor.best_score_)

    # Give model best parameters
    random_forest_best_params = random_forest_regressor.best_params_
    random_forest_regressor = RandomForestRegressor(**random_forest_best_params)

    return random_forest_regressor


# Extreme Gradient Boosting  
# All default parameter values were found by experimentation in the FittingAndEvaluation notebook
def xgboost_tune_fit(X, y, max_depth = [3], min_child_weight = [7], gamma = [0], subsample = [1], colsample_bytree = [1],
                     reg_alpha = [0], reg_lambda = [1]):
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split

    # Split the data to use eval_set in selecting the best values for n_estimators and learning_rate
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Find the best values for n_estimators and learning_rate
    # The best learning_rate was found through experimentation and multiple code executions. Here you can only see the result
    xgb_regressor_presearch = XGBRegressor(n_estimators = 1000, learning_rate = 0.25, random_state = 0) 
    xgb_regressor_presearch.fit(X_train, y_train,
                    early_stopping_rounds = 100,
                    eval_set = [(X_valid, y_valid)],
                    verbose = False)

    print("XGBoost, best value for n_estimators: ", xgb_regressor_presearch.best_iteration)

    # Create features for best iteretion and best learning rate
    xgb_best_iteration = xgb_regressor_presearch.best_iteration
    xgb_best_learning_rate = 0.25

    # Find the best values for all other parameters using GridSearchCV
    # All the values were found through experimentation and multiple code executions. Here you can only see the result
    xgb_regressor_sample = XGBRegressor()
    xgb_hyper_params = {'n_estimators': [xgb_best_iteration],
                        'learning_rate': [xgb_best_learning_rate],
                        'max_depth': max_depth, 
                        'min_child_weight': min_child_weight,  
                        'gamma': gamma,
                        'subsample': subsample,
                        'colsample_bytree': colsample_bytree,
                        'reg_alpha': reg_alpha,
                        'reg_lambda': reg_lambda,
                        'random_state': [0]
                    }

    # Create a GridSearch sample and fit the model
    xgb_regressor = GridSearchCV(xgb_regressor_sample, xgb_hyper_params, scoring = custom_scorer, cv = 10)
    xgb_regressor.fit(X, y)

    # Give model best parameters
    print('XGBoost, best parameters: ', xgb_regressor.best_params_)
    print('Best score: ', -1 * xgb_regressor.best_score_)

    # Give model best parameters
    xgb_best_params = xgb_regressor.best_params_
    xgb_regressor = XGBRegressor(**xgb_best_params)

    return xgb_regressor




# Save models
def save_tuned_models():
    import pickle

    X, y = load_processed_data()

    # Create a list for storing models
    models = []

    # Add linear regression
    models.append(linear_regression_fit(X, y))

    # Add other models. (I exclude elastic net, because in my case it's equal to lasso regression)
    for function in (ridge_regression_tune_fit, lasso_regression_tune_fit, decision_tree_tune_fit,
                random_forest_tune_fit, xgboost_tune_fit):
        new_model = function(X, y)
        models.append(new_model)
    
    # Check that right hyper parameters were saved
    print(models)
    
    # Save models to a file
    with open('models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    return models
    




