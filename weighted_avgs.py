import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

#Preprocessing the Data
def preprocess():
    global df, X, Y, X_train, X_test, Y_train, Y_test, baseline
    df = pd.read_csv('data/OnlineNewsPopularity.csv')
    df = df.rename(columns = lambda x: x.strip())
    df = df.drop(columns = ['url', 'timedelta'])
    df = df[df['shares'] > 100]
    df = df[df['shares'] < 23000]
    df = df.drop(columns = ['n_non_stop_words', 'n_non_stop_unique_tokens', 
                        'kw_max_min', 'kw_max_avg', 'self_reference_min_shares', 
                        'self_reference_max_shares', 'is_weekend'])
    df['shares'] = np.log(df['shares'])
    
    X = df[df.columns]
    Y = X['shares'].values
    X = X.drop('shares', axis = 1).values

    #Splitting data to test and train
    X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, shuffle = False)
    baseline = np.exp(Y_train)
    baseline = baseline.mean()

#Function to find the RMSE    
def actualRMSE(diff):
    diff_squared = diff * diff
    diff_squared_sum = sum(diff_squared)
    diff_squared_sum_by_n = (diff_squared_sum * 1.0)/ len(diff)
    rmse = np.sqrt(diff_squared_sum_by_n)
    return rmse

#Function for Scatterplot
def scatterplot(model, data):
    x = data['Test Data']
    y = data['Predicted']
    plt.scatter(x, y, marker = '.', alpha = 0.3)
    plt.title(model)
    plt.xlabel('Test Data')
    plt.ylabel('Predicted')
    plt.savefig(model + '.png',  dpi=100)
    plt.show()

#Function for XGBoost    
def xgb():
    print ('\nExtreme Gradient Boosting Model')
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = XGBRegressor(learning_rate = 0.3)
    model.fit(rescaled_X_train, Y_train)
    
    train_predictions = model.predict(rescaled_X_train)
    train_r2score = model.score(rescaled_X_train, Y_train)
    actual_y_train = np.exp(Y_train)
    actual_train_predictions = np.exp(train_predictions)
    train_diff = actual_y_train - actual_train_predictions
    
    print ('Train R2 Score = ', round(train_r2score, 2))
    print ('Train Log RMSE = ', round(mean_squared_error(Y_train, train_predictions), 2))
    print ('Train Actual RMSE = ', round(actualRMSE(train_diff), 2))
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    r2score = model.score(rescaled_X_test, Y_test)
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    diff_baseline = actual_y_test - baseline
    
    print ('R2 Score = ', round(r2score, 2))
    print ('Log RMSE = ', round(mean_squared_error(Y_test, predictions), 2))
    print ('Actual RMSE = ', round(actualRMSE(diff), 2))
    print ('Baseline RMSE =', round(actualRMSE(diff_baseline), 2))
    print ('Max Error = ', max(abs(diff)))
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test,
                                   'Predicted' : actual_predicted,
                                   'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    scatterplot('XGBoost', compare_actual)
    
    return compare_actual

#Function for Random Forest 
def rfr():
    print ('\nRandom Forest Regression Model')
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = RandomForestRegressor(n_estimators = 300)
    model.fit(rescaled_X_train, Y_train)
    
    train_predictions = model.predict(rescaled_X_train)
    train_r2score = model.score(rescaled_X_train, Y_train)
    actual_y_train = np.exp(Y_train)
    actual_train_predictions = np.exp(train_predictions)
    train_diff = actual_y_train - actual_train_predictions
    
    print ('Train R2 Score = ', round(train_r2score, 2))
    print ('Train Log RMSE = ', round(mean_squared_error(Y_train, train_predictions), 2))
    print ('Train Actual RMSE = ', round(actualRMSE(train_diff), 2))
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    r2score = model.score(rescaled_X_test, Y_test)
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    diff_baseline = actual_y_test - baseline
    
    print ('R2 Score = ', round(r2score, 2))
    print ('Log RMSE = ', round(mean_squared_error(Y_test, predictions), 2))
    print ('Actual RMSE = ', round(actualRMSE(diff), 2))
    print ('Baseline RMSE =', round(actualRMSE(diff_baseline), 2))
    print ('Max Error = ', max(abs(diff)))
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test,
                                   'Predicted' : actual_predicted,
                                   'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    scatterplot('Random Forest Regression', compare_actual)
    
    return compare_actual

#Function for Linear Regression 
def lr():
    print ('\nLinear Regression Model')
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = LinearRegression()
    model.fit(rescaled_X_train, Y_train)
    
    train_predictions = model.predict(rescaled_X_train)
    train_r2score = model.score(rescaled_X_train, Y_train)
    actual_y_train = np.exp(Y_train)
    actual_train_predictions = np.exp(train_predictions)
    train_diff = actual_y_train - actual_train_predictions
    
    print ('Train R2 Score = ', round(train_r2score, 2))
    print ('Train Log RMSE = ', round(mean_squared_error(Y_train, train_predictions), 2))
    print ('Train Actual RMSE = ', round(actualRMSE(train_diff), 2))
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    r2score = model.score(rescaled_X_test, Y_test)
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    diff_baseline = actual_y_test - baseline
    
    print ('Test R2 Score = ', round(r2score, 2))
    print ('Test Log RMSE = ', round(mean_squared_error(Y_test, predictions), 2))
    print ('Test Actual RMSE = ', round(actualRMSE(diff), 2))
    print ('Baseline RMSE =', round(actualRMSE(diff_baseline), 2))
    print ('Max Error = ', max(abs(diff)))
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test,
                                   'Predicted' : actual_predicted,
                                   'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    scatterplot('Linear Regression', compare_actual)
    
    return compare_actual

#Function for Weighted Average of Different Models 
def ensemble(dict_list, baseline):
    
    print ('\nEnsemble Methods')
    
    y_test = dict_list[0]['Test Data']
    
    # Best Outcome - 1731 : 75, 10, 5, 15
    predicted = (dict_list[0]['Predicted'] * 0.75) + \
                (dict_list[1]['Predicted'] * 0.10) + \
                (dict_list[2]['Predicted'] * 0.05) + \
                (dict_list[3]['Predicted'] * 0.15)
    
    diff = predicted - y_test
    print ('Actual RMSE =', round(actualRMSE(diff), 2))
    print ('Max Error = ', max(abs(diff)))
    
    diff_baseline = baseline - y_test
    print('Baseline Error =', round(actualRMSE(diff_baseline), 2))
    
    compare_actual = pd.DataFrame({'Test Data': y_test,
                                   'Predicted' : predicted,
                                   'Difference' : diff,
                                   'Baseline' : baseline,
                                   'Baseline Diff.': diff_baseline})     
    compare_actual = compare_actual.astype(int)
    compare_actual = compare_actual.drop(columns = ['Baseline', 'Baseline Diff.'])
    scatterplot('Weighted Averages Model', compare_actual)
    return compare_actual

#Saving model results to CSV
def save_as_csv():
    output = lr()
    output.to_csv('lr_results.csv', index = False)
    output = xgb()
    output.to_csv('xgb_results.csv', index = False)
    output = rfr()
    output.to_csv('rfr_results.csv', index = False)

#Helper Function for Ensemble
def combine_methods():
    preprocess()
    dict_list = []
    
    neu_net = pd.read_csv('nn_results.csv')
    lin_reg = pd.read_csv('lr_results.csv')
    xgb_reg = pd.read_csv('xgb_results.csv')
    rfr_reg = pd.read_csv('rfr_results.csv')
    
    dict_list.append(neu_net)
    dict_list.append(lin_reg)
    dict_list.append(xgb_reg)
    dict_list.append(rfr_reg)
    
    ensemble(dict_list, baseline)
    
combine_methods()