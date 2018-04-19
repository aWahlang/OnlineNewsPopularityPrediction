import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

def show_heat_map(data, columns):
    correlation_map = np.corrcoef(data[columns].values.T)
    sns.set(font_scale=1.0)
    sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)
    plt.show()    
    
def pipeline():
    pipelines = []
    pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
    pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
    pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
    pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
    
    results = []
    names = []
    for name, model in pipelines:
        kfold = KFold(n_splits=10, random_state=21)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

def svm(kernel_type = 'rbf'):
    print ('\nSupport Vector Machine Regression\nKernel Type =', kernel_type)
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = SVR(kernel = kernel_type)
    model.fit(rescaled_X_train, Y_train)
    #model.fit(X_train, Y_train)
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    #predictions = model.predict(X_test)
    print ('Log RMSE =', mean_squared_error(Y_test, predictions))
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    actual_error = diff*diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual RMSE =', actual_error)
    r2 = sklearn.metrics.r2_score(actual_y_test, actual_predicted)
    print('R2 Score =', r2)
    
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted' : actual_predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual

def rfr_nEstimators():
    n_estimators = [10, 50, 100, 200, 300, 400, 500]
    for item in n_estimators:
        callFunc = rfr(item)
        
def svm_kernels():
    kernel = ['linear', 'poly', 'rbf']
    for item in kernel:
        callFunc = svm(item)

def rfr(num = 300):
    print ('\nRandom Forest Regression\nN-Estimators =', num)
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = RandomForestRegressor(n_estimators = num)
    model.fit(rescaled_X_train, Y_train)
    #model.fit(X_train, Y_train)
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    #predictions = model.predict(X_test)
    print ('Log RMSE =', mean_squared_error(Y_test, predictions))
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    actual_error = diff*diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual RMSE =', actual_error)
    r2 = sklearn.metrics.r2_score(actual_y_test, actual_predicted)
    print('R2 Score =', r2)
    
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted' : actual_predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual

def knn():
    print ('\nK-Neighbors Regression')
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = KNeighborsRegressor()
    model.fit(rescaled_X_train, Y_train)
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    print ('Log RMSE =', mean_squared_error(Y_test, predictions))
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    actual_error = diff*diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual RMSE =', actual_error)
    r2 = sklearn.metrics.r2_score(actual_y_test, actual_predicted)
    print('R2 Score =', r2)
    
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted' : actual_predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual

def lr():
    print ('\nLinear Regression')
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = LinearRegression()
    model.fit(rescaled_X_train, Y_train)
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    print ('Log RMSE =', mean_squared_error(Y_test, predictions))
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    actual_error = diff*diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual RMSE =', actual_error)
    r2 = sklearn.metrics.r2_score(actual_y_test, actual_predicted)
    print('R2 Score =', r2)
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted' : actual_predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual

def cart():
    print ('\nDecision Tree Regression')
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = DecisionTreeRegressor()
    model.fit(rescaled_X_train, Y_train)
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    print ('Log RMSE =', mean_squared_error(Y_test, predictions))
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    actual_error = diff*diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual RMSE =', actual_error)
    r2 = sklearn.metrics.r2_score(actual_y_test, actual_predicted)
    print('R2 Score =', r2)
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted' : actual_predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual
        
def gbm():
    print ('\nGradient Boosting Regression')
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = GradientBoostingRegressor(n_estimators = 100)
    model.fit(rescaled_X_train, Y_train)
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    print ('Log RMSE =', mean_squared_error(Y_test, predictions))
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    actual_error = diff*diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual RMSE =', actual_error)
    r2 = sklearn.metrics.r2_score(actual_y_test, actual_predicted)
    print('R2 Score =', r2)
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted' : actual_predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual

def ensemble(dict_list, baseline):
    
    print ('\nEnsemble Methods')
    
    y_test = dict_list[0]['Test Data']
    size = len(dict_list)
    predicted = sum(item['Predicted'] for item in dict_list)/size
    
    diff = predicted - y_test
    actual_error = diff * diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual RMSE =', actual_error)
    
    
    diff_baseline = baseline - y_test
    baseline_error = diff_baseline * diff_baseline
    baseline_error = np.sqrt(sum(baseline_error))
    print('Baseline Error =', baseline_error)
    r2 = sklearn.metrics.r2_score(y_test, predicted)
    print('R2 Score =', r2)
    
    compare_actual = pd.DataFrame({'Test Data': y_test, 'Predicted' : predicted, 'Difference' : diff, 'Baseline' : baseline, 'Baseline Diff.': diff_baseline})     
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual

def combine_methods():
    dict_list = []
    #dict_list.append(gbm())
    #dict_list.append(lr())
    #dict_list.append(cart())
    #dict_list.append(knn())
    dict_list.append(svm())
    ensemble(dict_list, baseline)

df = pd.read_csv('data/PreprocessedFile.csv')
    
correlation = df.corr(method = 'pearson')
# N-Largest: 30
columns = correlation.nlargest(30, 'shares').index

X = df[columns]
Y = X['shares'].values
X = X.drop('shares', axis = 1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, shuffle = False, stratify = None)
baseline = np.exp(Y_train)
baseline = baseline.mean()