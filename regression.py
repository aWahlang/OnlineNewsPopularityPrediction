import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

def knn():
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = KNeighborsRegressor()
    model.fit(rescaled_X_train, Y_train)
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    print ('Log Mean Squared Error =', mean_squared_error(Y_test, predictions))
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    actual_error = diff*diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual Mean Squared Error =', actual_error)
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted Price' : actual_predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual

def lr():
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = LinearRegression()
    model.fit(rescaled_X_train, Y_train)
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    print ('Log Mean Squared Error =', mean_squared_error(Y_test, predictions))
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    actual_error = diff*diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual Mean Squared Error =', actual_error)
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted Price' : actual_predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual

def cart():
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = DecisionTreeRegressor()
    model.fit(rescaled_X_train, Y_train)
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    print ('Log Mean Squared Error =', mean_squared_error(Y_test, predictions))
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    actual_error = diff*diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual Mean Squared Error =', actual_error)
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted Price' : actual_predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual
        
def gbm():
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model = GradientBoostingRegressor(n_estimators = 100)
    model.fit(rescaled_X_train, Y_train)
    
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    print ('Log Mean Squared Error =', mean_squared_error(Y_test, predictions))
    
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = actual_y_test - actual_predicted
    actual_error = diff*diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual Mean Squared Error =', actual_error)
    
    compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted Price' : actual_predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual

def ensemble(dict_list):
    y_test = dict_list[0]['Test Data']
    size = len(dict_list)
    predicted = sum(item['Predicted Price'] for item in dict_list)/size
    diff = predicted - y_test
    actual_error = diff*diff
    actual_error = np.sqrt(sum(actual_error))
    print ('Actual Mean Squared Error =', actual_error)
    
    compare_actual = pd.DataFrame({'Test Data': y_test, 'Predicted Price' : predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))
    return compare_actual

def combine_methods():
    dict_list = []
    dict_list.append(gbm())
    dict_list.append(lr())
    dict_list.append(cart())
    dict_list.append(knn())
    ensemble(dict_list)

df = pd.read_csv('data/PreprocessedFile.csv')
    
correlation = df.corr(method = 'pearson')
columns = correlation.nlargest(30, 'shares').index

X = df[columns]
Y = X['shares'].values
X = X.drop('shares', axis = 1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, shuffle = False, stratify = None)

