import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from PreProcessing import parseReg
#from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import cross_val_predict
from keras.models import model_from_json

data = pd.read_csv('OnlineNewsPopularity.csv')
data = parseReg(data)
seed = 7
np.random.seed(seed)

#creating testing set and training set
train = data[:31110,:]
test = data[31110:38888,:]

#splitting into input and output sets
train_in = train[:, 0:51]
train_out = train[:, 51]
test_in = test[:, 0:51]
test_out = test[:, 51]

#Network Model
def base_model():
    model = Sequential()
    model.add(Dense(100, input_dim = 51, activation= 'relu', kernel_initializer='normal'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    op = Adam(lr = 0.0001)
    model.compile(loss = 'mean_squared_error', optimizer = op, metrics = ['MSE'])
    return model

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Model Saved.")

def load_model():
    json_file = open("model.json", "r")
    loaded_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_json)
    loaded_model.load_weights("model.h5")
    print("Model Loaded")
    
    return loaded_model

def train_model():
    model = base_model()
    model.fit(train_in, train_out, epochs=20, batch_size=5)
    save_model(model)
    return model

testModel = train_model()
#testModel = load_model()
#testModel.compile(loss = 'mean_squared_error', optimizer='adam', metrics = ['MSE'])

#estimator = KerasRegressor(build_fn=base_model, epochs = 10, batch_size = 10, verbose = 0)
#estimator.fit(train_in, train_out)
#kfold = KFold(n_splits = 2, random_state = seed, shuffle=True)
#results = cross_val_score(estimator, train_in, train_out, cv = kfold)
#print("Training Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#test = cross_val_predict(estimator, test_in, test_out)
#print("Test Results: %.2f (%.2f) MSE" % (test.mean(), test.std()))

error = []
x = []
y = []
print("Testing Model...")
for i in range(0,len(test_out)):
#    print("Testing: " +str(i), end = "" , flush = True)
    res = testModel.model.predict(test_in[[i],:])
#    print("Predicted: %f Truth: %f Error: %f" %(10**res[0], 10**test_out[i], 10**res[0] - 10**test_out[i]))
    y.append(10**res[0])
    x.append(10**test_out[i])
    err = 10**res[0] - 10**test_out[i]
    error.append(err)

error = [x**2 for x in error]
#error = [x**0.5 for x in error]
maxEr = max(error)**0.5
avgEr = (sum(error) / len(error))**0.5      
print("Max Error: %f"%(maxEr)) 
print("Avg Error: %f"%(avgEr))
plt.scatter(x,y, marker = '.', alpha = 0.2)
plt.title("Mlultilayer Perceptron")
plt.xlabel("Test data")
plt.ylabel("Predicted data")
plt.show()


