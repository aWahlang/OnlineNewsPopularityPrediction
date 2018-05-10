import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from nn_preprocessing import parseReg
from keras.models import model_from_json

data = pd.read_csv('OnlineNewsPopularity.csv')
data = parseReg(data)
seed = 7
np.random.seed(seed)
print("Dataset loaded")

#80% split of dataset 
split = int(data.shape[0] * 0.8)

#creating testing set and training set
train = data[:split,:]
test = data[split:data.shape[0],:]

#splitting into input and output sets
train_in = train[:, 0:51]
train_out = train[:, 51]
test_in = test[:, 0:51]
test_out = test[:, 51]
print("Training and Test sets created")

#Network Model
def base_model():
    #model Structure 
    model = Sequential()
    model.add(Dense(200, input_dim = 51, activation= 'relu', kernel_initializer='normal'))
    model.add(Dense(400, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    
    #model Optimizer
    optim = Adam(lr = 0.0001)
    model.compile(loss = 'mean_squared_error', optimizer = optim, metrics = ['MSE'])
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

def train_model(model):
    model.fit(train_in, train_out, epochs=60, batch_size=20, shuffle = True, validation_split= 1)
    save_model(model)
    return model

def test_model(testModel):  
    error = []
    x = []
    y = []
    print("Testing Model...")
    for i in range(0,len(test_out)):
        res = testModel.model.predict(test_in[[i],:])
        
        y.append(10**res[0][0])
        x.append(10**test_out[i])
        err = 10**res[0] - 10**test_out[i]
        error.append(err[0])
    
    #result csv
    result = pd.DataFrame({'Test Data' : x,
                           'Predicted' : y,
                           'Difference' : error })
    result = result.astype(int)
    result.to_csv('nn_results.csv', index = False)
    
    #result plot
    error = [x**2 for x in error]
    maxEr = max(error)**0.5
    avgEr = (sum(error) / len(error))**0.5      
    print("Max Error: %f"%(maxEr)) 
    print("Avg Error: %f"%(avgEr))
    plt.scatter(x,y, marker = '.', alpha = 0.2)
    plt.title("Mlultilayer Perceptron")
    plt.xlabel("Test data")
    plt.ylabel("Predicted data")
    plt.show()

#Creating model    
#myModel = base_model()
#myModel = train_model(myModel)

#Load model
myModel = load_model()

#Compile Model
myModel.compile(loss = 'mean_squared_error', optimizer='adam', metrics = ['MSE'])

#myModel = train_model(myModel)

#Test model
test_model(myModel)



