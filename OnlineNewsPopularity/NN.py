import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from PreProcessing import parse

# fix random seed for reproducibility
np.random.seed(7)
data = pd.read_csv('OnlineNewsPopularity.csv')
data = parse(data)

#creating testing set and training set
train = data[:30000,:]
test = data[30000:39644,:]

#splitting into input and output sets
train_in = train[:, 0:57]
train_out = train[:, 58]
test_in = test[:, 0:57]
test_out = test[:, 58]
#one hot encoding for classes
train_out = to_categorical(train_out)
test_out = to_categorical(test_out)
#remove 0s columns 
train_out = np.delete(train_out, np.s_[0], axis = 1)
test_out = np.delete(test_out, np.s_[0], axis = 1)

#Network Model
model = Sequential()
model.add(Dense(200, input_dim = 57, activation = 'relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(3,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
model.fit(train_in, train_out, epochs = 10, batch_size= 5)

#model evaluation
scores = model.evaluate(test_in, test_out)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

