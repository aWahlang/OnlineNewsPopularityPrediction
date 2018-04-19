import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn import preprocessing
from PreProcessing import parse

data = pd.read_csv('OnlineNewsPopularity.csv')
data = parse(data)

#creating testing set and training set
train = data[:35000,:]
test = data[35000:39644,:]

#splitting into input and output sets
train_in = train[:, 0:57]
train_out = train[:, 58]
test_in = test[:, 0:57]
test_out = test[:, 58]
#one hot encoding for classes and normalization for inputs
min_max = preprocessing.MinMaxScaler()
train_in = min_max.fit_transform(train_in)
test_in = min_max.fit_transform(test_in)
train_out = to_categorical(train_out)
test_out = to_categorical(test_out)
#remove 0s columns 
#train_out = np.delete(train_out, np.s_[0], axis = 1)
#test_out = np.delete(test_out, np.s_[0], axis = 1)

#Network Model
model = Sequential()
model.add(Dense(100, input_dim = 57, activation = 'relu'))
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dense(4,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
model.fit(train_in, train_out, epochs = 10, batch_size= 20)

#model evaluation
scores = model.evaluate(test_in, test_out)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print(scores)

