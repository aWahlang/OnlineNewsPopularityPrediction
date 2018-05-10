import pandas as pd
import math
from sklearn import preprocessing

def logOf(val):
    return math.log10(val)

def parseReg(dataFrame):
    #drop attributes
    dataFrame = dataFrame.drop(columns = ['url', ' timedelta', ' is_weekend',' n_non_stop_words', ' n_non_stop_unique_tokens', 
                        ' kw_max_min', ' kw_max_avg', ' self_reference_min_shares', 
                        ' self_reference_max_shares'])
    #drop outliers
    dataFrame = dataFrame[dataFrame[' shares'] > 100]
    dataFrame = dataFrame[dataFrame[' shares'] < 23000]
    
    #normalizing dataset
    min_max = preprocessing.MinMaxScaler()

    cols = dataFrame.columns.values
    for i in range(0,len(cols)-1):
        dataFrame[[cols[i]]] = min_max.fit_transform(dataFrame[[cols[i]]])
     
    #talking log of shares
    dataFrame[' shares'] = dataFrame[' shares'].apply(logOf)
    
    return dataFrame.values            
