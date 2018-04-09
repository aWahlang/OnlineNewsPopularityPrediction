import pandas as pd

def parse(dataFrame):
    dataFrame = dataFrame.drop(columns = ['url', ' timedelta'])
    bins = [0,1000,10000, float("inf")]
    labels = [1,2,3]
    dataFrame[' shares'] = pd.cut(dataFrame[' shares'], bins, labels = labels , include_lowest = True)
    
    return dataFrame.values            



