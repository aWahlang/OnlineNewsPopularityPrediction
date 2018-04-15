import pandas as pd
import numpy as np

'''
def normalize(df):
    result = df.copy()
    for feature in df.columns:
        if feature != 'shares':
            max_value = df[feature].max()
            min_value = df[feature].min()
            result[feature] = (df[feature] - min_value) / (max_value - min_value)
    return result
'''

df = pd.read_csv('data/OnlineNewsPopularity.csv')
df = df.rename(columns = lambda x: x.strip())
size = len(df.index)
'''
df = df.drop(columns = ['url', 'timedelta', 'weekday_is_saturday', 
                        'weekday_is_sunday', 'weekday_is_monday',
                        'weekday_is_tuesday', 'weekday_is_wednesday',
                        'weekday_is_thursday', 'weekday_is_friday',
                        'kw_min_min', 'kw_max_min',
                        'kw_min_max', 'kw_max_max',
                        'kw_min_avg', 'kw_max_avg',
                        'self_reference_min_shares', 'self_reference_max_shares',
                        'min_positive_polarity', 'max_positive_polarity',
                        'min_negative_polarity', 'max_positive_polarity'])
'''
df = df.drop(columns = ['url', 'timedelta', 'weekday_is_sunday', 'is_weekend'])


#print ('Before Removing Outliers')
#df.boxplot(column = 'shares', return_type = 'axes')

df = df[df['shares'] > 100]
df = df[df['shares'] < 23000]
print ('% of Data Removed: '+ str(((size - len(df.index))*100)/size) + '%')
size = len(df.index)

#print ('After Removing Outliers')
#df.boxplot(column = 'shares', return_type = 'axes')

df['shares'] = np.log(df['shares'])

#print ('After Log Normalizing')
#df.boxplot(column = 'shares', return_type = 'axes')

df = df.drop(columns = ['n_non_stop_words', 'n_non_stop_unique_tokens', 
                        'kw_max_min', 'kw_max_avg', 'self_reference_min_shares', 
                        'self_reference_max_shares'])
correlation = df.corr(method = 'pearson')

#df = normalize(df)

df.to_csv('data/PreprocessedFile.csv', index = False)

