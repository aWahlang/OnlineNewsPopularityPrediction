import pandas as pd
import numpy as np

df = pd.read_csv('data/OnlineNewsPopularity.csv')
df = df.rename(columns = lambda x: x.strip())
size = len(df.index)

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

df = df[df['shares'] > 100]
df = df[df['shares'] < 23000]
print ('% of Data Removed: '+ str(((size - len(df.index))*100)/size) + '%')
size = len(df.index)

df['shares'] = np.log(df['shares'])
df.boxplot(column = 'shares', return_type = 'axes')

df.to_csv('data/PreprocessedFile.csv', index = False)
