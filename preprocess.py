import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Function to generate heatmap
def show_heat_map(data, columns):
    correlation_map = np.corrcoef(data[columns].values.T)
    sns.set(font_scale=0.8)
    plt.subplots(figsize=(20,20))
    sns.heatmap(correlation_map, cbar=True, square=True, fmt='.2f', \
                yticklabels=columns.values, xticklabels=columns.values)
    plt.savefig('Heatmap.png',  dpi=100)
    plt.clf()    

#Reading the dataset
df = pd.read_csv('data/OnlineNewsPopularity.csv')
df = df.rename(columns = lambda x: x.strip())
size = len(df.index)

df = df.drop(columns = ['url', 'timedelta'])

#Finding Correlation Matrix
correlation = df.corr(method = 'pearson')
columns = correlation.nlargest(len(df.columns), 'shares').index

#show_heat_map(df, columns)

#Dropping Outliers
df = df[df['shares'] > 100]
df = df[df['shares'] < 23000]

#df.boxplot(column = 'shares', return_type = 'axes')
#plt.savefig('Boxplot_1.png',  dpi=100)

print ('% of Data Removed: '+ str(((size - len(df.index))*100)/size) + '%')
size = len(df.index)

#Log-normalizing 'shares'
df['shares'] = np.log(df['shares'])

#df.boxplot(column = 'shares', return_type = 'axes')
#plt.savefig('Boxplot_3.png',  dpi=100)

#Dropping highly correlated columns
df = df.drop(columns = ['n_non_stop_words', 'n_non_stop_unique_tokens', 
                        'kw_max_min', 'kw_max_avg', 'self_reference_min_shares', 
                        'self_reference_max_shares', 'is_weekend'])