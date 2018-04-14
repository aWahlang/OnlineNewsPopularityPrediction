1. Using Pearson's correlation, to find the correlation between each of the attributes.
2. Selected the top 30 (because rest are all negative, checked but deleted that code):
columns = df.corr(method = 'pearson')['shares'] > 0.0
df = df.loc[:, df.corr()['shares'] > 0.0]
3. Raw data gas different scales. Standardised data set has each attribute as mean of 0 and SD of 1. Pipelines implement standardisation.
4. Cross-validation to validate performance of algorithms.