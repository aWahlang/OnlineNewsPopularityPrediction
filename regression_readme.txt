1. Using Pearson's correlation, to find the correlation between each of the attributes.
2. Selected the top 30 (because rest are all negative, checked but deleted that code):
columns = df.corr(method = 'pearson')['shares'] > 0.0
df = df.loc[:, df.corr()['shares'] > 0.0]
3. Raw data gas different scales. Standardised data set has each attribute as mean of 0 and SD of 1. Pipelines implement standardisation.
4. Cross-validation to validate performance of algorithms.
5. combine_methods() function basically calls the ensemble with the chosen regressors and compares the result with a baseline test (which over here is the Zero Rule Baseline Algorithm (i.e. the mean)).
6. R-square is very high which means none of the first 4 algos had data through which a line could be passed through.
7. SVR and Random Forest Regressor has been tried.