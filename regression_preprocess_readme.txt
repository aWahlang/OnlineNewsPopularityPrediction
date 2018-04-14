Preprocessing the Data:

1. url: dropped, irrelevant
2. timedelta: dropped, irrelevant
3. weekday_is_saturday/sunday: dropped, already in is_weekend
4. weekday_is_monday/tuesday/wednesday/thursday/friday: dropped, if is_weekend is 0, then it is obvious that it will be a weekday
5. min, max of variables: dropped, the average already says something about it
6. shares: observations below 100 and above 23000 removed (too less or too high) - outliers, % of Data Removed: 1.90697205125618%
7. shares: log normalised (too large values)