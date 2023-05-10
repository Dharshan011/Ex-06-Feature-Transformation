# Ex-06-Feature-Transformation
## AIM
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM
## STEP 1
Read the given Data

## STEP 2
Clean the Data Set using Data Cleaning Process

## STEP 3
Apply Feature Transformation techniques to all the features of the data set

## STEP 4
Save the data to the file

## CODE
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.isnull().sum()

df.describe()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
## OUTPUT
## DATASET
![Screenshot 2023-05-10 124240](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/d8ed47d5-b5ac-44bc-a432-68c892b6cc26)


![Screenshot 2023-05-10 124247](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/abf44ef8-e134-4415-b6ab-f4a0b69cc17d)

![Screenshot 2023-05-10 124255](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/d7ccf7fb-17c9-4847-8fa0-2e7664b6e3ce)


![Screenshot 2023-05-10 124304](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/2cf9eab3-8ad7-4ebc-aad5-14a8f4701d03)

![Screenshot 2023-05-10 124313](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/571c6c97-131e-42af-a7cd-64d804a4144e)

![Screenshot 2023-05-10 124323](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/e3f7f2dd-5751-464a-91f3-bafd444daba2)

![Screenshot 2023-05-10 124332](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/ef7a6626-22ce-47b5-9861-4a43d35b2e14)

![Screenshot 2023-05-10 124339](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/171f1e44-a1b9-408b-baf4-8b043066c768)

![Screenshot 2023-05-10 124348](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/aeaf235d-c963-411e-b60c-0bc6447697ed)



![Screenshot 2023-05-10 124357](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/1012d73f-dbac-4241-8247-1811ac0da386)
![Screenshot 2023-05-10 124405](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/bc7320a8-121a-44e4-8290-9d811c880ab5)

![Screenshot 2023-05-10 124414](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/b469bce9-58b5-4031-be01-19586219a056)


![Screenshot 2023-05-10 124422](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/df0801f5-6243-4824-b03f-3485d6fd6b5f)

![Screenshot 2023-05-10 124430](https://github.com/Dharshan011/Ex-06-Feature-Transformation/assets/113497491/7bd74e3d-426d-4f8b-981c-c7bfbc3309f1)















## RESULT:
Thus, Feature transformation is performed and executed successfully for the given dataset
