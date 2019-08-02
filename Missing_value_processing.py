#Missing value processing

import pandas as pd  
import numpy as np  
from sklearn.preprocessing import Imputer  

# Generate missing data
out_dir = 'data/'
df = pd.read_csv(out_dir + 'StackOverFlow_2017.csv', skiprows=1, low_memory=False)

# Check which values are missing
nan_all = df.isnull()  # Get the Null value in all data frames
print(nan_all)  # print all nan value

# Check which columns are missing
nan_col1 = df.isnull().any()  # Get first column containing NA
nan_col2 = df.isnull().all()  # Get all columns containing NA
print(nan_col1)  # print first column containing NA
print(nan_col2)  # print all columns containing NA

# Discard missing values
df = df.dropna()  # discard records with NA value

# replace missing values with specific values using sklearn
nan_model = Imputer(missing_values='NaN', strategy='mean',
                    axis=0)  # Create a replacement rule: replace the missing value with a value of NaN with the mean
df = nan_model.fit_transform(df)  # Application rule

#replace missing values with specific values using pandas 
df = df.fillna(0)  # Replace missing values with 0
#save cleaned data into csv file
df.to_csv(out_dir + "cleaned_StackOverFlow.csv",index=False)


