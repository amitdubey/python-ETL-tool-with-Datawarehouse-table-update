
import pandas as pd 
from pandas import read_csv

df = pd.read_csv('/Users/amitdubey/Desktop/microsoft/DimTbl_may5.csv',index_col=1)
print(df.head(10))