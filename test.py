#!/usr/bin/env python
# coding: utf-8

# # Load the Fact and Dimensions tables and libraries

# In[ ]:


from datetime import date, datetime, timedelta
#My SQL server data retrival and injection tool
from sqlalchemy import create_engine
import mysql.connector
import pandas as pd

import regex as re
import numpy as np
# imputer libraries from Sklearn
from sklearn.impute import SimpleImputer
#EDA library to see data trends
from pivottablejs import pivot_ui
#missing value visual analysis of a dataframe
import missingno as msno
# panda data profiling and EDA tool
from pandas_profiling import ProfileReport

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set(color_codes=True)
#dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')



DirPath ="/Users/amitdubey/desktop/microsoft/"
Df = pd.read_csv(DirPath+'FactTbl_may5.csv',encoding='ISO-8859-1',na_values=['.'])
DimDf = pd.read_csv(DirPath+'DimTbl_may5.csv',encoding='ISO-8859-1',na_values=['.'])


# # Dataframe Quick Peek 

# In[ ]:


Df.describe()


# In[ ]:


DimDf.describe()


# # Fact Table RowxColumn Shape

# In[ ]:



Df.shape


# # Dimension Table RowxColumn Shape

# In[ ]:



DimDf.shape


# # Check the Data Types 

# In[ ]:


print(Df.dtypes)
print(Df.columns)


# # column name has white spaces - Strip em

# In[ ]:


Df.columns = Df.columns.str.strip()


# In[ ]:


Df['SellThruQTY']=Df.SellThruQTY.apply(lambda x: np.where(x.isdigit(),x,'0'))
Df['SellinQTY']=Df.SellinQTY.apply(lambda x: np.where(x.isdigit(),x,'0'))


# # Convert the columns to appropriate data types

# In[ ]:




Df['dateid'] = pd.to_datetime(Df['dateid'])
Df['CalendarDate'] = pd.to_datetime(Df['CalendarDate'])
Df['SubsidiaryName'] = Df['SubsidiaryName'].astype('str')
Df['TPNamee'] = Df['TPNamee'].astype('str')
Df['DeviceName'] = Df['DeviceName'].astype('str')
Df['PFAMName'] = Df['PFAMName'].astype('str')
Df['ProductPartNbr'] = Df['ProductPartNbr'].astype('str')
Df['SellThruQTY'] = pd.to_numeric(Df['SellThruQTY'])
Df['SellinQTY'] = Df['SellinQTY'].astype(int)


# # Create new Date Features

# In[ ]:


Df['Year'] = pd.DatetimeIndex(Df['CalendarDate']).year
Df['Month'] = pd.DatetimeIndex(Df['CalendarDate']).month
Df['Day']=pd.DatetimeIndex(Df['CalendarDate']).day
Df['Day_of_Week']=pd.DatetimeIndex(Df['CalendarDate']).weekday


# # Checking Null  or missing Value

# In[ ]:



print (Df.isnull().sum())


# # Count Null Values

# In[ ]:



print (Df.isnull().sum().sum())


# # Auto impute Data frame columns in case of missing values

# In[ ]:



# imp = SimpleImputer(missing_values=0, strategy='most_frequent')
# print(pd.DataFrame(imp.fit_transform(Df),
#                    columns=Df.columns,
#                    index=Df.index))


# In[ ]:



print(DimDf.isnull().sum())


# In[ ]:


print(DimDf.isnull().sum().sum())


# In[ ]:


Df.kurtosis()


# In[ ]:


Df.skew()


# In[ ]:


#Fact Table Quick Summary using panda profiling
profile = ProfileReport(Df, title='Pandas Profiling Report', html={'style':{'full_width':True}})


# In[ ]:


profile.to_widgets()


# In[ ]:


display(profile)


# In[ ]:


#Dimension Table Quick summary using Panda Profiler
DimProfile = ProfileReport(DimDf, title='Pandas Dimension table Profile Report', html={'style':{'full_width':True}})


# In[ ]:


DimProfile.to_widgets()


# ## Feature Engineering

# In[ ]:


#Creating a new columns for each of the new features
#Processor
#RAM
#HDD 
#comm Ind 0 or 1 if comm is in the text
#Bundle ind 0 or 1 if Bndl is present in the text


# # Features : Processor,RAM,HDD and Comm,Bundl Flags

# In[ ]:



#Df['Processor']= Df['PFAMName'].str.extract(r'\s(i[0-9])|(\s.\s)', expand=False)
Df['Processor']= Df['PFAMName'].str.extract(r'(i[1-9]|[M])', expand=False)
Df['Processor']= Df['Processor'].replace(np.nan,'N/A')
Df['RAM']= Df['PFAMName'].str.extract(r'([4]GB)', expand=False)
Df['RAM']= Df['PFAMName'].str.extract(r'(([4]GB)|([8]GB)|([1][6]GB)|([1][6])|([8]))', expand=False)
#Df['RAM']= Df['PFAMName'].str.extract(r'((\s[0-8]GB)|(\s[1][6])|([8])|([1][6])|(\s[0-1][0-6]GB))', expand=False)
Df['RAM']= Df['RAM'].replace(np.nan,'N/A')
Df['HDD']= Df['PFAMName'].str.extract(r'(\s[1-5][1-5][1-8])', expand=False)
Df['HDD']= Df['PFAMName'].str.extract(r'((\s[1-5][1-5][1-8])|([1]TB)|([2][5][6]))', expand=False)
#Df['HDD']= Df['PFAMName'].str.extract(r'((\s[1-5][1-5][1-8]GB)|([1]TB)|([2][5][6]))', expand=False)
Df['HDD']= Df['HDD'].replace(np.nan,'N/A')

Df['COMM_IND']=Df['PFAMName'].str.contains('COMM', flags=re.IGNORECASE, regex=True)
Df['Bundle_IND']=Df['PFAMName'].str.contains('Bndl', flags=re.IGNORECASE, regex=True)

Df



# # Cleaning up Dimension Table

# In[ ]:


DimDf.shape
#df2 = pd.DataFrame(DimDf['PFAMName'].str.split(' ').tolist())


# In[ ]:


DimDf.columns


# In[ ]:


print(DimDf.dtypes)


# # Checking for Null or missing Values

# In[ ]:


print(DimDf.isnull().sum())


# # Data Type conversion into appropriate type(if needed)

# In[ ]:


DimDf['SubsidiaryName'] = DimDf['SubsidiaryName'].astype('str')
DimDf['TPNamee'] = DimDf['TPNamee'].astype('str')
DimDf['DeviceName'] = DimDf['DeviceName'].astype('str')
DimDf['PFAMName'] = DimDf['PFAMName'].astype('str')
DimDf['ProductPartNbr'] = DimDf['ProductPartNbr'].astype('str')


# In[ ]:



#Df['Processor']= Df['PFAMName'].str.extract(r'\s(i[0-9])|(\s.\s)', expand=False)
DimDf['Processor']= DimDf['PFAMName'].str.extract(r'(i[0-9]|[M])', expand=False)
DimDf['Processor']= DimDf['Processor'].replace(np.nan,'N/A')
DimDf['RAM']= DimDf['PFAMName'].str.extract(r'(\s[0-8]GB)', expand=False)
DimDf['RAM']= DimDf['PFAMName'].str.extract(r'((\s[0-8]GB)|([1][6])|([4]GB)|([8]G)|([8]))', expand=False)
#DimDf['RAM']= DimDf['PFAMName'].str.extract(r'((\s[0-8]GB)|([1][6])|([4]GB)|([8]G)|([8]))', expand=False)
DimDf['RAM']= DimDf['RAM'].replace(np.nan,'N/A')
DimDf['HDD']= DimDf['PFAMName'].str.extract(r'(\s[1][2][8]GB)', expand=False)
DimDf['HDD']= DimDf['PFAMName'].str.extract(r'((\s[1][2][8]GB)|(\s[2][5][6]GB)|(\s[5][1][2]GB)|([5][1][2])|([2][5][6])|([1]TB))', expand=False)
DimDf['HDD']= DimDf['HDD'].replace(np.nan,'N/A')

DimDf['COMM_IND']=DimDf['PFAMName'].str.contains('COMM', flags=re.IGNORECASE, regex=True)
DimDf['Bundle_IND']=DimDf['PFAMName'].str.contains('Bndl', flags=re.IGNORECASE, regex=True)


# In[ ]:


DimDf.head(20)


# In[ ]:


print(DimDf.dtypes)


# In[ ]:




pivot_ui(DimDf)


# In[ ]:



get_ipython().run_line_magic('matplotlib', 'inline')
msno.matrix(Df)


# In[288]:


sns.pairplot(Df)


# In[ ]:


sns.pairplot(Df, hue="Day");


# In[ ]:


sns.pairplot(DimDf, hue="DeviceName");


# # Feature selection methods ( based on importance for ML)

# Feature selection can be done in multiple ways but there are broadly 3 categories of it:
# 1. Filter Method
# 2. Wrapper Method
# 3. Embedded Method

# # Building Dummy Variables from Categorical variables

# In[289]:


Df1 = pd.get_dummies(Df)


# In[290]:


X = Df1[[  'TPId','SellinQTY', 'SellThruQTY', 'Year',
       'Month', 'Day_of_Week','COMM_IND', 'Day','RAM_16', 'RAM_16GB', 'RAM_8', 'RAM_8GB', 'HDD_ 128', 'HDD_ 256',
       'HDD_ 512', 'HDD_1TB', 'HDD_256', 'HDD_N/A']]  #independent columns
y = Df1['Bundle_IND']    #target column i.e bundle
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[ ]:


print(Df1.columns)


# In[ ]:


Df[['CYear','QTR']] = Df.CalendarQuarter.str.split("-",expand=True)
Df=Df.drop(columns=['CYear'],axis=1)
Df2 = pd.DataFrame(Df['PFAMName'].str.split(' ').tolist())


# In[ ]:


Df2.head()

Df2=Df2.drop(columns=[6, 7],axis=1)

Df2.columns
Df2.columns = ["DeviceType", "HDD","Processor", "RAM", "COMM","BUNDL"]
#pd.concat([df, df2], axis=1)
Df2


# # Drop Duplicates before we merge Fact and Dim Tables

# In[ ]:


Df.apply(lambda x: x.duplicated().any())


# In[ ]:


DF=Df.drop_duplicates()


# In[ ]:


DF.shape


# In[ ]:


print(DimDf.apply(lambda x: x.duplicated().any()))
DF=DimDf.drop_duplicates()


# In[ ]:



DF.shape


# In[ ]:


## test code to showcase that we can join the two data tables in pandas too,using pandasql

from pandasql import sqldf 

ddim = sqldf("select * from Df left outer join DimDf  on Df.TPId=DimDf.TPId")
ddim


# In[ ]:


ddim.shape


# # ETL automation on above script saved as .Py file and fact/dimension creation in DB

# In[ ]:


import os 
file_path = 'D://python_scripts' 
os.chdir(file_path)
#exec(open("msft_etl.py").read())


# # Create new Fact/Dim table using SQL bulk injection -first time 

# ### Call the stored procedures for Fact and Dim Tables to have new data updated

# In[ ]:


# import MySQL connector
import mysql.connector
import sqlalchemy
from sqlalchemy import create_engine
# connect to server

connection = mysql.connector.connect(user='root', password='test@123',
                            host='localhost', database='testdb')
engine = sqlalchemy.create_engine('mysql+pymysql://root:test@123@localhost:3306/testdb')
connection = engine.connect()

db_name = 'microsoft'
create_db_query = "CREATE DATABASE IF NOT EXISTS {0} DEFAULT CHARACTER SET 'UTF8MB4'".format(db_name)
connection.execute(create_db_query) 
print('Connected to database.')
#cursor = connection.cursor()

#inject Fact and Dimension table to MySQL Microsoft Database
Df.to_sql(name='FactTbl_may5',con=engine,if_exists='replace',index=False,chunksize = 10000)
DimDf.to_sql(name='DimTbl_may5',con=engine,if_exists='replace',index=False,chunksize = 10000)

# First update dimensions Table using stored procedure

# cursor.callproc('microsoftdwh.DimCleanup')
print('Dimension tables updated.')

# Second update facts table using in the Dataware house

#cursor.callproc('microsoft.FactCleanup')
print('Fact tables updated.')

# commit & close connection
#cursor.close()
#connection.commit()


# # Retriving data once the SQL script updates the Fact Table from Dimension Table

# In[ ]:


## another good library to connect and retrive data from mysql server or PyODBC for MS-SQLServer
import pymysql

db_connection_str = 'mysql+pymysql://root:test@123@localhost/testdb'
db_connection = create_engine(db_connection_str)

DfFact = pd.read_sql("""
 SELECT  d.DeviceName ,
  d.PFAMName ,
  d.SubsidiaryCode ,
  d.SubsidiaryName ,
  d.TPId   ,
  d.TPNamee ,
  d.ProductPartNbr ,
  d.Processor ,
  d.RAM ,
  d.HDD ,
  d.COMM_IND,
  d.Bundle_IND ,
	F.CALENDARDATE as CalendarDate,
	f.SellinQty,
	f.SellThruQty,
	f.CalendarQuarter ,
	f.Year ,
	f.Month ,
	f.Day ,
	f.Day_of_Week 
  FROM     DIMTBL_MAY5 D
   INNER   JOIN    FACTTBL_MAY5  F ON
			f.SubsidiaryName =d.SubsidiaryName and
			f.TPId =d.TPId  and
			f.TPNamee =d.TPNamee and
			f.DeviceName =d.DeviceName and
			f.PFAMName =d.PFAMName and
			f.ProductPartNbr =d.ProductPartNbr and
			f.COMM_IND=d.COMM_IND and
			f.Bundle_IND=d.Bundle_IND and
			f.Processor =d.Processor and
			f.RAM =d.RAM
    Group by         
     d.DeviceName ,
			d.PFAMName ,
			d.SubsidiaryCode ,
			d.SubsidiaryName ,
			d.TPId   ,
			d.TPNamee ,
			d.ProductPartNbr ,
			d.Processor ,
			d.RAM ,
			d.HDD ,
			d.COMM_IND,
			d.Bundle_IND ,
            F.CALENDARDATE,
            f.SellinQty,
            f.SellThruQty,
			f.CalendarQuarter ,
			f.Year ,
			f.Month ,
			f.Day ,
			f.Day_of_Week ;""", con=db_connection)
#DfDim = pd.read_sql('SELECT * FROM DimTbl_may5', con=db_connection)


# # BUILDING Entity relationship diagram using python

# In[ ]:


from sqlalchemy_schemadisplay import create_schema_graph
from sqlalchemy import MetaData

graph = create_schema_graph(metadata=MetaData('mysql+pymysql://root:test@123@localhost/testdb'))
graph.write_png('my_erd.png')


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("my_erd.png")
plt.imshow(image)
plt.show()


# # Finally Once data is retrieved, Close the DB connections

# In[ ]:



db_connection.close()
connection.close()
print('Disconnected from database.')


# # Visualize the Data for a short Summary 

# In[ ]:


from os import path, getcwd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator,STOPWORDS

  
comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in DfFact.PFAMName: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[ ]:


DfFact.columns


# In[ ]:


import plotly.graph_objects as go
import plotly.express as px
df = DfFact.sort_values(['CalendarDate'], ascending=[1])
fig = go.Figure(data=go.Scatter(x=Df['CalendarDate'],
                                y=Df['HDD'],
                                mode='markers',
                                marker_color=Df['COMM_IND'],
                                text=Df['DeviceName'])) # hover text goes here

fig.update_layout(title ='Device By HDD')
fig.show()


# In[ ]:



df = Df.sort_values(['CalendarDate'], ascending=[1])
fig = px.bar(df, x="CalendarQuarter", y="COMM_IND", color="Processor", hover_data=['PFAMName'])
fig.show()


# In[ ]:


df = DfFact.sort_values(['CalendarDate'], ascending=[1])
fig = px.bar(df, x="CalendarQuarter", y="SubsidiaryName", color="Processor", hover_data=['PFAMName'])
fig.show()


# In[ ]:


df = DfFact.sort_values(['CalendarDate'], ascending=[1])
fig = px.scatter(df, x="CalendarDate", y="HDD", color="Processor", hover_data=['PFAMName'], title='HDD YOY by Processor Type')
fig.show()


# In[ ]:


df = DfFact.sort_values(['CalendarDate'], ascending=[1])
#df= df[(df['SubsidiaryName']=='United States')]
fig = px.bar(df, x="CalendarQuarter", y="SubsidiaryName", color="Processor", hover_data=['PFAMName'])
fig.show()


# In[ ]:


df = DfFact.sort_values(['CalendarDate'], ascending=[1])
#df= df[(df['SubsidiaryName']=='United States')]
fig = px.bar(df, x="CalendarQuarter", y="Processor", color="HDD", hover_data=['PFAMName'])
fig.show()


# In[ ]:


DfFact['SellinQTY'].plot()
# import pandas as pd
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(ncols=len(DfDim.columns), figsize=(10,5))
# for col, ax in zip(DfDim, axes):
#     DfDim[col].value_counts().sort_index().plot.bar(ax=ax, title=col)

# plt.tight_layout()    
# plt.show()


# In[ ]:





# In[ ]:



df = DimDf.sort_values(['TPNamee'], ascending=[1])
fig = px.bar(DimDf, x="TPNamee", y="Processor",color='HDD' ,title ='Processors Type by Retailers')
fig.show()


# In[ ]:




