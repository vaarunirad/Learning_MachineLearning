#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# # Step 1: Importing Data

# In[12]:


retail=pd.read_csv("OnlineRetail.csv", encoding="unicode_escape")

#to escape the different encoding in the csv file


# #Using file path to read csv:
# 
# r'C:\...  .csv'

# In[13]:


retail.head()


# In[15]:


retail.shape


# In[16]:


retail.describe()


# In[17]:


retail.info()


# # Step 2: Cleaning Data
# 

# In[18]:


#count of null values

retail.isnull().sum()


# In[19]:


#null values as a percentage of the total

round((retail.isna().sum()/retail.shape[0]*100),2)


# In[20]:


retail.head(50)


# In[21]:


#dropping the null values

retail.dropna(inplace=True)
retail.shape


# In[22]:


retail.dtypes


# In[23]:


#converting customerID to an object

retail.loc[:,'CustomerID']=retail.loc[:,'CustomerID'].astype(str)


# In[24]:


retail.dtypes


# # Step 3: Data Preparation
We are going to analyze the customers based on below 3 factors:

* R (Recency): Number of days since last purchase of customer. 
* F (Frequency): Number of transactions of customers. 
* M (Monetary): Total amount of transactions (revenue contributed by each customer).

# In[26]:


#Amount spent in each transaction

retail["Amount"]=retail.Quantity*retail.UnitPrice
retail.head()


# In[28]:


#Sum of transactions for a particular customerid

#grouping based on customerid

rfm_m=retail.groupby("CustomerID")["Amount"].sum()
rfm_m.head()


# In[29]:


retail.groupby("CustomerID")["Amount"].size()


# In[30]:


#Monetary Dataframe

m=pd.DataFrame(rfm_m)
m.reset_index(inplace=True)
m.head()


# In[33]:


#for no. of transactions by a customer

#grouping only by customerId
r_in=retail.groupby("CustomerID")["InvoiceNo"].count()
r_in.head()


# In[34]:


retail[retail["CustomerID"]=="12347.0"]["InvoiceNo"].unique()
#Only 7 vs 182
#Mismatch because there is a repetition of invoice nos.


# In[42]:


#grouping for frequency

rfm_f=retail.groupby(["CustomerID","InvoiceNo"]).size()
rfm_f


# In[43]:


#grouping again to aggregate the count of the levels

rfm_f=retail.groupby(["CustomerID","InvoiceNo"]).size().groupby(level=0).count()
rfm_f


# In[44]:


#Frequency Dataframe

f=pd.DataFrame(rfm_f)
f.reset_index(inplace=True)
f.head()


# In[46]:


#Merging monetary and frequency dataframes

rfm=pd.merge(m,f,on="CustomerID", how="inner")
#inner join-only common column data is merged
rfm.columns=["CustomerID","Amount","Frequency"]
rfm.head()


# In[47]:


#For recency

retail.dtypes


# In[50]:


#change invoice date to datetime

retail["InvoiceDate"]=pd.to_datetime(retail["InvoiceDate"],format="%m/%d/%Y %H:%S")
retail.dtypes


# In[52]:


#finding max date of invoicedate

max_date=max(retail["InvoiceDate"])
max_date


# In[53]:


#column for recency

retail["DIFF"]=max_date-retail["InvoiceDate"]
retail.head()


# In[55]:


#Minimum- to get the highest recency (less diff) for each custID

rfm_r=retail.groupby("CustomerID")["DIFF"].min()
rfm_r


# In[56]:


#Recency Dataframe

r=pd.DataFrame(rfm_r)
r.reset_index(inplace=True)
r.head()


# In[57]:


#to remove time

r["Recency"]=r["DIFF"].dt.days
r.head()


# In[64]:


#dropping the diff column

r.drop(["DIFF"],axis=1,inplace=True)

#specify the index of the column as axis

r.head()


# In[65]:


#merging recency with the previous rfm

rfm=pd.merge(rfm,r,on="CustomerID", how="inner")
#inner join-only common column data is merged

rfm.columns=["CustomerID","Amount","Frequency","Recency"]
rfm.head()


# # Scaling/Data Normalization

# In[66]:


rfm_df=rfm[["Amount","Frequency","Recency"]]
#Not for custid

scaler=StandardScaler() #imported

rfm_df_scaled=scaler.fit_transform(rfm_df)
rfm_df_scaled


# # Step 4: Building the Model

# In[67]:


#Elbow method: for the optimal no. of clusters

ssd=[]  #sum of the square of distances of the datapoints

range_cluster=[2,3,4,5,6,7,8]
for i in range_cluster:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(rfm_df_scaled)
    ssd.append(kmeans.inertia_)
ssd


# In[68]:


#plotting to find the elbow

plt.plot(ssd)

#here in the x-axis, 0 corresponds 2, 1 to 3 and so on....
#elbow is at 3


# In[69]:


#Assignining no. of clusters as 3

kmeans=KMeans(n_clusters=3)
kmeans.fit(rfm_df_scaled)


# In[70]:


#find labels as it is unsupervised

kmeans.labels_


# In[71]:


#Assigning the labels to the rfm df

rfm["Label"]=kmeans.labels_
rfm.head()


# In[72]:


rfm["Label"].unique()


# In[73]:


rfm.columns=["CustomerID","Amount","Frequency","Recency","Cluster_ID"]


# In[74]:


#Boxplotting for Monetary

sns.boxplot(x="Cluster_ID",y="Amount",data=rfm)

#There are outliers


# In[75]:


#For frequency

sns.boxplot(x="Cluster_ID",y="Frequency",data=rfm)


# In[76]:


#For Recency

sns.boxplot(x="Cluster_ID",y="Recency",data=rfm)

Comparing the plots:

*Customers with high recency (less diff in days) i.e.Cluster-2 have spent more and were more frequent.
*So, Cluster-2 is very important for the company.
# In[77]:


#A list of cluster no.2 customers

imp_cust=rfm[rfm["Cluster_ID"]==2]
imp_cust


# In[78]:


imp_cust.shape

#25 important customers


# # Conclusion
* People who belong to cluster 2 have spent more than the people belonging to other clusters, followed by cluster 1.
* People who belong to cluster 2 have visited more than the people belonging to other clusters, followed by cluster 1.
* People belonging to cluster 2 are the customers with most recent visits, followed by cluster 1.

* The Retail company should focus more on the people who belong to cluster 2.