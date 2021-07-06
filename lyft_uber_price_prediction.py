#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


rides_df=pd.read_csv('cab_rides.csv')
rides_df.head()


# In[3]:


weather_df=pd.read_csv('weather.csv')
weather_df.head(5)


# In[4]:


rides_df.info()


# In[5]:


weather_df.info()


# In[6]:


rides_df.isna().sum()


# In[7]:


weather_df.isna().sum()


# In[8]:


weather_df=weather_df.fillna(0)


# In[9]:


rides_df['date']=pd.to_datetime(rides_df['time_stamp']/ 1000,unit='s')

weather_df['date']=pd.to_datetime(weather_df['time_stamp'],unit='s')


# In[10]:


rides_df['merge_date'] = rides_df['source'].astype(str) +" - "+ rides_df['date'].dt.date.astype("str") +" - "+ rides_df['date'].dt.hour.astype("str")
weather_df['merge_date'] = weather_df['location'].astype(str) +" - "+ weather_df['date'].dt.date.astype("str") +" - "+ weather_df['date'].dt.hour.astype("str")


# In[11]:


weather_df.index = weather_df['merge_date']
df_joined = rides_df.join(weather_df,on=['merge_date'],rsuffix ='_w')


# In[12]:


df_joined.info()


# In[13]:


df_joined['id'].value_counts()


# In[14]:


df_joined[df_joined['id']=='6fa6c718-15cf-48a0-aa4f-49efa5d6974e'].iloc[:,10:22]


# In[15]:


id_group = pd.DataFrame(df_joined.groupby('id')['temp','clouds','pressure','rain','humidity','wind'].mean())
df_rides_weather=rides_df.join(id_group,on = ['id'])


# In[16]:


df_rides_weather['Month']=df_rides_weather['date'].dt.month
df_rides_weather['Hour']=df_rides_weather['date'].dt.hour
df_rides_weather['Day']=df_rides_weather['date'].dt.strftime('%A')
df_rides_weather.tail(5)


# In[17]:


import matplotlib.pyplot as plt
uber_day_count =df_rides_weather[df_rides_weather['cab_type']=='Uber']['Day'].value_counts()
uber_day_count=uber_day_count.reindex(index = ['Friday','Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday'])
lyft_day_count =df_rides_weather[df_rides_weather['cab_type']=='Lyft']['Day'].value_counts()
lyft_day_count=lyft_day_count.reindex(index = ['Friday','Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday'])

fig , ax = plt.subplots(figsize = (10,10))
ax.plot(uber_day_count.index, uber_day_count,label='Uber')
ax.plot(lyft_day_count.index, lyft_day_count,label='Lyft')
ax.set(ylabel = 'Number of Rides',xlabel = 'Weekdays')
ax.legend()
plt.show()


# In[18]:


fig , ax = plt.subplots(figsize= (12,12))
ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('Hour').Hour.count().index, df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('Hour').Hour.count(), label = 'Lyft')
ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('Hour').Hour.count().index, df_rides_weather[df_rides_weather['cab_type'] =='Uber'].groupby('Hour').Hour.count(), label = 'Uber')
ax.legend()
ax.set(xlabel = 'Hours', ylabel = 'Number of Rides')
plt.xticks(range(0,24,1))
plt.show()


# In[19]:


uber_order =[ 'UberPool', 'UberX', 'UberXL', 'Black','Black SUV','WAV' ]
lyft_order = ['Shared', 'Lyft', 'Lyft XL', 'Lux', 'Lux Black', 'Lux Black XL']
fig, ax = plt.subplots(2,2, figsize = (20,15))
ax1 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].name, y = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].price , ax = ax[0,0], order = uber_order)
ax2 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].name, y = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].price , ax = ax[0,1], order = lyft_order)
ax3 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('name').name.count().index, y = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('name').name.count(), ax = ax[1,0] ,order = uber_order)
ax4 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('name').name.count().index, y = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('name').name.count(), ax = ax[1,1],order = lyft_order)
for p in ax1.patches:
    ax1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
for p in ax2.patches:
    ax2.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax1.set(xlabel = 'Type of Service', ylabel = 'Average Price')
ax2.set(xlabel = 'Type of Service', ylabel = 'Average Price')
ax3.set(xlabel = 'Type of Service', ylabel = 'Number of Rides')
ax4.set(xlabel = 'Type of Service', ylabel = 'Number of Rides')
ax1.set_title('The Uber Average Prices by Type of Service')
ax2.set_title('The Lyft Average Prices by Type of Service')
ax3.set_title('The Number of Uber Rides by Type of Service')
ax4.set_title('The Number of Lyft Rides by Type of Service')
plt.show()


# In[20]:


fig , ax = plt.subplots(figsize = (12,12))
ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('distance').price.mean().index, df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('distance')['price'].mean(), label = 'Lyft')
ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('distance').price.mean().index, df_rides_weather[df_rides_weather['cab_type'] =='Uber'].groupby('distance').price.mean(), label = 'Uber')
ax.set_title('The Average Price by distance', fontsize= 15)
ax.set(xlabel = 'Distance', ylabel = 'Price' )
ax.legend()
plt.show()


# In[21]:


fig, ax = plt.subplots(1,2 , figsize = (20,5))
for i,col in enumerate(df_rides_weather[df_rides_weather['cab_type'] == 'Uber']['name'].unique()):
    ax[0].plot(df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean().index, df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean(), label = col)
ax[0].set_title('Uber Average Prices by Distance')
ax[0].set(xlabel = 'Distance in Mile', ylabel = 'Average price in USD')
ax[0].legend()
for i,col in enumerate(df_rides_weather[df_rides_weather['cab_type'] == 'Lyft']['name'].unique()):
    ax[1].plot(df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean().index, df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean(), label = col)
ax[1].set(xlabel = 'Distance in Mile', ylabel = 'Average price in USD')
ax[1].set_title('Lyft Average Prices by Distance')
ax[1].legend()
plt.show()


# In[22]:


rides_df=rides_df.drop('merge_date',axis=1)
rides_df=rides_df.drop('date',axis=1)
weather_df=weather_df.drop('merge_date',axis=1)
weather_df=weather_df.drop('date',axis=1)


# In[23]:


weather_df


# In[24]:


weather_df.groupby('location').mean()

avg_weather_df = weather_df.groupby('location').mean().reset_index(drop=False)
avg_weather_df = avg_weather_df.drop('time_stamp', axis=1)

source_weather_df= avg_weather_df.rename(columns={'location':'source','temp':'source_temp','clouds':'source_clouds','pressure':'source_pressure','rain':'source_rain','hummidity':'source_hummidity','wind':'source_wind'})
source_weather_df


# In[25]:


destination_weather_df = avg_weather_df.rename(
    columns={
        'location': 'destination',
        'temp': 'destination_temp',
        'clouds': 'destination_clouds',
        'pressure': 'destination_pressure',
        'rain': 'destination_rain',
        'humidity': 'destination_humidity',
        'wind': 'destination_wind'
    }
)

destination_weather_df


# In[26]:


data = rides_df    .merge(source_weather_df, on='source')    .merge(destination_weather_df, on='destination')

data


# In[27]:


cat=data.dtypes[data.dtypes=='O'].index.values
cat


# In[28]:


from collections import Counter
for i in cat:
    print('Coulum : ',i)
    print('Count of classes : ',data[i].nunique())
    print(Counter(data[i]))
    print('*'*80)


# In[29]:


data.dtypes[data.dtypes!='O'].index.values


# In[30]:


data1=data.copy()
from sklearn.preprocessing import LabelEncoder
x="*"
for i in cat:
    print("LABEL ENCODING OF : ",i)
    LE=LabelEncoder()
    print(Counter(data[i]))
    data[i]=LE.fit_transform(data[i])
    print(Counter(data[i]))
    print('*'*80)


# In[31]:


data.info()


# In[32]:



data['price'] = data['price'].fillna(value=data["price"].mean())

x=data.drop(['price','distance','time_stamp','surge_multiplier','source_temp','id','source_clouds','source_pressure','source_rain','humidity','source_wind','destination_temp','destination_clouds','destination_pressure','destination_rain','destination_humidity','destination_wind'],axis=1)
x=pd.DataFrame(x)

y=data['price']
y=pd.DataFrame(y)


# In[33]:


from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te= train_test_split(x, y, train_size=0.7, shuffle=True, random_state=1)
print(x_tr.shape)
print(x_te.shape)


# In[34]:


x_tr.describe()


# In[35]:


from sklearn.ensemble import RandomForestRegressor
import random
rand=RandomForestRegressor(n_estimators=20,random_state=42,n_jobs=-1,max_depth=5)
random.seed('42')
rand.fit(x_tr,y_tr)


# In[36]:


y_pred = rand.predict(x_te)
print(y_pred)


# In[37]:


from sklearn.metrics import r2_score
print((r2_score(y_te,y_pred)).round(2))


# In[38]:


pred=rand.predict([['0.556559','3.000000','5.879777','5.000000','9.000000']])
print(pred.round(2))


# In[39]:


import pickle
pickle.dump(rand,open("model.pkl","wb"))

