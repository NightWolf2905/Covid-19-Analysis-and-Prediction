#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv(r"C:\Users\User\Downloads\Compressed\archive\covid_19_india.csv")


# In[5]:


df.head()


# In[6]:


df['Date'].max()


# In[7]:


import plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[8]:


df.isnull().sum()


# In[10]:


df.info()


# In[12]:


df = df.drop(['Sno' , 'ConfirmedIndianNational' , 'ConfirmedForeignNational'],axis = 1)


# In[13]:


df.head()


# In[14]:


df['Active']= df['Confirmed']-df['Cured']-df['Deaths']


# In[15]:


df.head()


# In[16]:


df['Date'] = pd.to_datetime(df['Date'])


# In[17]:


df.info()


# In[18]:


india_cases = df[df['Date'] == df['Date'].max()].copy().fillna(0)
india_cases.index = india_cases["State/UnionTerritory"]
india_cases = india_cases.drop(['State/UnionTerritory', 'Time','Date'], axis=1)


# In[19]:


india_cases.head()


# In[20]:


data = pd.DataFrame(pd.to_numeric(india_cases.sum())).transpose()
data.style.background_gradient(cmap='BuGn',axis=1)


# In[24]:


Trend = df.groupby(['Date'])[['Confirmed', 'Deaths','Cured',]].sum().reset_index()


# In[25]:


Trend.head()


# In[26]:


fig = go.Figure(go.Bar(x= Trend.Date, y= Trend.Cured, name='Recovered'))
fig.add_trace(go.Bar(x=Trend.Date, y= Trend.Deaths, name='Deaths'))
fig.add_trace(go.Bar(x=Trend.Date, y= Trend.Confirmed, name='Confirmed'))

fig.update_layout(barmode='stack',legend_orientation="h",legend=dict(x= 0.3, y=1.1),
                 paper_bgcolor='white',
                 plot_bgcolor = "white",)
fig.show()


# In[27]:


import plotly.express as px


# In[28]:


def horizontal_bar_chart(data, x, y, title, x_label, y_label, color):
    fig = px.bar(df, x=x, y=y, orientation='h', title=title, 
                 labels={x.name: x_label,
                         y.name: y_label}, color_discrete_sequence=[color])
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.show()


# In[29]:


top_10_death_states = india_cases.sort_values('Deaths',ascending = False)[:10]
horizontal_bar_chart(top_10_death_states,top_10_death_states.Deaths, top_10_death_states.index,
                     'Top 10 States with most deaths','Number of deaths(In Thousands)','State Name','Orange')


# In[30]:


top_10_confirmed_states = india_cases.sort_values('Confirmed', ascending=False)[:10]
horizontal_bar_chart(top_10_confirmed_states,top_10_confirmed_states.Confirmed, top_10_confirmed_states.index,
            'Top 10 Indian States (Confirmed Cases)', 'Number of Confirmed cases (in Thousands)','States Name','blue')


# In[32]:


top_10_recovered_states = india_cases.sort_values('Cured', ascending=False)[:10]
horizontal_bar_chart(top_10_recovered_states, top_10_recovered_states.Cured, top_10_recovered_states.index,
                    'Top 10 States (Cured Cases)', 'Number of Cured cases (in Thousands)', 'States Name', 'Purple')


# In[33]:


vaccination = pd.read_csv(r"C:\Users\User\Downloads\Compressed\archive\covid_vaccine_statewise.csv")


# In[34]:


vaccination.tail()


# In[35]:


vaccination['Total Vaccinatons'] = vaccination['First Dose Administered']+vaccination['Second Dose Administered']

#Renaming columns
vaccination.rename(columns = {'Updated On':'Date'}, inplace = True)


# In[36]:


Maharashtra = vaccination[vaccination["State"]=="Maharashtra"]
fig = px.line(Maharashtra,x="Date",y="Total Vaccinatons",title="Vaccination till date in Maharashtra")  
fig.update_xaxes(rangeslider_visible=True) 


# In[47]:


pip install prophet plotly


# In[48]:


import prophet
from prophet.plot import plot_plotly, add_changepoints_to_plot
from plotly.offline import iplot, init_notebook_mode


# In[51]:


from prophet import Prophet


# In[52]:


model = Prophet()


# In[53]:


Confirmed = Trend.loc[:, ['Date', 'Confirmed']] 
Confirmed.tail()


# In[54]:


Cured = Trend.loc[:, ['Date', 'Cured']] 
Cured.tail()


# In[55]:


Confirmed.columns = ['ds', 'y']
model.fit(Confirmed)


# In[56]:


future = model.make_future_dataframe(periods=60) # helper function to extend the dataframe for specified days
future.tail()


# In[57]:


forecast_india_conf = model.predict(future)
forecast_india_conf


# In[58]:


fig = plot_plotly(model, forecast_india_conf) 

fig.update_layout(template='plotly_white')

iplot(fig) 


# END

# In[ ]:




