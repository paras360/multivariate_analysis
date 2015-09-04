
# coding: utf-8


import pandas as pd 
import collections as c 
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls 
from plotly.graph_objs import * 
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm 


#loads the data
loansdata = pd.read_csv('LoanStats3d.csv')

#cleaning up the data in the int rate column
clean_interest_rate = loansdata['int_rate']
clean_interest_rate = clean_interest_rate.map(lambda x: str(x))
clean_interest_rate = clean_interest_rate.map(lambda x: x.rstrip('%'))
clean_interest_rate = clean_interest_rate.map(lambda x: float(x))

#place clean interest rate into the data frame 
loansdata['int_rate'] = clean_interest_rate

#creates a new datafraame with just interest rate and annual income to begin the modeling of those two variables
new_df = loansdata[['annual_inc', 'int_rate']]
new_df.dropna(inplace=True)

#defining variables for input into model 

interest_rate = new_df['int_rate']
annual_income = new_df['annual_inc']


#defining variables
y = np.matrix(interest_rate).transpose()
x1 = np.matrix(annual_income).transpose()
x = np.column_stack([x1])
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print results.summary()


#creates a new dataframe with homeownership. i will map the data on top of itself and assign a number 0 to rent and 1 to mortgage
new_df_with_home_ownership = loansdata[['annual_inc', 'int_rate', 'home_ownership']]
new_df_with_home_ownership['home_ownership'] = new_df_with_home_ownership['home_ownership'].map({'RENT':0, 'MORTGAGE':1})

new_df_with_home_ownership.dropna(inplace=True)

#defining variables for input into the new mode. this is now called new_df_with_Home_ownership and it represents home ownership dataframe

interest_rate_new = new_df_with_home_ownership['int_rate']
annual_income_new = new_df_with_home_ownership['annual_inc']
home_ownership_new = new_df_with_home_ownership['home_ownership']

#defining variables for input into a multiple independent variable OLS model. this model now includes home ownership as a dummy variable
y = np.matrix(interest_rate_new).transpose()
x1 = np.matrix(annual_income_new).transpose()
x2 = np.matrix(home_ownership_new).transpose()
x = np.column_stack([x1, x2])
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print results.summary()


#define the interaction as annual income multipled by a dummy variable 
interaction_new = annual_income_new * home_ownership_new


y = np.matrix(interest_rate_new).transpose()
x1 = np.matrix(annual_income_new).transpose()
x2 = np.matrix(home_ownership_new).transpose()
x3 = np.matrix(interaction_new).transpose()
x = np.column_stack([x1, x2, x3])
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print results.summary()





