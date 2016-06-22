# -*- coding: utf-8 -*-
"""
Created on Tue June 20 15:47:22 2016

@author: Chris
"""
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
import statsmodels.formula.api as smf
import scipy.stats

#from IPython.display import display
%matplotlib inline

#bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

#Set Pandas to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set Pandas to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

#data here will act as the data frame containing the Mars crater data
data = pandas.read_csv('D:\\Coursera\\marscrater_pds.csv', low_memory=False)

#convert the latitude and diameter columns to numeric and ejecta morphology is categorical
data['LATITUDE_CIRCLE_IMAGE'] = pandas.to_numeric(data['LATITUDE_CIRCLE_IMAGE'])
data['DIAM_CIRCLE_IMAGE'] = pandas.to_numeric(data['DIAM_CIRCLE_IMAGE'])
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].astype('category')

#Any crater with no designated morphology will be replaced with NaN
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].replace(' ',numpy.NaN)

#Here we will subset out craters with the three ejecta morphologies we are interested in
morphofinterest = ['Rd','SLEPS','SLERS']
data = data.loc[data['MORPHOLOGY_EJECTA_1'].isin(morphofinterest)]

#We now look at our data now that we've extracted the data we wish to use
data.describe()

#Because of the bug in seaborn plotting, we now extract the data from the original data frame as arrays and make a new data frame
latitude = numpy.array(data['LATITUDE_CIRCLE_IMAGE'])
diameter = numpy.array(data['DIAM_CIRCLE_IMAGE'])
morphology = numpy.array(data['MORPHOLOGY_EJECTA_1'])
data2 = pandas.DataFrame({'LATITUDE':latitude,'DIAMETER':diameter,'MORPHOLOGY_EJECTA_1':morphology})

#We now loop through the three morphologies, plotting them out, and printing a summary of the correlation
summarycorrelations = pandas.DataFrame(
    columns=('MORPHOLOGY_EJECTA_1','R**2','F-STATISTIC','P-VALUE','EXPLANATORY_COEFFICIENT','INTERCEPT'))

for a0 in morphofinterest:
    print('MORPHOLOGY EJECTA: ' + a0)
    datatemp = data2.loc[(data2['MORPHOLOGY_EJECTA_1']==a0)]
    tempmodel = smf.ols(formula='DIAMETER ~ LATITUDE',data=datatemp).fit()
    print(tempmodel.summary())
    print('\n')
    #create a list with all the statistical results parameters of interest
    templist = [a0,tempmodel.rsquared,tempmodel.fvalue,tempmodel.f_pvalue,tempmodel.params[1],tempmodel.params[0]]
    #append this list to a dataframe for easy to see summary
    summarycorrelations.loc[morphofinterest.index(a0)] = templist

print('Summary statistical results for linear regression of crater diameter vs. latitude')
print(summarycorrelations)

print('We now do linear regression for diameter onto latitude for the different ejecta morphologyies.')
seaborn.lmplot(x='LATITUDE',y='DIAMETER',col='MORPHOLOGY_EJECTA_1',hue='MORPHOLOGY_EJECTA_1',data=data2)

