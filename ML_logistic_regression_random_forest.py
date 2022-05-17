#================================================
# Author:		Zuzana Knapekova
# Create date:  2022-03-29
# ===============================================


#%%
# Import packages
import pandas as pd
from pandas import concat
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, Lasso
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess
from scipy.stats import chi2_contingency
from patsy import dmatrices


#%%
# Functions
def Frequencies(table, column):
    # Function returns table with relative and absolute frequencies
    freq_table = concat([table[column].value_counts(), table[column].value_counts(normalize=True)], axis=1)
    freq_table.columns=['AbsFreq','RelFreq']
    return freq_table


def Missing_values(table, column):
    #Function returns information about number of missing values
    df1_notNan = table[table[column].notna()]
    print("N. rows with", column, "not null:", df1_notNan.shape[0])

    df1_Nan  = table[table[column].isna()]
    print("N. rows with", column ,"null:", df1_Nan.shape[0])


def ChiSqTest(table, column1, column2, alpha):
    FreqTable=pd.crosstab(table[column1], table[column2], margins=True) 
    ChiSqResult = chi2_contingency(FreqTable)
    if ChiSqResult[1]>alpha:
        print('The P-Value of the ChiSq Test of', column1, 'and',  column2,  'is', ChiSqResult[1], 'and we fail to reject the null hypothesis at the level of significance', str(alpha)) 
    else:
        print('The P-Value of the ChiSq Test of', column1, 'and',  column2,  'is', ChiSqResult[1], 'and we reject the null hypothesis at the level of significance', str(alpha)) 


def Cross_validation(X, Y, model):
    #Function performs k-fold cross validation and returns array with mean square errors
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	MSE = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=cv)
	print('Average MSE for the model is %.2f'  %  np.mean(abs(MSE)))

def Get_dummies(X, source_table):
    # Function changes categorical variables to multiple binomial dummy variables
    df_agecat_dummies=pd.get_dummies(source_table['agecat'])
    df_agecat_dummies.columns = ['agecat1','agecat2','agecat3','agecat4','agecat5','agecat6']

    df_area_dummies=pd.get_dummies(source_table['area'])
    df_area_dummies.columns = ['area1','area2','area3','area4','area5','area6']

    df_veh_body_dummies=pd.get_dummies(source_table['veh_body'])
    df_veh_body_dummies.columns = ['veh_body1','veh_body2','veh_body3','veh_body4','veh_body5','veh_body6', 'veh_body7', 'veh_body8', 
    'veh_body9', 'veh_body10', 'veh_body11', 'veh_body12', 'veh_body13']

    dummies = pd.concat([df_agecat_dummies,df_area_dummies,df_veh_body_dummies ],axis=1)
    X_concat = pd.concat([X,dummies],axis=1)
    X_final=X_concat.drop(['agecat1', 'area1', 'veh_body1'], axis=1)
    return X_final

#%%
# Data load
df1 = pd.read_csv("auto_policies_2021.csv")
df2 = pd.read_csv("auto_potential_customers_2022.csv")
#%%
###### Analysis of training data auto_policies_2021.csv ######


###############################################################################################################
# Exploratory analysis 
###############################################################################################################

df1.info()
df1.describe()
df1.nunique()
df1.dtypes

#%%
# Prepare input - changing lables to numeric since sklearn package for model estimation cannot work with 'text' variables 
df1['gender_num'] = np.where(df1['gender']=='F', 1,0)
df1['gender_num']=df1['gender_num'].astype(object)

df1['pol_eff_date']=pd.to_datetime(df1['pol_eff_dt'])
df1['date_of_birth_dt']=pd.to_datetime(df1['date_of_birth'])
df1['area']=df1['area'].astype(object)
df1['veh_body']=df1['veh_body'].astype(object)

# %%
#Create frequency tables for categorical variables
Frequencies(df1, 'numclaims')
#%%
Frequencies(df1, 'gender')
#%%
Frequencies(df1, 'agecat')
#%%
Frequencies(df1, 'area')
#%%
Frequencies(df1, 'veh_age')
#%%
Frequencies(df1, 'veh_body')
#%%
Frequencies(df1, 'claim_office')

# %%
#Checking claim amount and other continuous variables
# %%
#Credit score#
sns.distplot(df1['credit_score'], hist=True, hist_kws={'edgecolor':'black'}).set(xlabel='Credit score',ylabel='Density',
title='Credit score')
#%%
sns.boxplot(x=df1['numclaims'], y=df1['credit_score']).set(xlabel='Number of claims', ylabel='Credit score',
title='Credit score by number of claims')
# %%
#Vehicle value#
sns.distplot(df1['veh_value'], hist=True, hist_kws={'edgecolor':'black'}).set(xlabel='Vehicle value',ylabel='Density',
title='Vehicle value')
#%%
sns.boxplot(x=df1['numclaims'], y=df1['veh_value']).set(xlabel='Number of claims', ylabel='Vehicle value',
title='Vehicle value by number of claims')
# %%
#Traffic index
sns.distplot(df1['traffic_index'], hist=True, hist_kws={'edgecolor':'black'}).set(xlabel='Traffic index',ylabel='Density',
title='Traffic index')
#%%
sns.boxplot(x=df1['numclaims'], y=df1['traffic_index']).set(xlabel='Number of claims', ylabel='Traffic index',
title='Traffic index by number of claims')

# %%
#Claim amount
sns.boxplot(y=df1[df1['claimcst0']>0]['claimcst0']).set(ylabel='Claim amount in $',
title='Claim amount')
#%%
sns.distplot(df1[df1['claimcst0']>0]['claimcst0'], hist=True, hist_kws={'edgecolor':'black'}).set(xlabel='Claim amount',ylabel='Density',
title='Claim amount')
#%%
sns.boxplot(x=df1[df1['claimcst0']>0]['numclaims'], y=df1[df1['claimcst0']>0]['claimcst0']).set(xlabel='Number of claims', ylabel='Claim amount in $',
title='Claim amount by number of claims')
#%%
sns.boxplot(x=df1[df1['claimcst0']>0]['numclaims'], y=df1[df1['claimcst0']>0]['claimcst0'], hue=df1[df1['claimcst0']>0]['gender']).set(xlabel='Number of claims', ylabel='Claim amount in $',
title='Claim amount by number of claims and gender')
#%%
sns.boxplot(x=df1[df1['claimcst0']>0]['numclaims'], y=df1[df1['claimcst0']>0]['claimcst0'], hue=df1[df1['claimcst0']>0]['agecat']).set(xlabel='Number of claims', ylabel='Claim amount in $',
title='Claim amount by number of claims and age category')
plt.legend(loc=2, bbox_to_anchor=(1,1))
#%%
sns.boxplot(x=df1[df1['claimcst0']>0]['numclaims'], y=df1[df1['claimcst0']>0]['claimcst0'], hue=df1[df1['claimcst0']>0]['area']).set(xlabel='Number of claims', ylabel='Claim amount in $',
title='Claim amount by number of claims and area')
order = [3,2,0,1,4,5]
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=2, bbox_to_anchor=(1,1))
#%%
sns.boxplot(x=df1[df1['claimcst0']>0]['numclaims'], y=df1[df1['claimcst0']>0]['claimcst0'], hue=df1[df1['claimcst0']>0]['veh_age']).set(xlabel='Number of claims', ylabel='Claim amount in $',
title='Claim amount by number of claims and vehicle age')
plt.legend(loc=2, bbox_to_anchor=(1,1))
# %%
###############################################################################################################
# MISSING VALUES
###############################################################################################################
#%%
Missing_values(df1, 'veh_age') 
Missing_values(df1, 'veh_body') 
Missing_values(df1, 'traffic_index') 
Missing_values(df1, 'area') 
Missing_values(df1, 'credit_score') 
Missing_values(df1, 'gender') 
Missing_values(df1, 'agecat') 
Missing_values(df1, 'veh_value') 
Missing_values(df1, 'claim_office') 
Missing_values(df1, 'numclaims') 
Missing_values(df1, 'claimcst0') 
Missing_values(df1, 'pol_eff_dt')


# %% 
# calculate missing agecat based on date of birth - checked the interval from the rest of data
for j in range(df1.shape[0]):
    if pd.isna(df1['agecat'][j]):
        if df1['date_of_birth_dt'][j]<=pd.to_datetime('1949-12-31'):
            df1['agecat'][j]=6
        elif (df1['date_of_birth_dt'][j]>pd.to_datetime('1949-12-31')) & (df1['date_of_birth_dt'][j]<=pd.to_datetime('1959-12-31')):
            df1['agecat'][j]=5
        elif (df1['date_of_birth_dt'][j]>pd.to_datetime('1959-12-31')) & (df1['date_of_birth_dt'][j]<=pd.to_datetime('1969-12-31')):
            df1['agecat'][j]=4
        elif (df1['date_of_birth_dt'][j]>pd.to_datetime('1969-12-31')) & (df1['date_of_birth_dt'][j]<=pd.to_datetime('1979-12-31')):
            df1['agecat'][j]=3
        elif (df1['date_of_birth_dt'][j]>pd.to_datetime('1979-12-31')) & (df1['date_of_birth_dt'][j]<=pd.to_datetime('1989-12-31')):
            df1['agecat'][j]=2
        elif (df1['date_of_birth_dt'][j]>pd.to_datetime('1989-12-31')) & (df1['date_of_birth_dt'][j]<=pd.to_datetime('1999-12-31')):
            df1['agecat'][j]=1

df1['agecat']=df1['agecat'].astype(object)

# %%
# missing values in credit score and traffic index were replaced by median

df1['credit_score'].fillna(df1['credit_score'].median(), inplace=True)
df1['traffic_index'].fillna(df1['traffic_index'].median(), inplace=True)

# %%
###############################################################################################################
# REDUCE NUMBER OF CATEGORIES
###############################################################################################################

df1['numclaims_new'] = np.where(df1['numclaims'].isin([1,2,3,4,5]), 1,0)
Frequencies(df1, 'numclaims_new') # the table shows imbalanced classes


# %%
##############################################################################################################
# Checking the relationship between dependent variable and independent variables
###############################################################################################################
ct=pd.crosstab(df1['numclaims_new'], df1['agecat'])
ax=ct.plot(kind='bar', stacked=True, rot=0)
ax.legend(title='Age category', bbox_to_anchor=(1, 1.02), loc='upper left')
for c in ax.containers:
    ax.bar_label(c, label_type='center')

# %%
# Frequency tables for numclaims_new:
pd.crosstab(df1['numclaims_new'], df1['gender'], margins=True)
#%%
pd.crosstab(df1['numclaims_new'], df1['area'], margins=True)
#%%
pd.crosstab(df1['numclaims_new'], df1['veh_age'], margins=True)
#%%
pd.crosstab(df1['numclaims_new'], df1['veh_body'], margins=True)
#%%
# Boxplots for continuous variables by number of claims
sns.boxplot(x=df1['numclaims_new'], y=df1['credit_score']).set(xlabel='Claims', ylabel='Credit score',
title='Credit score by claim')
plt.legend(loc=2, bbox_to_anchor=(1,1))
# %%
sns.boxplot(x=df1['numclaims_new'], y=df1['traffic_index']).set(xlabel='Claims', ylabel='Traffic index',
title='Traffic index by claim')
plt.legend(loc=2, bbox_to_anchor=(1,1))
# %%
sns.boxplot(x=df1['numclaims_new'], y=df1['veh_value']).set(xlabel='Claims', ylabel='Vehicle value',
title='Vehicle value by claim')
plt.legend(loc=2, bbox_to_anchor=(1,1))

#%%
# show heatmap
# we see high correlation between vehicle age and vehicle value - let's check if multicolinearity is present
sns.heatmap(df1.drop(['annual_premium', 'pol_eff_dt'], axis=1).corr(), annot= True)
# %%
# Check for multicolinearity using variance inflation factor
y, X = dmatrices('numclaims_new ~ veh_value+traffic_index+credit_score+veh_age', data=df1, return_type='dataframe')
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns
vif
# %%
# Chi-square test to test association between numclaims_new and categorical variables

ChiSqTest(df1, 'numclaims_new', 'gender', 0.05)
ChiSqTest(df1, 'numclaims_new', 'agecat', 0.05)
ChiSqTest(df1, 'numclaims_new', 'agecat', 0.05)
ChiSqTest(df1, 'numclaims_new', 'veh_age', 0.05)
ChiSqTest(df1, 'numclaims_new', 'veh_body', 0.05)




#%%
##############################################################################################################
# Data preparation 
###############################################################################################################

X = df1.drop(['veh_body', 'area', 'agecat', 'gender','pol_number', 'pol_eff_dt', 'date_of_birth', 'claim_office', 'numclaims', 'numclaims_new', 'pol_eff_date', 'date_of_birth_dt', 'annual_premium', 'claimcst0' ],axis=1)
Y=df1['numclaims_new']
X=Get_dummies(X, df1)

#%%
# splitting into training and validating dataset
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=11)

#%%
##############################################################################################################
# Logisitic regression model
###############################################################################################################

# estimation
log_reg = sm.Logit(endog=Y_train.astype(float), exog=X_train.astype(float)).fit()
log_reg.summary()

#%%
#prediction
log_model = LogisticRegression(max_iter=10000)
log_model.fit(X_train, Y_train)
Y_train_predict = log_model.predict(X_train)
Y_predict = log_model.predict(X_valid)
#%%
#evaluation
# k-fold cross validation for average MSE comparison
Cross_validation(X, Y, log_model)
# comparing training and validating MSE:
print("MSE for model on train data: %.2f"  % np.mean((Y_train - Y_train_predict) ** 2))  
print("MSE for model on validation data: %.2f"  %np.mean((Y_valid - Y_predict) ** 2))

#%%
auc=metrics.roc_auc_score(Y_valid, Y_predict)
gini=2*auc-1

print('Accuracy for logistic regression: %.2f'  % metrics.accuracy_score(Y_valid,Y_predict))
print('AUC for logistic regression: %.2f' % auc)
print('Gini coefficient: %.2f' % gini)
tn, fp, fn, tp = metrics.confusion_matrix(Y_valid, Y_predict).ravel()
specificity = tn / (tn+fp)
print('Specificity: %.2f' % specificity )
sensitivity1 = tp / (tp+fn)
print('Sensitivity: %.2f' % sensitivity1 )
metrics.plot_confusion_matrix(log_model, X_valid, Y_valid)



#%%
##############################################################################################################
# Random forest
###############################################################################################################

forest = RandomForestClassifier(n_estimators=25, random_state=11, 
                                 min_samples_leaf=10)
forest.fit(X_train, Y_train)
Y_train_predict_forest = forest.predict(X_train)
Y_predict_forest = forest.predict(X_valid)

#%%
#evaluation
auc=metrics.roc_auc_score(Y_valid, Y_predict_forest)
gini=2*auc-1
print('Accuracy for random forest: %.2f'  % metrics.accuracy_score(Y_valid,Y_predict_forest))
print('AUC for random forest: %.2f' % auc)
print('Gini coefficient: %.2f' % gini)

tn, fp, fn, tp = metrics.confusion_matrix(Y_valid, Y_predict_forest).ravel()
specificity = tn / (tn+fp)
print('Specificity: %.2f' % specificity )
sensitivity1 = tp / (tp+fn)
print('Sensitivity: %.2f' % sensitivity1 )
metrics.plot_confusion_matrix(forest, X_valid, Y_valid)

#%%
# k-fold cross validation for average MSE comparison
Cross_validation(X, Y, forest)
# comparing training and validating MSE:
print("MSE for model on train data : %.2f"  % np.mean((Y_train - Y_train_predict) ** 2))  
print("MSE for model on validation data : %.2f"  %np.mean((Y_valid - Y_predict) ** 2))
#%%
# For logistic regression get optimal threshold based on maximazing (TPR-FPR) 
y_pred_proba = log_model.predict_proba(X_valid)[::,1]
fpr, tpr, thresholds = metrics.roc_curve(Y_valid,  y_pred_proba)

diff=tpr - fpr
optimal_idx1 = np.argmax(diff)
print('Best Threshold=%f, TPR-FPR=%.3f' % (thresholds[optimal_idx1], diff[optimal_idx1]))


plt.plot(fpr,tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
#%%
# setting new threshold 
Y_predict2 = (log_model.predict_proba(X_valid)[:,1] >= thresholds[optimal_idx1]).astype(bool) # set threshold as 0.3
auc=metrics.roc_auc_score(Y_valid, Y_predict2)
gini=2*auc-1
print('Accuracy for logistic regression with new threshold: %.2f'  % metrics.accuracy_score(Y_valid,Y_predict2))
print('AUC for logistic regression with new threshold: %.2f' % auc)

tn, fp, fn, tp = metrics.confusion_matrix(Y_valid, Y_predict2).ravel()
specificity = tn / (tn+fp)
print('Specificity: %.2f' % specificity )
sensitivity1 = tp / (tp+fn)
print('Sensitivity: %.2f' % sensitivity1 )
print(metrics.confusion_matrix(Y_valid, Y_predict2))


#%%
###############################################################################################################
# Cost per claim analysis
###############################################################################################################
# Charts for claim amount
sns.boxplot(x=df1[df1['claimcst0']>0]['numclaims_new'], y=df1[df1['claimcst0']>0]['claimcst0'], hue=df1[df1['claimcst0']>0]['agecat']).set(xlabel='Number of claims', ylabel='Claim amount in $',
title='Claim amount by  age category')
plt.legend(loc=2, bbox_to_anchor=(1,1))
# %%
sns.boxplot(x=df1[df1['claimcst0']>0]['numclaims_new'], y=df1[df1['claimcst0']>0]['claimcst0'], hue=df1[df1['claimcst0']>0]['gender']).set(xlabel='Number of claims', ylabel='Claim amount in $',
title='Claim amount by gender')
plt.legend(loc=2, bbox_to_anchor=(1,1))
# %%
sns.boxplot(x=df1[df1['claimcst0']>0]['numclaims_new'], y=df1[df1['claimcst0']>0]['claimcst0'], hue=df1[df1['claimcst0']>0]['area']).set(xlabel='Number of claims', ylabel='Claim amount in $',
title='Claim amount by area')
plt.legend(loc=2, bbox_to_anchor=(1,1))
# %%
sns.boxplot(x=df1[df1['claimcst0']>0]['numclaims_new'], y=df1[df1['claimcst0']>0]['claimcst0'], hue=df1[df1['claimcst0']>0]['veh_age']).set(xlabel='Number of claims', ylabel='Claim amount in $',
title='Claim amount by vehicle age')
plt.legend(loc=2, bbox_to_anchor=(1,1))

# %%
sns.boxplot(x=df1[df1['claimcst0']>0]['numclaims_new'], y=df1[df1['claimcst0']>0]['claimcst0'], hue=df1[df1['claimcst0']>0]['veh_body']).set(xlabel='Number of claims', ylabel='Claim amount in $',
title='Claim amount by vehicle body')
plt.legend(loc=2, bbox_to_anchor=(1,1))
#%%
# Calculate cost per claim
df1b=df1[df1['numclaims']>0]
df1b['cost_per_claim']=df1b['claimcst0'].div(df1b['numclaims'])
sns.distplot(df1b['cost_per_claim'], hist=True, hist_kws={'edgecolor':'black'})

#%%
df1b['cost_per_claim'].describe()
# check outliers
df1b.sort_values(by=['cost_per_claim'], ascending=False).head(10)
#%%
#applying logarithm and using log-level model since data are rightly skewed
df1b['log_cost_per_claim']=np.log(df1b['cost_per_claim'])
sns.distplot(df1b['log_cost_per_claim'], hist=True, hist_kws={'edgecolor':'black'})

#%%
# checking functional forms of continuous covariates
#vehicle value
plt.scatter(df1b['veh_value'], df1b['log_cost_per_claim'], c='b',alpha=0.5)
lowess_x, lowess_y=sm_lowess(df1b['log_cost_per_claim'], df1b['veh_value'],  frac=1./10.,  return_sorted = True).T
plt.plot(lowess_x, lowess_y, color='black')
plt.ylabel("Log Cost per claim")
plt.xlabel("Vehicle value")

#%%
#traffic index
plt.scatter(df1b['traffic_index'], df1b['log_cost_per_claim'], c='b',alpha=0.5)
lowess_x, lowess_y=sm_lowess(df1b['log_cost_per_claim'], df1b['traffic_index'],  frac=1./10., return_sorted = True).T
plt.plot(lowess_x, lowess_y, color='black')
plt.ylabel("Log Cost per claim")
plt.xlabel("Traffic index")

#%%
#credit score
plt.scatter(df1b['credit_score'], df1b['log_cost_per_claim'], c='b',alpha=0.5)
lowess_x, lowess_y=sm_lowess(df1b['log_cost_per_claim'], df1b['credit_score'],  frac=1./10.,  return_sorted = True).T
plt.plot(lowess_x, lowess_y, color='black')
plt.ylabel("Log Cost per claim")
plt.xlabel("Credit score")

#%%
##############################################################################################################
# Data preparation 
###############################################################################################################
X = df1b.drop(['veh_body', 'area', 'agecat', 'gender','pol_number', 'pol_eff_dt', 'date_of_birth', 'claim_office', 'numclaims', 'numclaims_new', 'pol_eff_date', 'date_of_birth_dt', 'annual_premium', 'claimcst0', 'cost_per_claim', 'log_cost_per_claim'  ],axis=1)
Y=df1b['log_cost_per_claim']
X=Get_dummies(X, df1b)
#add traffic index squared
X['traffic_index_sq']=X['traffic_index']**2
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=11)


#%% 
##############################################################################################################
# Linear regression 
###############################################################################################################

#estimation
X_train_OLS = sm.add_constant(X_train)
model = sm.OLS(endog=Y_train.astype(float), exog=X_train_OLS.astype(float)).fit()
model.summary()
#%%
#prediction
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
Y_pred_train = lin_reg.predict(X_train)
Y_pred_valid = lin_reg.predict(X_valid)



# %%
print("MSE for model on train data: %.2f"  % np.mean((Y_train - Y_pred_train) ** 2))
print("MSE for model on validation data: %.2f"  %np.mean((Y_valid - Y_pred_valid) ** 2))

#%%
### Models diagnostics ###
# plotting residuals vs fitted values

plt.scatter(Y_pred_train, (Y_train-Y_pred_train),c='b',alpha=0.5)
plt.ylabel("Residuals")
plt.xlabel("Predicted values")
#%%
plt.scatter(Y_pred_valid,(Y_valid-Y_pred_valid),c='r',alpha=0.5)
plt.ylabel("Residuals")
plt.xlabel("Predicted values")
# no systematic patterns visible
#%%
# checking E(e)=0, i.e. no incorretcly specified form of independence
plt.scatter(X_train['veh_value'],(Y_train-Y_pred_train),c='r',alpha=0.5)
plt.ylabel("Residuals")
plt.xlabel("Vehicle value")
#%%
plt.scatter(X_train['traffic_index'],(Y_train-Y_pred_train),c='r',alpha=0.5)
plt.ylabel("Residuals")
plt.xlabel("Traffic index")
#%%
plt.scatter(X_train['credit_score'],(Y_train-Y_pred_train),c='r',alpha=0.5)
plt.ylabel("Residuals")
plt.xlabel("Credit score")

#%%
### Breusch Pagan test for checking homoskedasticity ###
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(model.resid, model.model.exog)
lzip(names, test)
# the p-value suggests that residuals are heteroskedastic
#%%
# fitting the model again using HCE
model_HC = sm.OLS(endog=Y_train.astype(float), exog=X_train_OLS.astype(float)).fit(cov_type='HC1')
# %%
test2 = sms.het_breuschpagan(model_HC.resid, model_HC.model.exog)
lzip(names, test2)
# Heteroskedasticity persists - OLS estimators are still unbiased and consistent, although standard error and t-statistics are imprecise

#%%
### Durbin-Watson test for checking correlation of residuals ###
durbin_watson(model_HC.resid) 
# test statistics does not indicate serial correlation

# %%
### checking normality of residuals ###

influence = model.get_influence()
standardized_residuals = influence.resid_studentized_internal
sns.distplot(standardized_residuals, hist=True, hist_kws={'edgecolor':'black'}).set(xlabel='Standardized residuals',ylabel='Density')

# %%
#since we cannot use p-values of t-statistics for evaluation of significance we will use LASSO regression

# finding optimal alpha:
lasso_reg_cv = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X_train, Y_train)
lasso_reg_cv.alpha_

# %%
# fit the model
lasso_reg = Lasso(alpha=lasso_reg_cv.alpha_)
lasso_reg.fit(X_train, Y_train)
Y_pred_train_lasso = lasso_reg.predict(X_train)
Y_pred_valid_lasso = lasso_reg.predict(X_valid)

#%%
#evaluation
print("MSE for model on train data: %.2f"  % np.mean((Y_train - Y_pred_train_lasso) ** 2))
print("MSE for model on validation data: %.2f"  %np.mean((Y_valid - Y_pred_valid_lasso) ** 2))
# MSE for lasso regression has increased compared to classic LR model - will use LR instead
#%%
###############################################################################################################
# Analysis of potential customers
###############################################################################################################

df2.info()
df2.dtypes
# %%
df2.describe()
# %%
df2.nunique()
# %%
df2['gender_num'] = np.where(df2['gender']=='F', 1,0)
df2['gender_num']=df2['gender_num'].astype(object)


df2['date_of_birth_dt']=pd.to_datetime(df2['date_of_birth'])
df2['area']=df1['area'].astype(object)
df2['veh_body']=df2['veh_body'].astype(object)

# %%
###############################################################################################################
# MISSING VALUES
###############################################################################################################
#%%
Missing_values(df2, 'veh_age') 
Missing_values(df2, 'veh_body') 
Missing_values(df2, 'traffic_index') 
Missing_values(df2, 'area') 
Missing_values(df2, 'credit_score') 
Missing_values(df2, 'gender') 
Missing_values(df2, 'agecat') 
Missing_values(df2, 'veh_value') 
#missing values in agecat, credit score and traffic index

# %% 
# calculate missing agecat based on date of birth 
for j in range(df2.shape[0]):
    if pd.isna(df2['agecat'][j]):
        if df2['date_of_birth_dt'][j]<=pd.to_datetime('1949-12-31'):
            df2['agecat'][j]=6
        elif (df2['date_of_birth_dt'][j]>pd.to_datetime('1949-12-31')) & (df2['date_of_birth_dt'][j]<=pd.to_datetime('1959-12-31')):
            df2['agecat'][j]=5
        elif (df2['date_of_birth_dt'][j]>pd.to_datetime('1959-12-31')) & (df2['date_of_birth_dt'][j]<=pd.to_datetime('1969-12-31')):
            df2['agecat'][j]=4
        elif (df2['date_of_birth_dt'][j]>pd.to_datetime('1969-12-31')) & (df2['date_of_birth_dt'][j]<=pd.to_datetime('1979-12-31')):
            df2['agecat'][j]=3
        elif (df2['date_of_birth_dt'][j]>pd.to_datetime('1979-12-31')) & (df2['date_of_birth_dt'][j]<=pd.to_datetime('1989-12-31')):
            df2['agecat'][j]=2
        elif (df2['date_of_birth_dt'][j]>pd.to_datetime('1989-12-31')) & (df2['date_of_birth_dt'][j]<=pd.to_datetime('1999-12-31')):
            df1['agecat'][j]=1

df2['agecat']=df2['agecat'].astype(object)

# %%
# missing values in credit score and traffic index were replaced by median

df2['credit_score'].fillna(df2['credit_score'].median(), inplace=True)
df2['traffic_index'].fillna(df2['traffic_index'].median(), inplace=True)
# %%
###############################################################################################################
# Exploratory analysis
###############################################################################################################

#Create frequency tables for categorical variables
Frequencies(df2, 'gender')
#%%
Frequencies(df2, 'agecat')
#%%
Frequencies(df2, 'area')
#%%
Frequencies(df2, 'veh_age')
#%%
Frequencies(df2, 'veh_body')
#%%
#Credit score#
sns.distplot(df2['credit_score'], hist=True, hist_kws={'edgecolor':'black'}).set(xlabel='Credit score',ylabel='Density',
title='Credit score')
# %%
#Vehicle value#
sns.distplot(df2['veh_value'], hist=True, hist_kws={'edgecolor':'black'}).set(xlabel='Vehicle value',ylabel='Density',
title='Vehicle value')
#%%
# %%
#Traffic index
sns.distplot(df2['traffic_index'], hist=True, hist_kws={'edgecolor':'black'}).set(xlabel='Traffic index',ylabel='Density',
title='Traffic index')

# %%
###############################################################################################################
# Data preparation
###############################################################################################################
X_test1 = df2.drop(['veh_body', 'area', 'agecat', 'gender', 'quote_number', 'date_of_birth', 'date_of_birth_dt'  ],axis=1)
X_test=Get_dummies(X_test1, df2)
# %%
#Logistic regression for predicting probability of claim
Y_predict_clam_prob = log_model.predict_proba(X_test)[:,1]

# %%
# Linear regression for predicting log cost per claim
X_test['traffic_index_sq']=X_test['traffic_index']**2
X_test_OLS = sm.add_constant(X_test)
Y_predict_cost_per_claim=model_HC.predict(X_test_OLS)
# %%
#final_data=X_test_OLS
final_data=pd.DataFrame()
final_data['claim_factor']=np.where(Y_predict_clam_prob >= thresholds[optimal_idx1], 1,0)
final_data['claim_prob']=Y_predict_clam_prob
final_data['log_cost_per_claim'] = np.where(final_data['claim_factor']==1, Y_predict_cost_per_claim, int(0))

# %%
# using elbow method for determinig number of clusters 
wcss = {}
 
for k in range(2,20):
    KMmodel = KMeans(n_clusters=k, random_state=11)
    KMmodel.fit(final_data)
    wcss[k] = KMmodel.inertia_
     
plt.plot(wcss.keys(), wcss.values(), 'gs-')
plt.xlabel('Values of "k"')


# %%
#choose 5 based on plot
KMmodel_final = KMeans(n_clusters=5, random_state=11)
# %%
pred = KMmodel_final.fit_predict(final_data)


# %%
colours = ['red', 'blue', 'green', 'yellow', 'orange']

for i in np.unique(KMmodel_final.labels_):
    plt.scatter(final_data['claim_prob'][pred==i],
                final_data['log_cost_per_claim'][pred==i],
                c = colours[i])
     
plt.title('K Means clustering')
plt.xlabel('Predicted probability')
plt.ylabel('Log cost per claim')
# %%
final_data['log_cost_per_claim']=final_data['log_cost_per_claim'].astype('int')
#%%
### Analysis of the first group ###
final_data[pred==1].drop('claim_factor', axis=1).describe()

#%%
df2[pred==1].drop(['quote_number', 'date_of_birth', 'date_of_birth_dt'], axis=1).describe()
#%%
Frequencies(df2[pred==1], 'gender')
#%%
Frequencies(df2[pred==1], 'agecat')
#%%
Frequencies(df2[pred==1], 'veh_body')
#%%
### Analysis of the second group ###
#%%
final_data[pred==0].drop('claim_factor', axis=1).describe()
#%%
df2[pred==0].drop(['quote_number', 'date_of_birth', 'date_of_birth_dt'], axis=1).describe()
#%%
df2[pred==0].hist()
#%%
Frequencies(df2[pred==0], 'gender')
#%%
Frequencies(df2[pred==0], 'agecat')
#%%
Frequencies(df2[pred==0], 'veh_body')
# %%
### Analysis of the third group ###
final_data[pred==3].drop('claim_factor', axis=1).describe()
#%%
df2[pred==3].drop(['quote_number', 'date_of_birth', 'date_of_birth_dt'], axis=1).describe()
#%%
df2[pred==3].hist()
#%%
Frequencies(df2[pred==3], 'gender')
#%%
Frequencies(df2[pred==3], 'agecat')
#%%
Frequencies(df2[pred==3], 'veh_body')

# %%
### Analysis of the fourth group ###
final_data[pred==2].drop('claim_factor', axis=1).describe()
#%%
df2[pred==2].drop(['quote_number', 'date_of_birth', 'date_of_birth_dt'], axis=1).describe()
#%%
df2[pred==2].hist()
#%%
Frequencies(df2[pred==2], 'gender')
#%%
Frequencies(df2[pred==2], 'agecat')
#%%
Frequencies(df2[pred==2], 'veh_body')
# %%
### Analysis of the fifth group ###
final_data[pred==4].drop('claim_factor', axis=1).describe()
#%%
df2[pred==4].drop(['quote_number', 'date_of_birth', 'date_of_birth_dt'], axis=1).describe()

#%%
df2[pred==4].hist()
#%%
Frequencies(df2[pred==4], 'gender')
#%%
Frequencies(df2[pred==4], 'agecat')
#%%
Frequencies(df2[pred==4], 'veh_body')
# %%
#END