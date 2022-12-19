# %%

# load data 
# import pandas as pd
# data = pd.read_csv('train.csv')
# data_train = pd.read_csv('train.csv')
# # data.head()
# # %% 

# # make a list of columns with : lotarea, overall cond, year built, bedroom

# col = ['LotArea', 'OverallCond', 'YearBuilt', 'BedroomAbvGr','FullBath','Utilities','LotShape','Street','YearRemodAdd','1stFlrSF','2ndFlrSF','GrLivArea','KitchenAbvGr','YrSold','MoSold','Neighborhood','Condition1','Heating','Electrical','SaleType','SaleCondition']



# data_train = data_train[col+['SalePrice']]

# # drop row with missing values

# data_train = data_train.dropna(axis=0)

# # drop saleprice column as it is the target variable


# target_train = data_train['SalePrice']
# # data_train.head()

# data_train = data_train.drop(['SalePrice'], axis=1)



# data_test = pd.read_csv('test.csv')
# df_test = data_test[col]
# # %%

# # write a pipepline with standard sscaler on numerical data and one hot encoder on categorical data
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import make_column_transformer


# # get numerical and categorical columns
# numerical_cols = [cname for cname in data_train.columns if data_train[cname].dtype in ['int64', 'float64']]
# categorical_cols = [cname for cname in data_train.columns if data_train[cname].dtype == "object"]

# # create a pipeline for numerical data

# numerical_transformer = StandardScaler()

# # create a pipeline for categorical data
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# # create a column transformer
# preprocessor = make_column_transformer(
#     (numerical_transformer, numerical_cols),
#     (categorical_transformer, categorical_cols))

# # combine with random forest regressor

# from sklearn.ensemble import RandomForestRegressor

# reg = make_pipeline(preprocessor, RandomForestRegressor())


# # %% 

# # grid search over hyperparameters 

# from sklearn.model_selection import GridSearchCV
# import numpy as np

# # param_grid = {'randomforestregressor__n_estimators': [10,50,100,200],
# #               'randomforestregressor__max_depth': [1,5,10,20],
# #               'randomforestregressor__min_samples_leaf': [1,10,20,50],
# #               }


# n_estimators = [100,200,500] # number of trees in the random forest
# max_features = [1,2,5,7,10,15,20] # number of features in consideration at every split
# max_depth = [1,2,5,10,20,30] # maximum number of levels allowed in each decision tree


# param_grid = {'randomforestregressor__n_estimators': n_estimators,

# 'randomforestregressor__max_features': max_features,

# 'randomforestregressor__max_depth': max_depth}

# grid_search = GridSearchCV(reg, param_grid, cv=5, n_jobs=-1, verbose=10,scoring='neg_mean_absolute_error')

# grid_search.fit(data_train, target_train)

# print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)

# # %% 

# # choose best model

# model = grid_search.best_estimator_

# # %%    
# # eval model 

# from sklearn.model_selection import cross_val_score

# scores = -1 * cross_val_score(model, data_train, target_train,cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# print("r2 scores:", scores.mean())




# # %%

# # train model

# _ = model.fit(data_train, target_train)

# # %%
# # predict on test data




# predictions = model.predict(df_test)

# submissions = pd.concat([data_test['Id'], pd.DataFrame(predictions)], axis=1)

# # to csv 
# submissions.to_csv('submission.csv', index=False, header=['Id', 'SalePrice'])



# %%
# let's try with autokeras 

import numpy as np
import pandas as pd
import tensorflow as tf

import autokeras as ak

# %%

x_train = pd.read_csv('train.csv')
y_train = x_train['SalePrice']
x_train = x_train.drop(['SalePrice'], axis=1)
x_test = pd.read_csv('test.csv')
# %% 
# It tries 10 different models.
reg = ak.StructuredDataRegressor(max_trials=10, overwrite=True)
# Feed the structured data regressor with training data.
reg.fit(x_train, y_train)
# Predict with the best model.
predicted_y = reg.predict(x_test)



# %%

submissions = pd.concat([x_test['Id'], pd.DataFrame(predicted_y)], axis=1)

# to csv 
submissions.to_csv('submission.csv', index=False, header=['Id', 'SalePrice'])

# %%
