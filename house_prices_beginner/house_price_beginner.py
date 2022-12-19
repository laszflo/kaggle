# %%

import pandas as pd

# load data

data_train = pd.read_csv('train.csv')

target_train = data_train.SalePrice

data_train.drop(['SalePrice','Id'], axis=1, inplace=True)
data_train.head()

# %%

# use a simple imputer to fill missing values
import numpy as np
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')

cat_imputer = SimpleImputer(strategy='most_frequent')

# make column transformer for num and cat cols with respectively num imputer / standard scaler  and cat imputer / one hot encoder 

from sklearn.compose import make_column_transformer

num_col = [col for col in data_train.columns
                if data_train[col].dtype in ['int64', 'float64']]

cat_col = [col for col in data_train.columns        
                if data_train[col].dtype == 'object']

from sklearn.compose import make_column_selector




preprocessor1 = make_column_transformer(
    (num_imputer, num_col),
    (cat_imputer, cat_col)
)

# preprocess data

x_train_nomiss = preprocessor1.fit_transform(data_train)

df_train_nomiss = pd.DataFrame(x_train_nomiss,columns=num_col+cat_col)


# %% 

# use standard scaler for num col and one hot encoder for cat col with make column transformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

scaler = StandardScaler()

encoder = OneHotEncoder(handle_unknown='ignore')

preprocessor2 = make_column_transformer(
    (scaler, num_col),
    (encoder, cat_col)
)

# preprocess data

x_train = preprocessor2.fit_transform(df_train_nomiss)



# %% do the same with test data

data_test = pd.read_csv('test.csv')

data_test.drop(['Id'], axis=1, inplace=True)

x_test_nomiss = preprocessor1.transform(data_test)

# to dataframe

df_test_nomiss = pd.DataFrame(x_test_nomiss, columns=num_col+cat_col)

x_test = preprocessor2.transform(df_test_nomiss)


# %%
# model of regression with random forest regressor

import time 

start_time = time.time()

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()


# cross validation 

from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_reg, x_train, target_train, cv=10, scoring='neg_root_mean_squared_error', n_jobs=-1)
elapsed_time = time.time() - start_time
print('root mean squared error rf: ', f"{-1*np.mean(scores)}+/-{np.std(scores)}", '\nelapsed time: ', elapsed_time)

# %% 
# model of regression with gradient boosting regressor

from sklearn.ensemble import GradientBoostingRegressor

gb_reg = GradientBoostingRegressor()
# without early stopping

start_time = time.time()

# cross validation

scores = cross_val_score(gb_reg, x_train, target_train, cv=10, scoring='neg_root_mean_squared_error', n_jobs=-1)
elapsed_time = time.time() - start_time
print('root mean squared error gbr: ', f"{-1*np.mean(scores)}+/-{np.std(scores)}", '\nelapsed time: ', elapsed_time)

# randomize search for best parameters

from sklearn.model_selection import RandomizedSearchCV

param_grid = {'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9],
                'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8],
                'max_features': [1, 'sqrt', 'log2'],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]}
# run randomized search

n_iter_search = 5000

random_search = RandomizedSearchCV(gb_reg, param_distributions=param_grid,
                                      n_iter=n_iter_search, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1,verbose=1)

start_time = time.time()

random_search.fit(x_train, target_train)

elapsed_time = time.time() - start_time

print('root mean squared error gbr with randsearch : ', f"{-1*random_search.best_score_}", '\nelapsed time: ', elapsed_time)



# %%
# model of regression with hist gradient boosting regressor

from sklearn.ensemble import HistGradientBoostingRegressor

hgbr_reg = HistGradientBoostingRegressor()

start_time = time.time()

# cross validation

scores = cross_val_score(hgbr_reg, x_train.toarray(), target_train.values, cv=10, scoring='neg_root_mean_squared_error', n_jobs=-1)

elapsed_time = time.time() - start_time

print('root mean squared error hgbr: ', f"{-1*np.mean(scores)}+/-{np.std(scores)}", '\nelapsed time: ', elapsed_time)


# %%
# fit model with best result 
model = random_search.best_estimator_
_ = model.fit(x_train, target_train)
# predict test data
predictions = model.predict(x_test)




# %% save result

submission = pd.DataFrame({'Id': pd.read_csv('test.csv').Id,
                          'SalePrice': predictions})



submission.to_csv('submission.csv', index=False)


# %%

# cross validation with best estimator 

scores = cross_val_score(model, x_train, target_train, cv=10, scoring='neg_root_mean_squared_error', n_jobs=-1)

print('root mean squared error gbr with randsearch : ', f"{-1*np.mean(scores)}+/-{np.std(scores)}")
# %%
