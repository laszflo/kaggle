# %%

import pandas as pd

# load data

data_train = pd.read_csv('train.csv')


target_train = data_train.Transported

data_train.drop(['Transported','PassengerId','Name'], axis=1, inplace=True)

data_train.head()

# %%  cols with missing values and count of missing values

cols_with_missing = [col for col in data_train.columns
                        if data_train[col].isnull().any()]

# for cols with missing values that are numbers fill with median 

for col in cols_with_missing:
    if data_train[col].dtype in ['int64', 'float64']:
        data_train[col].fillna(data_train[col].median(), inplace=True)

cols_with_missing = [col for col in data_train.columns
                        if data_train[col].isnull().any()]




# replace missing values with most frequent value
for col in ['HomePlanet','CryoSleep','VIP']:
    data_train[col].fillna(data_train[col].value_counts().index[0], inplace=True)

cols_with_missing = [col for col in data_train.columns
                        if data_train[col].isnull().any()]


# for the cabin column elements that look lite 'B/O/P' split into three col 'B' '0' and 'P' and fill with most frequent value

data_train['CabinList'] = data_train['Cabin'].apply(lambda x: str(x).split('/'))

data_train['Cabin1'] = data_train['CabinList'].apply(lambda x: x[0] if len(x)>1  else '.' )
# replace "." with most frequent value
data_train['Cabin1'].replace('.', data_train['Cabin1'].value_counts().index[0], inplace=True)

data_train['Cabin3'] = data_train['CabinList'].apply(lambda x: x[2] if len(x) > 1 else '.')
# replace "." with most frequent value


# drop cabin column and cabin list column

data_train.drop(['Cabin','CabinList'], axis=1, inplace=True)





# for the destination col replace missing values with most frequent value

data_train['Destination'].fillna(data_train['Destination'].value_counts().index[0], inplace=True)



cols_with_missing = [col for col in data_train.columns
                        if data_train[col].isnull().any()]



for col in cols_with_missing:   
    print(col, data_train[col].isnull().sum())

# %%

# load test data

data_test = pd.read_csv('test.csv')

data_test.drop(['PassengerId','Name'], axis=1, inplace=True)

# data test col list with missing values and count of missing values

cols_with_missing = [col for col in data_test.columns

                        if data_test[col].isnull().any()]   

# for cols with missing values that are numbers fill with median

for col in cols_with_missing:
    if data_test[col].dtype in ['int64', 'float64']:
        data_test[col].fillna(data_test[col].median(), inplace=True)

cols_with_missing = [col for col in data_test.columns
                        if data_test[col].isnull().any()]

# count of missing values in data test



# for the columns Homeplanet cryosleep and VIP and destination replace missing values with most frequent value


for col in ['HomePlanet','CryoSleep','VIP','Destination']:
    data_test[col].fillna(data_test[col].value_counts().index[0], inplace=True)

cols_with_missing = [col for col in data_test.columns
                        if data_test[col].isnull().any()]






# for the cabin column elements that look lite 'B/O/P' split into three col 'B' '0' and 'P' and fill with most frequent value

data_test['CabinList'] = data_test['Cabin'].apply(lambda x: str(x).split('/'))
data_test['Cabin1'] = data_test['CabinList'].apply(lambda x: x[0] if len(x)>1  else '.' )
# replace "." with most frequent value
data_test['Cabin1'].replace('.', data_test['Cabin1'].value_counts().index[0], inplace=True)

data_test['Cabin3'] = data_test['CabinList'].apply(lambda x: x[2] if len(x) > 1 else '.')
# replace "." with most frequent value
data_test['Cabin3'].replace('.', data_test['Cabin3'].value_counts().index[0], inplace=True)




# # drop cabin column and cabin list column

data_test.drop(['Cabin','CabinList'], axis=1, inplace=True)


cols_with_missing = [col for col in data_test.columns
                        if data_test[col].isnull().any()]



# %% preprocess data


# categorical columns 

categorical_cols = [cname for cname in data_train.columns if
                    data_train[cname].dtype in ["object","bool"]]

# numerical columns

numerical_cols = [cname for cname in data_train.columns if
                data_train[cname].dtype in ['int64', 'float64']]


# make column transformer

from sklearn.compose import make_column_transformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    (StandardScaler(), numerical_cols)
)

# %% make pipeline

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

model = make_pipeline(preprocessor, RandomForestClassifier())

# %%

# grid search for best parameters : max_depth, n_estimators     

from sklearn.model_selection import GridSearchCV

param_grid = {
    'randomforestclassifier__max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'randomforestclassifier__n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}

grid_search = GridSearchCV(model, param_grid, cv=10, n_jobs=-1, verbose=5)


# %% fit grid search

grid_search.fit(data_train, target_train)

# %% best parameters

print(grid_search.best_params_)

# %% best score

print(grid_search.best_score_)

# %% best estimator

print(grid_search.best_estimator_)

best_model = grid_search.best_estimator_


# %% predict on test data

predictions = best_model.predict(data_test)

# %% save predictions to file
data= pd.read_csv('test.csv')
output = pd.DataFrame({'PassengerId': data.PassengerId, 'Transported': predictions})
# %%
# to csv    
output.to_csv('submission.csv', index=False)
# %%
