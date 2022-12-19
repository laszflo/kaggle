# %%
# load data
import numpy as np
import pandas as pd

data_train = pd.read_csv('train.csv')

target_train = data_train.Survived

data_train.info()
# allows to see dtypes and missing values


data_train.describe()
# allows to see numerical columns statistics


# %% 

data_train.drop(['Survived','PassengerId'], axis=1, inplace=True)

data_train.head()

# drop name 

data_train.drop(['Name'], axis=1, inplace=True)

# let's split the data into numerical and categorical columns

num_col = [col for col in data_train.columns
            if data_train[col].dtype in ['int64', 'float64']]

cat_col = [col for col in data_train.columns
            if data_train[col].dtype == 'object']
# simple imputer for numerical columns

from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')

# simple imputer for categorical columns

cat_imputer = SimpleImputer(strategy='most_frequent')

# make column transformer for num and cat cols with respectively num imputer / standard scaler  and cat imputer / one hot encoder

from sklearn.compose import make_column_transformer

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

# %%
# do the same for the test data

data_test = pd.read_csv('test.csv')

data_test.drop(['PassengerId'], axis=1, inplace=True)

# drop name

data_test.drop(['Name'], axis=1, inplace=True)

# preprocess data

x_test_nomiss = preprocessor1.transform(data_test)

df_test_nomiss = pd.DataFrame(x_test_nomiss,columns=num_col+cat_col)

x_test = preprocessor2.transform(df_test_nomiss)



# %%


# train random forest classifier
import time 

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=42)

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# # grid search on max_depth and n_estimators for random forest

# from sklearn.model_selection import GridSearchCV

# param_grid = {'max_depth': np.linspace(1, 50, 30, dtype=int),
#                 'n_estimators': np.linspace(10, 600, 6, dtype=int),
#                 'min_samples_split': np.linspace(1, 30, 10, dtype=int),
#                 'max_features': [1, 'sqrt', 'log2']}

# start = time.time()

# search = GridSearchCV(rf_clf, param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=-1, verbose=1)

# search.fit(x_train, target_train)

# print(f"best balanced_accuracy {search.best_score_:.3f} with params {search.best_params_}")
# end = time.time()

# print(f"training time {end-start:.3f} seconds")



start = time.time()
# score model on train data with cross validation with stratified k fold



from sklearn.model_selection import cross_val_score



scores = cross_val_score(rf_clf, x_train, target_train, cv=cv, scoring='balanced_accuracy')

print(f"balanced_accuracy with random-forest clf {scores.mean():.3f} +/- {scores.std():.3f}")

end = time.time()

print(f"training time {end-start:.3f} seconds")

# %%
# train logistic regression

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression()






start = time.time()
# score model on train data with cross validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(lr_clf, x_train, target_train, cv=cv, scoring='balanced_accuracy')

end = time.time()

print(f"balanced_accuracy with logistic regression {scores.mean():.3f} +/- {scores.std():.3f}")

print(f"training time {end-start:.3f} seconds")

# %%
# train svm

from sklearn.svm import SVC

svc_clf = SVC()


# score model on train data with cross validation

from sklearn.model_selection import cross_val_score
start = time.time()
scores = cross_val_score(svc_clf, x_train, target_train, cv=cv, scoring='balanced_accuracy')
end = time.time()
print(f"balanced_accuracy with svm {scores.mean():.3f} +/- {scores.std():.3f}")

print(f"training time {end-start:.3f} seconds")

# %% gradient boosting

from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier()


# score model on train data with cross validation

from sklearn.model_selection import cross_val_score
start = time.time()

scores = cross_val_score(gb_clf, x_train, target_train, cv=cv, scoring='balanced_accuracy')

end = time.time()

print(f"balanced_accuracy with gradient boosting {scores.mean():.3f} +/- {scores.std():.3f}")

print(f"training time {end-start:.3f} seconds")



# %% voting classifier

from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[('lr', lr_clf), ('rf', rf_clf), ('svc', svc_clf), ('gb', gb_clf)], voting='hard')

start = time.time()

scores = cross_val_score(voting_clf, x_train, target_train, cv=cv, scoring='balanced_accuracy')

print(f"balanced_accuracy with voting classifier {scores.mean():.3f} +/- {scores.std():.3f}")

end = time.time()

print(f"training time {end-start:.3f} seconds")


# %% predict with random forest

rf_clf.fit(x_train, target_train)

predictions = rf_clf.predict(x_test)

# %% 

submissions = pd.concat([pd.read_csv('test.csv')['PassengerId'],pd.DataFrame(predictions)],axis=1)

submissions.columns = ['PassengerId','Survived']

submissions.to_csv('submission.csv',index=False)









# predictions = model.predict(data_test)
# indexes =pd.read_csv('test.csv')['PassengerId']
# submission = pd.concat([indexes,pd.DataFrame(predictions)],axis=1)

# submission.columns = ['PassengerId','Survived']

# submission.to_csv('submission.csv',index=False)

# # %%

# # let's try with autokeras


# import numpy as np
# import pandas as pd
# import tensorflow as tf

# import autokeras as ak

# # %%

# x_train = pd.read_csv('train.csv')
# y_train = x_train['Survived']
# x_train = x_train.drop(['Survived'], axis=1)
# x_test = pd.read_csv('test.csv')

# # %%


# # It tries 3 different models.
# clf = ak.StructuredDataClassifier()
# # Feed the structured data classifier with training data.
# clf.fit(x_train, y_train)
# # Predict with the best model.
# predicted_y = clf.predict(x_test)

# # %%

# submissions = pd.concat([x_test['PassengerId'], pd.DataFrame(predicted_y)], axis=1)

# # to csv 
# submissions.to_csv('submission.csv', index=False, header=['PassengerId', 'Survived'])# %%

# # %%

# %%
