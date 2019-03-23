import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv('E:/Data/LennoxInternational/DataExtract/Data_Train.csv')
df_test = pd.read_csv('E:/Data/LennoxInternational/DataExtract/Data_Validation.csv')

#Columns to drop
drop_columns = [
	'Sales',		#Sales (y variable)
	'Customer No.',
	'Plant',
	'Sold_To_Party'
]

#Columns that are categories
cat_columns = [
	'Cat_Category',
	'FISCAL_MONTH',
	'Cat_Marketing_Package',
	'Cat_Cluster'
]

y_train = df_train['Sales']
y_test = df_test['Sales']
X_train = df_train.drop(drop_columns, 1)
X_test = df_test.drop(drop_columns, 1)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
}

print("Starting training . . .")

categoricals = []
for col in cat_columns:
	try:
		categoricals.append(df_train.columns.tolist().index(col))
	except ValueError:
		continue
print(categoricals)
print(df_train.columns.tolist())

# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_valid,
                early_stopping_rounds=15,
				categorical_feature=categoricals)
# feature names
print('Feature names:', gbm.feature_name())

# feature importances
print('Feature importances:', list(gbm.feature_importance()))
# predict
#y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
#print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)