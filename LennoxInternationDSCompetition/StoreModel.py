#This file focuses more on exploration of store data. I'm trying to predict foottraffic and see what is important
#Then maybe add foottraffic somehow to the actual billing data and see if that becomes important
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from graphviz import Digraph

df = pd.read_csv('E:/Data/LennoxInternational/DataExtract/Data_Stores.csv')
#Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)
split = int(len(df.index) * 0.8)	#Keep 80% of store data as training and rest as validation
df_train = df.iloc[:split]
df_test = df.iloc[split:]

#Columns to drop
drop_columns = [
	'Foottraffic',
	'Plant'	#Plant is a string so cannot train - maybe encode it to integer?
]

#Columns that are categories
cat_columns = [
	'Year',
	'Month',
	'Opening_Year',
	'Cat_Store_Type',
	'Cat_Store_Size',
	'Cat_Trade_Area_Size',
	'Does Store Have a Fleet Delivery Truck?',
]

y_train = df_train['Foottraffic']
y_test = df_test['Foottraffic']
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

# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=lgb_valid,
                early_stopping_rounds=15,
				categorical_feature=categoricals)
# feature names
print('Feature names:', gbm.feature_name())

# feature importances
print('Feature importances:', list(gbm.feature_importance()))

graph = lgb.create_tree_digraph(gbm)
graph.view(cleanup=True)