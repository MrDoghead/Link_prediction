import pickle as pkl
import pandas as pd
import csv
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, mean_squared_error

def read_from_pkl(path):
    with open(path,'rb') as f:
        data = pkl.load(f)
    return data

def get_train_df(data):
    train_df = pd.DataFrame(data).T
    train_df.rename(columns={
                        0:'label',
                        1:'source',
                        2:'sink',
                        3:'num_of_neighbors_source',
                        4:'num_of_in_neighbors_source',
                        5:'num_of_out_neighbors_source',
                        6:'num_of_neighbors_sink',
                        7:'num_of_in_neighbors_sink',
                        8:'num_of_out_neighbors_sink',
                        9:'num_of_neighbors_sum',
                        10:'num_of_in_neighbors_sum',
                        11:'num_of_out_neighbors_sum',
                        12:'adamic_adar_index',
                        13:'cosine_sim',
                        14:'jaccard_coefficient',
                        15:'salton_sim',
                        16:'preferential_attachment',
                        17:'friends_mearsure',
                        18:'resource_allocation',
                        #19:'adamic_adar_index_out',
                        #20:'cosine_sim_out',
                        #21:'jaccard_coefficient_out',
                        #22:'salton_sim_out',
                        #23:'preferential_attachment_out',
                        #24:'resource_allocation_out'
                        },inplace=True)
    train_df = train_df.astype(float)
    train_df[['label','source','sink']] = train_df[['label','source','sink']].astype(int)
    return train_df

def get_test_df(data):
    test_df = pd.DataFrame(data).T
    test_df.rename(columns={
                        0:'label',
                        1:'source',
                        2:'sink',
                        3:'num_of_neighbors_source',
                        4:'num_of_in_neighbors_source',
                        5:'num_of_out_neighbors_source',
                        6:'num_of_neighbors_sink',
                        7:'num_of_in_neighbors_sink',
                        8:'num_of_out_neighbors_sink',
                        9:'num_of_neighbors_sum',
                        10:'num_of_in_neighbors_sum',
                        11:'num_of_out_neighbors_sum',
                        12:'adamic_adar_index',
                        13:'cosine_sim',
                        14:'jaccard_coefficient',
                        15:'salton_sim',
                        16:'preferential_attachment',
                        17:'friends_mearsure',
                        18:'resource_allocation',
                        #19:'adamic_adar_index_out',
                        #20:'cosine_sim_out',
                        #21:'jaccard_coefficient_out',
                        #22:'salton_sim_out',
                        #23:'preferential_attachment_out',
                        #24:'resource_allocation_out'
                        },inplace=True)
    test_df = test_df.drop('label',axis=1)
    test_df = test_df.astype(float)
    test_df[['source','sink']] = test_df[['source','sink']].astype(int)
    return test_df

def standardise(data):
    """remove the mean and transform to unit variance"""
    scaler = StandardScaler()
    scaler.fit(data)
    result = scaler.transform(data)
    return pd.DataFrame(result)

def save_as_pkl(data,path):
    with open(path,'wb') as f:
        pkl.dump(data,f,protocol=pkl.HIGHEST_PROTOCOL)
    print(path,'saved..')

def save_as_csv(result,path):
    headers = ['id','Predicted']
    with open(path, 'w', encoding = 'utf8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(result)
    print(path,'saved...')

# load data
train_data = read_from_pkl('./data/train_19_features.pkl')
train_df = get_train_df(train_data)
#print(train_df.info())
#print(train_df.describe())

# choose features
train = train_df.iloc[:,3:]
label = train_df.iloc[:,0]
train_x,val_x,train_y,val_y = train_test_split(train,label,test_size=0.2)

# standardize
train_x_std = (train_x - train_x.mean()) / train_x.std()
print('train data:')
print('train x:',train_x_std.head(3))
print('train y:',train_y.head(3))
val_x_std = (val_x - train_x.mean()) / train_x.std()
print('val data:')
print('val x:',val_x_std.head(3))
print('val y:',val_y.head(3))

# grid search cv
pipeline = Pipeline([
    ('clf', RandomForestRegressor(criterion='mse'))
])
parameters = {
       'clf__n_estimators': (50,100, 300),
       'clf__max_depth': (30, 50, 80),
       'clf__min_samples_split': (50, 100, 1000),
       'clf__min_samples_leaf': (50, 100, 1000)
}
'''
parameters = {
       'clf__n_estimators': [300],
       'clf__max_depth': [80],
       'clf__min_samples_split': [50],
       'clf__min_samples_leaf': [50]
        }
'''
grid_search = GridSearchCV(pipeline,parameters,n_jobs=-1,verbose=1,cv=4,scoring='roc_auc')
grid_search.fit(train_x_std,train_y)
print('Best score: %0.3f' % grid_search.best_score_)
print('Best parameters set:')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
#save_as_pkl(best_parameters,'./rf_best_parameters.pkl')

val_pred = grid_search.predict(val_x_std)
fpr, tpr, thresholds = roc_curve(val_y, val_pred, pos_label=1)
print('mean_squared_error:', mean_squared_error(val_y, val_pred))
print('Auc:',auc(fpr, tpr))

# prediction
print('\nstart prediction...')
test_data = read_from_pkl('./data/test_19_features.pkl')
test_df = get_test_df(test_data)
test_x = test_df.iloc[:,2:]
test_x_std = (test_x - train_x.mean()) / train_x.std()
print('test x:',test_x_std.head(3))
test_pred = grid_search.predict(test_x_std)
results = []
cnt = 1
for y in test_pred:
    results.append((cnt,y))
    cnt += 1

# save results
save_as_csv(results,'./results/rf_19_features_prediction.csv')





