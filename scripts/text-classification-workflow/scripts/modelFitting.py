import os
import numpy as np
import xgboost as xgb

import pandas as pd
from sklearn.model_selection import StratifiedKFold
def configEnv():# {{{
    env = {}
    env["base_dir"] = os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) )
    env["data_dir"] = os.path.join(env["base_dir"], "data")
    env["preproc_dir"] = os.path.join(env["data_dir"],"preprocessed_data")
    env["ml_dir"] = os.path.join(env["preproc_dir"],"ml")
    env["models_dir"] = os.path.join(env["data_dir"],"models")
    return env# }}}

def load_data(env):# {{{
    dataset_filename = "dataset.csv"
    dataset_abspath = os.path.join(env["ml_dir"],dataset_filename)
    return pd.read_csv(dataset_abspath,header=None)# }}}

def gini(y, pred):# {{{
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)# }}}

def gini_xgb(pred, y):# {{{
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)# }}}

def train_xgb(params,d_train,nrounds,watchlist,feval):# {{{
    xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100,
                                   feval=gini_xgb, maximize=True, verbose_eval=100)
    return xgb_model# }}}


if __name__ == "__main__":
    env = configEnv()
    D = load_data(env)
    y_colName = D.columns[-1]
    y = D.ix[:,y_colName]
    X = D.ix[:,D.columns != y_colName ]



    nrounds = 10
    xgb_params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}
    
    folders_perfomance = []
    kfold = 2  # need to change to 5
    skf = StratifiedKFold(n_splits=kfold, random_state=0)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        X_train, X_valid = X.ix[train_index], X.ix[test_index]
        y_train, y_valid = y.ix[train_index], y.ix[test_index]
        d_train = xgb.DMatrix(X_train, y_train)
        d_valid = xgb.DMatrix(X_valid, y_valid)
        d_train = xgb.DMatrix(X,y)
        watchlist = [(d_train, 'train'), (d_train,'valid')]
        
        trained_xgb = train_xgb(xgb_params,d_train,nrounds,watchlist,gini_xgb)
        
        pred_probas = trained_xgb.predict(xgb.DMatrix(X.ix[test_index]),ntree_limit=trained_xgb.best_ntree_limit+50)

