import torch
import random
import numpy as np
import scipy.sparse
from sklearn import metrics
from munkres import Munkres

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def drop_data(adata, rate, datatype='real'):
    
    X = adata.X

    if scipy.sparse.issparse(X):
        X = np.array(X.todense())

    if datatype == 'real':
        X_train = np.copy(X)
        i, j = np.nonzero(X)

        ix = np.random.choice(range(len(i)), int(
            np.floor(rate * len(i))), replace=False)
        X_train[i[ix], j[ix]] = 0.0

        drop_index = {'i':i, 'j':j, 'ix':ix}
        adata.uns['drop_index'] = drop_index        
        adata.obsm["train"] = X_train
        adata.obsm["test"] = X

        # for training
        adata.raw.X[i[ix],j[ix]] = 0.0

    elif datatype == 'simul':
        adata.obsm["train"] = X

    return adata


def cluster_acc(y_true, y_pred):

        #######
        y_true = y_true.astype(int)
        #######

        y_true = y_true - np.min(y_true)
        l1 = list(set(y_true))
        numclass1 = len(l1)
        l2 = list(set(y_pred))
        numclass2 = len(l2)

        ind = 0
        if numclass1 != numclass2:
            for i in l1:
                if i in l2:
                    pass
                else:
                    y_pred[ind] = i
                    ind += 1

        l2 = list(set(y_pred))
        numclass2 = len(l2)

        if numclass1 != numclass2:
            print('n_cluster is not valid')
            return

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
                cost[i][j] = len(mps_d)

        m = Munkres()
        cost = cost.__neg__().tolist()
        indexes = m.compute(cost)

        new_predict = np.zeros(len(y_pred))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]
            ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(y_true, new_predict)
        # y_true：Like 1d array or label indicator array/sparse matrix (correct) label
        # y_pred：Like a one-dimensional array or label indicator array/sparse matrix predicted labels, returned by the classifier
        
        f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
        f1_micro = metrics.f1_score(y_true, new_predict, average='micro')

        return acc, f1_macro, f1_micro


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals

def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        st_ = "{} <- {} / ".format(name, val)
        st += st_

    return st[:-1]

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['device']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]
