import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import modAL
#import sklearn
#from pyDOE import *
#from functools import partial
#from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner, Committee
from customprim import make_box, in_box, peel_one, paste_one
from smt.sampling_methods import LHS
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import metrics

RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)

mpl.use('agg')
lowess = sm.nonparametric.lowess
kernel = 1.0 * RBF(1.0)

y_fun_val = None

def get_limits(data):
    limits = np.zeros((data.shape[1],2))
    for i in range(data.shape[1]):
        values = data[:,i]
        limits[i,0] = np.min(values)
        limits[i,0] = np.amax(values)
    return limits


def get_metrics(pred, orig):
    prec = np.sum(pred) / pred.size
    rec = np.sum(pred) / np.sum(orig)
    return [prec, rec]


def consistencyd(box1, box2, ds):
    pred1 =  np.zeros(ds.shape[0])
    pred2 =  np.zeros(ds.shape[0])
    pred1_idx = in_box(ds, box1, box1.shape[1], boolean=True)
    pred2_idx = in_box(ds, box2, box2.shape[1], boolean=True)
    pred1[pred1_idx] = 1
    pred2[pred2_idx] = 1
    res = np.sum(np.logical_and(pred1==1, pred2==1)) / np.sum(np.logical_or(pred1==1, pred2==1))
    return res


def consistencyv(box1, box2, box_init):
    box1[0,:] = np.maximum(box1[0,:], box_init[0,:])
    box2[0,:] = np.maximum(box2[0,:], box_init[0,:])
    box1[1,:] = np.minimum(box1[1,:], box_init[1,:])
    box2[1,:] = np.minimum(box2[1,:], box_init[1,:])

    inter = box_init
    inter[0,:] = np.maximum(box1[0,:], box2[0,:])
    inter[1,:] = np.minimum(box1[1,:], box2[1,:])

    sides1 = np.subtract(box1[1,:], box1[0,:])
    sides2 = np.subtract(box2[1,:], box2[0,:])
    sides_inter = np.subtract(inter[1,:], inter[0,:])

    if(np.sum(sides_inter<=0) > 0):
        return 0
    else:
        return 1/(np.prod(np.divide(sides1,sides_inter)) + np.prod(np.divide(sides2,sides_inter)) - 1)


def interpretability(res_box, box_init):
    #print(box_init)
    #print(res_box)
    res_box[0, :] = np.maximum(res_box[0,:], box_init[0,:])
    res_box[1, :] = np.minimum(res_box[1,:], box_init[1,:])
    #print(res_box)
    bx = box_init != res_box
    #print(bx)
    res = np.sum(np.amax(bx, axis=0))
    #print(res)
    return res


def auc(pr):
    y = pr[:,0]
    x = pr[:,1]
    return np.trapz(x, y)


def prim_single(x, y, box, x_init, y_init, pasting, peel_alpha=0.05, threshold=1.0, paste_alpha=0.01):
    d = box.shape[1]
    y_fun_val = np.mean(y)
    box_p = box

    if (y_fun_val >= threshold):
        res = [box, x, y]
    else:
        res = peel_one(x,y,box,peel_alpha,0,0,d,1)
        if res is  None:
            res = [box,x,y]
        else:
            res = [res[3],res[0],res[1]]

        box_p = res[0]
        if pasting == True:
            res_paste = res
            while res_paste is not None:
                res_p = res_paste
                res_paste = paste_one(res_p[1][0],res_p[2],res_p[0],x_init,y_init,paste_alpha,0,0,d,1)
            box_p = res_p[0]
    y_fun_val = np.mean(res[2])
    return [(y_fun_val >= threshold), res, box_p]


def prim_norm(x_train, y_train, x_test, y_test, x_eval, y_eval,box, minpts=20.0, max_peels=99, pasting=False):
    x_init = x_train
    y_init = y_train
    boxes = []
    boxes.append(box)

    pr_eval = [get_metrics(y_eval, y_eval)]
    pr_test = [get_metrics(y_test, y_test)]

    res = prim_single(x_train, y_train,box, pasting=pasting, x_init=x_init, y_init=y_init)
    ind_in_box = in_box(x_eval, res[2], d=res[2].shape[1], boolean=True)
    i = 0

    while (res[1][2].shape[0] >= minpts and ind_in_box.shape[0] >= minpts and res[0] == False and i < max_peels):
        i = i + 1

        pred_e = y_eval[ind_in_box]
        pr_eval.append(get_metrics(pred_e, y_eval))
        ind_in_box = in_box(x_test,res[2],d=res[2].shape[1],boolean=True)
        pred_t = y_test[ind_in_box]
        pr_test.append(get_metrics(pred_t, y_test))
        boxes.append(res[2])

        num_before = res[1][2].shape[0]
        res = prim_single(res[1][1][0], res[1][2], box=res[1][0], pasting=pasting, x_init=x_init, y_init=y_init)
        ind_in_box = in_box(x_eval, res[2], d=res[2].shape[1], boolean=True)

        if (num_before == res[1][2].shape[0] and res[0] == False):
            print("can't peel any direction")
            res[0] = True

    np_pr_eval = np.array(pr_eval)
    np_pr_test = np.array(pr_test)

    return [np_pr_test,np_pr_eval,boxes]


def prim_rf(x_train, y_train, x_test, y_test, box, npts=100000, minpts=20.0, grow_deep=False, pasting=False):
    dim = x_train.shape[1]

    limits = get_limits(x_train)
    sampling = LHS(xlimits=limits)
    samples = sampling(npts)
    dp = samples
    #dp = sampling_lhs(dim,npts,box)

    rf = RandomForestClassifier(random_state=0)
    rf.fit(x_train, y_train)
    #print("finished with training RF")

    prd =  rf.predict(x_test)
    acc_test = metrics.accuracy_score(y_test, prd)

    pred = rf.predict(dp)

    if grow_deep == True:
        x_train = dp
        y_train = pred

    temp = prim_norm(dp, pred, x_test, y_test, x_train, y_train, box, minpts=minpts, pasting=pasting)

    pr_pred = temp[0]
    pr_pred_train = temp[1]
    boxes_pred = temp[2]

    return [pr_pred, pr_pred_train, boxes_pred, acc_test]


def prim_rf_unc_alg(x_train, y_train, x_test, y_test, box, npts = 100000, minpts=20.0, grow_deep=False, pasting=False):
    X_raw = x_train
    y_raw = y_train
    dim = x_train.shape[1]
    n_labeled_examples = x_train.shape[0]

    limits = get_limits(x_train)
    sampling = LHS(xlimits=limits)
    samples = sampling(npts)
    dp = samples
    #dp = sampling_lhs(dim,npts,box)
    #print(dp)
    #print(dp.shape)
    #s_size= 20

    #np.random.seed(random_seed)
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=2)
    found = False
    while found is False:
            training_indices = np.random.randint(low=0, high=n_labeled_examples, size=2)
            uniqueValues, occurCount = np.unique(y_train[training_indices], return_counts=True)
            if uniqueValues.shape[0]==2:
                if occurCount[0] == 1 and occurCount[1] == 1:
                    found = True

    #print(training_indices)
    X_train_init = X_raw[training_indices]
    y_train_init = y_raw[training_indices]
    X_pool = np.delete(X_raw, training_indices, axis=0)
    y_pool = np.delete(y_raw, training_indices, axis=0)

    learner = ActiveLearner(estimator=RandomForestClassifier(random_state=0), X_training=X_train_init, y_training=y_train_init)

    performance = [metrics.accuracy_score(y_raw,learner.predict(X_raw))]

    nsamples = int(X_pool.shape[0] * 0.75)
    for n in range(nsamples):
        query_index, query_instance = learner.query(X_pool)
        X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
        learner.teach(X=X, y=y)
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
        performance.append(metrics.accuracy_score(y_raw,learner.predict(X_raw)))

    #print("finished with training RF(AL)")
    prd = learner.predict(x_test)
    acc_test = metrics.accuracy_score(y_test, prd)

    pred = learner.predict(dp)

    if grow_deep == True:
        x_train = dp
        y_train = pred

    temp = prim_norm(dp, pred, x_test, y_test, x_train, y_train, box, pasting=pasting)

    pr_pred = temp[0]
    pr_pred_train = temp[1]
    boxes_pred = temp[2]

    return [pr_pred, pr_pred_train, boxes_pred, acc_test]


def prim_rf_qbc_alg(x_train, y_train, x_test, y_test, box, npts = 100000, minpts=20.0, grow_deep=False, pasting=False):
    X_raw = x_train
    y_raw = y_train
    dim = x_train.shape[1]
    n_labeled_examples = x_train.shape[0]

    limits = get_limits(x_train)
    sampling = LHS(xlimits=limits)
    samples = sampling(npts)
    dp = samples
    #dp = sampling_lhs(dim,npts,box)

    n_members = 2
    learner_list = list()
    #s_size= 20

    #np.random.seed(random_seed)
    for member_idx in range(n_members):
        train_idx = np.random.randint(low=0, high=n_labeled_examples, size=2)
        found = False
        while found is False:
                train_idx = np.random.randint(low=0, high=n_labeled_examples, size=2)
                uniqueValues, occurCount = np.unique(y_raw[train_idx], return_counts=True)
                if uniqueValues.shape[0]==2:
                    if occurCount[0] == 1 and occurCount[1] == 1:
                        found = True

        X_train_init = X_raw[train_idx]
        y_train_init = y_raw[train_idx]
        X_pool = np.delete(X_raw, train_idx, axis=0)
        y_pool = np.delete(y_raw, train_idx)
        learner = ActiveLearner(estimator=RandomForestClassifier(random_state=0), X_training=X_train_init, y_training=y_train_init)
        learner_list.append(learner)

    committee = Committee(learner_list=learner_list)

    nsamples = int(X_pool.shape[0] * 0.75)
    for n in range(nsamples):
        query_idx, query_instance = committee.query(X_pool)
        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

    prd = committee.predict(x_test)
    acc_test = metrics.accuracy_score(y_test, prd)

    pred = committee.predict(dp)

    if grow_deep == True:
        x_train = dp
        y_train = pred

    temp = prim_norm(dp, pred, x_test, y_test, x_train, y_train, box, minpts=minpts, pasting=pasting)

    pr_pred = temp[0]
    pr_pred_train = temp[1]
    boxes_pred = temp[2]

    return [pr_pred, pr_pred_train, boxes_pred, acc_test]


def prim_svm_qbc_alg(x_train, y_train, x_test, y_test, box, npts = 100000, minpts=20.0, grow_deep=False, pasting=False):
    X_raw = x_train
    y_raw = y_train
    dim = x_train.shape[1]
    n_labeled_examples = x_train.shape[0]

    limits = get_limits(x_train)
    sampling = LHS(xlimits=limits)
    samples = sampling(npts)
    dp = samples
    #dp = sampling_lhs(dim,npts,box)

    n_members = 2
    learner_list = list()
    #s_size = 20

    #np.random.seed(random_seed)
    for member_idx in range(n_members):
        train_idx = np.random.randint(low=0, high=n_labeled_examples, size=2)
        found = False
        while found is False:
                train_idx = np.random.randint(low=0, high=n_labeled_examples, size=2)
                uniqueValues, occurCount = np.unique(y_raw[train_idx], return_counts=True)
                if uniqueValues.shape[0]==2:
                    if occurCount[0] == 1 and occurCount[1] == 1:
                        found = True

        X_train_init = X_raw[train_idx]
        y_train_init = y_raw[train_idx]
        X_pool = np.delete(X_raw, train_idx, axis=0)
        y_pool = np.delete(y_raw, train_idx)
        learner = ActiveLearner(estimator=svm.SVC(kernel='rbf', random_state=0, probability=True), X_training=X_train_init, y_training=y_train_init)
        learner_list.append(learner)
    committee = Committee(learner_list=learner_list)

    nsamples = int(X_pool.shape[0] * 0.75)
    for n in range(nsamples):
        query_idx, query_instance = committee.query(X_pool)
        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

    prd = committee.predict(x_test)
    acc_test = metrics.accuracy_score(y_test, prd)

    pred = committee.predict(dp)

    if grow_deep == True:
        x_train = dp
        y_train = pred

    temp = prim_norm(dp, pred, x_test, y_test, x_train, y_train, box, minpts=minpts, pasting=pasting)

    pr_pred = temp[0]
    pr_pred_train = temp[1]
    boxes_pred = temp[2]

    return [pr_pred, pr_pred_train, boxes_pred, acc_test]


def prim_svm_unc_alg(x_train, y_train, x_test, y_test, box, npts = 100000, minpts=20.0, grow_deep=False, pasting=False):
    X_raw = x_train
    y_raw = y_train
    dim = x_train.shape[1]
    n_labeled_examples = x_train.shape[0]
    #s_size = 20

    limits = get_limits(x_train)
    sampling = LHS(xlimits=limits)
    samples = sampling(npts)
    dp = samples
    #dp = sampling_lhs(dim,npts,box)

    #np.random.seed(random_seed)
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=2)
    found = False
    while found is False:
            training_indices = np.random.randint(low=0, high=n_labeled_examples, size=2)
            uniqueValues, occurCount = np.unique(y_train[training_indices], return_counts=True)
            if uniqueValues.shape[0]==2:
                if occurCount[0] == 1 and occurCount[1] == 1:
                    found = True

    X_train_init = X_raw[training_indices]
    y_train_init = y_raw[training_indices]
    X_pool = np.delete(X_raw, training_indices, axis=0)
    y_pool = np.delete(y_raw, training_indices, axis=0)

    learner = ActiveLearner(estimator=svm.SVC(kernel='rbf', random_state=0, probability=True), X_training=X_train_init, y_training=y_train_init)

    performance = [metrics.accuracy_score(y_raw,learner.predict(X_raw))]

    nsamples = int(X_pool.shape[0] * 0.75)
    for n in range(nsamples):
        query_index, query_instance = learner.query(X_pool)
        X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
        learner.teach(X=X, y=y)
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
        performance.append(metrics.accuracy_score(y_raw,learner.predict(X_raw)))

    #print("finished with training RF(AL)")
    prd = learner.predict(x_test)
    acc_test = metrics.accuracy_score(y_test, prd)

    pred = learner.predict(dp)

    if grow_deep == True:
        x_train = dp
        y_train = pred

    temp = prim_norm(dp, pred, x_test, y_test, x_train, y_train, box, minpts=minpts, pasting=pasting)

    pr_pred = temp[0]
    pr_pred_train = temp[1]
    boxes_pred = temp[2]

    return [pr_pred, pr_pred_train, boxes_pred, acc_test]


def prim_gpc_unc_alg(x_train, y_train, x_test, y_test, box, npts = 100000, minpts=20.0, grow_deep=False, pasting=False):
    X_raw = x_train
    y_raw = y_train
    dim = x_train.shape[1]
    n_labeled_examples = x_train.shape[0]
    #s_size = 20

    limits = get_limits(x_train)
    sampling = LHS(xlimits=limits)
    samples = sampling(npts)
    dp = samples
    #dp = sampling_lhs(dim,npts,box)

    #np.random.seed(random_seed)
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=2)
    found = False
    while found is False:
            training_indices = np.random.randint(low=0, high=n_labeled_examples, size=2)
            uniqueValues, occurCount = np.unique(y_train[training_indices], return_counts=True)
            if uniqueValues.shape[0]==2:
                if occurCount[0] == 1 and occurCount[1] == 1:
                    found = True

    X_train_init = X_raw[training_indices]
    y_train_init = y_raw[training_indices]
    X_pool = np.delete(X_raw, training_indices, axis=0)
    y_pool = np.delete(y_raw, training_indices, axis=0)

    learner = ActiveLearner(estimator=GaussianProcessClassifier(kernel=kernel, random_state=0), X_training=X_train_init, y_training=y_train_init)

    performance = [metrics.accuracy_score(y_raw,learner.predict(X_raw))]

    nsamples = int(X_pool.shape[0] * 0.75)
    for n in range(nsamples):
        query_index, query_instance = learner.query(X_pool)
        X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
        learner.teach(X=X, y=y)
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
        performance.append(metrics.accuracy_score(y_raw,learner.predict(X_raw)))

    #print("finished with training RF(AL)")
    prd = learner.predict(x_test)
    acc_test = metrics.accuracy_score(y_test, prd)

    pred = learner.predict(dp)

    if grow_deep == True:
        x_train = dp
        y_train = pred

    temp = prim_norm(dp, pred, x_test, y_test, x_train, y_train, box, minpts=minpts, pasting=pasting)

    pr_pred = temp[0]
    pr_pred_train = temp[1]
    boxes_pred = temp[2]

    return [pr_pred, pr_pred_train, boxes_pred, acc_test]


def prim_gpc_qbc_alg(x_train, y_train, x_test, y_test, box, npts = 100000, minpts=20.0, grow_deep=False, pasting=False):
    X_raw = x_train
    y_raw = y_train
    dim = x_train.shape[1]
    n_labeled_examples = x_train.shape[0]

    limits = get_limits(x_train)
    sampling = LHS(xlimits=limits)
    samples = sampling(npts)
    dp = samples
    #dp = sampling_lhs(dim,npts,box)

    n_members = 2
    learner_list = list()
    #s_size = 20

    #np.random.seed(random_seed)
    for member_idx in range(n_members):
        train_idx = np.random.randint(low=0, high=n_labeled_examples, size=2)
        found = False
        while found is False:
                train_idx = np.random.randint(low=0, high=n_labeled_examples, size=2)
                uniqueValues, occurCount = np.unique(y_raw[train_idx], return_counts=True)
                if uniqueValues.shape[0]==2:
                    if occurCount[0] == 1 and occurCount[1] == 1:
                        found = True

        X_train_init = X_raw[train_idx]
        y_train_init = y_raw[train_idx]
        X_pool = np.delete(X_raw, train_idx, axis=0)
        y_pool = np.delete(y_raw, train_idx)
        learner = ActiveLearner(estimator=GaussianProcessClassifier(kernel=kernel, random_state=0), X_training=X_train_init, y_training=y_train_init)
        learner_list.append(learner)
    committee = Committee(learner_list=learner_list)

    nsamples = int(X_pool.shape[0] * 0.75)
    for n in range(nsamples):
        query_idx, query_instance = committee.query(X_pool)
        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

    prd = committee.predict(x_test)
    acc_test = metrics.accuracy_score(y_test, prd)

    pred = committee.predict(dp)

    if grow_deep == True:
        x_train = dp
        y_train = pred

    temp = prim_norm(dp, pred, x_test, y_test, x_train, y_train, box, minpts=minpts, pasting=pasting)

    pr_pred = temp[0]
    pr_pred_train = temp[1]
    boxes_pred = temp[2]

    return [pr_pred, pr_pred_train, boxes_pred, acc_test]


def main():
    sdre = pd.read_csv("data/sdre.csv")
    sdre_test = pd.read_csv("data/sdre_test.csv")
    morris = pd.read_csv("data/morris.csv")
    ellipse = pd.read_csv("data/ellipse.csv")
    hart3 = pd.read_csv("data/hart3.csv")
    linketal06dec = pd.read_csv("data/linketal06dec.csv")
    linketal06simple = pd.read_csv("data/linketal06simple.csv")
    moon10hdc1e = pd.read_csv("data/moon10hdc1e.csv")
    moon10low = pd.read_csv("data/moon10low.csv")
    morretal06 = pd.read_csv("data/morretal06.csv")
    willetal06 = pd.read_csv("data/willetal06.csv")
    wingweight = pd.read_csv("data/wingweight.csv")
    oakoh04 = pd.read_csv("data/oakoh04.csv")

    datasets = [sdre,morris,ellipse,hart3,linketal06dec,linketal06simple,moon10hdc1e,moon10low,morretal06,willetal06,wingweight,oakoh04]


    avg_auc_prim = []
    avg_auc_prim_rf = []
    avg_auc_prim_rf_unc = []
    avg_auc_prim_rf_qbc = []
    avg_auc_prim_svm_qbc = []
    avg_auc_prim_svm_unc = []
    avg_auc_prim_gpc_unc = []
    avg_auc_prim_gpc_qbc = []
    avg_res_dim_prim = []
    avg_res_dim_prim_rf = []
    avg_res_dim_prim_rf_unc = []
    avg_res_dim_prim_rf_qbc = []
    avg_res_dim_prim_svm_unc = []
    avg_res_dim_prim_svm_qbc = []
    avg_res_dim_prim_gpc_unc = []
    avg_res_dim_prim_gpc_qbc = []
    avg_cons_prim = []
    avg_cons_prim_rf = []
    avg_cons_prim_rf_unc = []
    avg_cons_prim_rf_qbc = []
    avg_cons_prim_svm_qbc = []
    avg_cons_prim_svm_unc = []
    avg_cons_prim_gpc_unc = []
    avg_cons_prim_gpc_qbc = []
    avg_dens_prim = []
    avg_dens_prim_rf = []
    avg_dens_prim_rf_unc = []
    avg_dens_prim_rf_qbc = []
    avg_dens_prim_svm_qbc = []
    avg_dens_prim_svm_unc = []
    avg_dens_prim_gpc_unc = []
    avg_dens_prim_gpc_qbc = []
    avg_acc_rf = []
    avg_acc_rf_unc = []
    avg_acc_rf_qbc = []
    avg_acc_svm_qbc = []
    avg_acc_svm_unc = []
    avg_acc_gpc_unc = []
    avg_acc_gpc_qbc = []

    print("-------------------------------------- Started Experiments with all datasets ----------------------------------")
    pool_sizes = [400,800,1600]

    for pool_size in pool_sizes:
        d = 0
        prec_n = []
        prec_r = []
        prec_rf_unc = []
        prec_rf_qbc = []
        prec_svm_qbc = []
        prec_svm_unc = []
        prec_gpc_unc = []
        prec_gpc_qbc = []
        rec_n = []
        rec_r = []
        rec_rf_unc = []
        rec_rf_qbc = []
        rec_svm_qbc = []
        rec_svm_unc = []
        rec_gpc_unc = []
        rec_gpc_qbc = []
        auc_n = []
        auc_r = []
        auc_rf_unc = []
        auc_rf_qbc = []
        auc_svm_qbc = []
        auc_svm_unc = []
        auc_gpc_unc = []
        auc_gpc_qbc = []
        res_dim_primn = []
        res_dim_primr = []
        res_dim_rf_unc = []
        res_dim_rf_qbc = []
        res_dim_svm_qbc = []
        res_dim_svm_unc = []
        res_dim_gpc_unc = []
        res_dim_gpc_qbc = []
        cons_prim_n = []
        cons_prim_r = []
        cons_prim_rf_unc = []
        cons_prim_rf_qbc = []
        cons_prim_svm_unc = []
        cons_prim_svm_qbc = []
        cons_prim_gpc_unc = []
        cons_prim_gpc_qbc = []
        dens_prim_n = []
        dens_prim_r = []
        dens_prim_rf_unc = []
        dens_prim_rf_qbc = []
        dens_prim_svm_unc = []
        dens_prim_svm_qbc = []
        dens_prim_gpc_unc = []
        dens_prim_gpc_qbc = []
        acc_rf = []
        acc_rf_unc = []
        acc_rf_qbc = []
        acc_svm_qbc = []
        acc_svm_unc = []
        acc_gpc_unc = []
        acc_gpc_qbc = []

        main_dp = True

        for dataset in datasets:
            #print(dataset)
            X_train = dataset.iloc[:, 1:-1].values
            y_train = dataset.iloc[:, -1].values
            #if main_dp is True:
            #    X_test = sdre_test.iloc[:, 1:-1].values
            #    y_test = sdre_test.iloc[:, -1].values
            #    main_dp = False
            #else:
            X_test = dataset.iloc[:, 1:-1].values
            y_test = dataset.iloc[:, -1].values

            print("--------------- Started Experiment with " + str(pool_size) + " samples and dataset "+str(d)+" ---------------")
            NRUNS = 50

            prev_box_n = None
            prev_box_n = None
            prev_box_n = None
            prev_box_n = None
            prev_box_n = None
            prev_box_n = None
            prev_box_n = None


            for i in range(NRUNS):
                print("Run : " + str(i+1) + " out of " + str(NRUNS))
                training_indices = np.random.randint(low=0, high=X_train.shape[0], size=pool_size)
                found = False
                while found is False:
                    training_indices = np.random.randint(low=0, high=X_train.shape[0], size=pool_size)
                    uniqueValues, occurCount = np.unique(y_train[training_indices], return_counts=True)
                    if uniqueValues.shape[0]==2:
                        if occurCount[0] > 10 and occurCount[1] > 10:
                            found = True

                test_indices = np.random.randint(low=0, high=X_test.shape[0], size=10000)

                dp_train = X_train[training_indices]
                label_train = y_train[training_indices]
                dp_test = X_test[test_indices]
                label_test = y_test[test_indices]
                init_box = make_box(dp_train)

                prim_n = prim_norm(dp_train, label_train, dp_test, label_test, dp_train, label_train, init_box)
                prim_r = prim_rf(dp_train, label_train, dp_test, label_test, init_box, grow_deep=True)
                prim_rf_unc = prim_rf_unc_alg(dp_train, label_train, dp_test, label_test, init_box, grow_deep=True)
                prim_rf_qbc = prim_rf_qbc_alg(dp_train, label_train, dp_test, label_test, init_box, grow_deep=True)
                prim_svm_unc = prim_svm_unc_alg(dp_train, label_train, dp_test, label_test, init_box, grow_deep=True)
                prim_svm_qbc = prim_svm_qbc_alg(dp_train, label_train, dp_test, label_test, init_box, grow_deep=True)
                prim_gpc_unc = prim_gpc_unc_alg(dp_train, label_train, dp_test, label_test, init_box, grow_deep=True)
                prim_gpc_qbc = prim_gpc_qbc_alg(dp_train, label_train, dp_test, label_test, init_box, grow_deep=True)

                prec_n.append(prim_n[0][:,0])
                prec_r.append(prim_r[0][:,0])
                prec_rf_unc.append(prim_rf_unc[0][:,0])
                prec_rf_qbc.append(prim_rf_qbc[0][:,0])
                prec_svm_qbc.append(prim_svm_qbc[0][:,0])
                prec_svm_unc.append(prim_svm_unc[0][:,0])
                prec_gpc_unc.append(prim_gpc_unc[0][:,0])
                prec_gpc_qbc.append(prim_gpc_qbc[0][:,0])

                rec_n.append(prim_n[0][:,1])
                rec_r.append(prim_r[0][:,1])
                rec_rf_unc.append(prim_rf_unc[0][:,1])
                rec_rf_qbc.append(prim_rf_qbc[0][:,1])
                rec_svm_qbc.append(prim_svm_qbc[0][:,0])
                rec_svm_unc.append(prim_svm_unc[0][:,0])
                rec_gpc_unc.append(prim_gpc_unc[0][:,0])
                rec_gpc_qbc.append(prim_gpc_qbc[0][:,0])

                auc_n.append(auc(prim_n[0]))
                auc_r.append(auc(prim_r[0]))
                auc_rf_unc.append(auc(prim_rf_unc[0]))
                auc_rf_qbc.append(auc(prim_rf_qbc[0]))
                auc_svm_qbc.append(auc(prim_svm_qbc[0]))
                auc_svm_unc.append(auc(prim_svm_unc[0]))
                auc_gpc_unc.append(auc(prim_gpc_unc[0]))
                auc_gpc_qbc.append(auc(prim_gpc_qbc[0]))

                res_dim_primn.append(interpretability(prim_n[2][-1],init_box))
                res_dim_primr.append(interpretability(prim_r[2][-1],init_box))
                res_dim_rf_unc.append(interpretability(prim_rf_unc[2][-1],init_box))
                res_dim_rf_qbc.append(interpretability(prim_rf_qbc[2][-1],init_box))
                res_dim_svm_qbc.append(interpretability(prim_svm_qbc[2][-1],init_box))
                res_dim_svm_unc.append(interpretability(prim_svm_unc[2][-1],init_box))
                res_dim_gpc_unc.append(interpretability(prim_gpc_unc[2][-1],init_box))
                res_dim_gpc_qbc.append(interpretability(prim_gpc_qbc[2][-1],init_box))

                if(i>0):
                    cons_prim_n.append(consistencyv(prev_box_n, prim_n[2][-1], init_box))
                    cons_prim_r.append(consistencyv(prev_box_r, prim_r[2][-1], init_box))
                    cons_prim_rf_unc.append(consistencyv(prev_box_rf_unc,prim_rf_unc[2][-1], init_box))
                    cons_prim_rf_qbc.append(consistencyv(prev_box_rf_qbc, prim_rf_qbc[2][-1], init_box))
                    cons_prim_svm_unc.append(consistencyv(prev_box_svm_unc, prim_svm_unc[2][-1], init_box))
                    cons_prim_svm_qbc.append(consistencyv(prev_box_svm_qbc, prim_svm_qbc[2][-1], init_box))
                    cons_prim_gpc_unc.append(consistencyv(prev_box_n_gpc_unc, prim_gpc_unc[2][-1], init_box))
                    cons_prim_gpc_qbc.append(consistencyv(prev_box_n_gpc_qbc, prim_gpc_qbc[2][-1], init_box))

                prev_box_n = prim_n[2][-1]
                prev_box_r = prim_r[2][-1]
                prev_box_rf_unc = prim_rf_unc[2][-1]
                prev_box_rf_qbc = prim_rf_qbc[2][-1]
                prev_box_svm_unc = prim_svm_unc[2][-1]
                prev_box_svm_qbc = prim_svm_qbc[2][-1]
                prev_box_n_gpc_unc = prim_gpc_unc[2][-1]
                prev_box_n_gpc_qbc = prim_gpc_qbc[2][-1]

                dens_prim_n.append(prim_n[0][-1,0])
                dens_prim_r.append(prim_r[0][-1,0])
                dens_prim_rf_unc.append(prim_rf_unc[0][-1,0])
                dens_prim_rf_qbc.append(prim_rf_qbc[0][-1,0])
                dens_prim_svm_qbc.append(prim_svm_qbc[0][-1,0])
                dens_prim_svm_unc.append(prim_svm_unc[0][-1,0])
                dens_prim_gpc_unc.append(prim_gpc_unc[0][-1,0])
                dens_prim_gpc_qbc.append(prim_gpc_qbc[0][-1,0])

                acc_rf.append(prim_r[3])
                acc_rf_unc.append(prim_rf_unc[3])
                acc_rf_qbc.append(prim_rf_qbc[3])
                acc_svm_qbc.append(prim_svm_qbc[3])
                acc_svm_unc.append(prim_svm_unc[3])
                acc_gpc_unc.append(prim_gpc_unc[3])
                acc_gpc_qbc.append(prim_gpc_qbc[3])

            prec_n_np = np.hstack(np.array(prec_n))
            prec_r_np = np.hstack(np.array(prec_r))
            prec_rf_unc_np = np.hstack(np.array(prec_rf_unc))
            prec_rf_qbc_np = np.hstack(np.array(prec_rf_qbc))
            prec_svm_qbc_np = np.hstack(np.array(prec_svm_qbc))
            prec_svm_unc_np = np.hstack(np.array(prec_svm_unc))
            prec_gpc_unc_np = np.hstack(np.array(prec_gpc_unc))
            prec_gpc_qbc_np = np.hstack(np.array(prec_gpc_qbc))

            rec_n_np = np.hstack(np.array(rec_n))
            rec_r_np = np.hstack(np.array(rec_r))
            rec_rf_unc_np = np.hstack(np.array(rec_rf_unc))
            rec_rf_qbc_np = np.hstack(np.array(rec_rf_qbc))
            rec_svm_qbc_np = np.hstack(np.array(rec_svm_qbc))
            rec_svm_unc_np = np.hstack(np.array(rec_svm_unc))
            rec_gpc_unc_np = np.hstack(np.array(rec_gpc_unc))
            rec_gpc_qbc_np = np.hstack(np.array(rec_gpc_qbc))

            res_n_avg = lowess(prec_n_np, rec_n_np, frac=1./5)
            res_r_avg = lowess(prec_r_np, rec_r_np, frac=1./5)
            res_rf_unc_avg = lowess(prec_rf_unc_np, rec_rf_unc_np, frac=1./5)
            res_rf_qbc_avg = lowess(prec_rf_qbc_np, rec_rf_qbc_np, frac=1./5)
            res_svm_qbc_avg = lowess(prec_svm_qbc_np, rec_svm_qbc_np, frac=1./5)
            res_svm_unc_avg = lowess(prec_svm_unc_np, rec_svm_unc_np, frac=1./5)
            res_gpc_unc_avg = lowess(prec_gpc_unc_np, rec_gpc_unc_np, frac=1./5)
            res_gpc_qbc_avg = lowess(prec_gpc_qbc_np, rec_gpc_qbc_np, frac=1./5)

            prec_n_avg = res_n_avg[:,1]
            prec_r_avg = res_r_avg[:,1]
            prec_rf_unc_avg = res_rf_unc_avg[:,1]
            prec_rf_qbc_avg = res_rf_qbc_avg[:,1]
            prec_svm_qbc_avg = res_svm_qbc_avg[:,1]
            prec_svm_unc_avg = res_svm_unc_avg[:,1]
            prec_gpc_unc_avg = res_gpc_unc_avg[:,1]
            prec_gpc_qbc_avg = res_gpc_qbc_avg[:,1]

            rec_n_avg = res_n_avg[:,0]
            rec_r_avg = res_r_avg[:,0]
            rec_rf_unc_avg = res_rf_unc_avg[:,0]
            rec_rf_qbc_avg = res_rf_qbc_avg[:,0]
            rec_svm_qbc_avg = res_svm_qbc_avg[:,0]
            rec_svm_unc_avg = res_svm_unc_avg[:,0]
            rec_gpc_unc_avg = res_gpc_unc_avg[:,0]
            rec_gpc_qbc_avg = res_gpc_qbc_avg[:,0]

            aucs = [auc_n, auc_r, auc_rf_unc, auc_rf_qbc, auc_svm_unc, auc_svm_qbc, auc_gpc_unc, auc_gpc_qbc]
            interpretabilities = [res_dim_primn, res_dim_primr, res_dim_rf_unc, res_dim_rf_qbc, res_dim_svm_unc, res_dim_svm_qbc, res_dim_gpc_unc, res_dim_gpc_qbc]
            consistencies = [cons_prim_n, cons_prim_r, cons_prim_rf_unc, cons_prim_rf_qbc, cons_prim_svm_unc, cons_prim_svm_qbc, cons_prim_gpc_unc, cons_prim_gpc_qbc]
            densities = [dens_prim_n, dens_prim_r, dens_prim_rf_unc, dens_prim_rf_qbc, dens_prim_svm_unc, dens_prim_svm_qbc, dens_prim_gpc_unc, dens_prim_gpc_qbc]
            accuracies = [acc_rf, acc_rf_unc, acc_rf_qbc, acc_svm_unc, acc_svm_qbc, acc_gpc_unc, acc_gpc_qbc]

            dts = ""
            if d == 0:
                dts = "dsgc"
            elif d == 1:
                dts = "morris"
            elif d == 2:
                dts = "ellipse"
            elif d == 3:
                dts = "hart3"
            elif d == 4:
                dts = "linketal06dec"
            elif d == 5:
                dts = "linketal06simple"
            elif d == 6:
                dts = "moon10hdc1e"
            elif d == 7:
                dts = "moon10low"
            elif d == 8:
                dts = "morretal06"
            elif d == 9:
                dts = "willetal06"
            elif d == 10:
                dts = "wingweight"
            elif d == 11:
                dts = "oakoh04"
            else:
                dts = "undef"

            print("------------------- Results: |d| = " +str(pool_size)+ " -------------------")
            print("Mean AUC (PRIM_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(auc_n)))
            print("Mean AUC (PRIM+RF_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(auc_r)))
            print("Mean AUC (PRIM+RF+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(auc_rf_unc)))
            print("Mean AUC (PRIM+RF+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(auc_rf_qbc)))
            print("Mean AUC (PRIM+SVM+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(auc_svm_unc)))
            print("Mean AUC (PRIM+SVM+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(auc_svm_qbc)))
            print("Mean AUC (PRIM+GPC+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(auc_gpc_unc)))
            print("Mean AUC (PRIM+GPC+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(auc_gpc_qbc)))
            print("-----------------------------------------------------------------------")
            print("Mean Interpretability (PRIM_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(res_dim_primn)))
            print("Mean Interpretability (PRIM+RF_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(res_dim_primr)))
            print("Mean Interpretability (PRIM+RF+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(res_dim_rf_unc)))
            print("Mean Interpretability (PRIM+RF+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(res_dim_rf_qbc)))
            print("Mean Interpretability (PRIM+SVM+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(res_dim_svm_unc)))
            print("Mean Interpretability (PRIM+SVM+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(res_dim_svm_qbc)))
            print("Mean Interpretability (PRIM+GPC+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(res_dim_gpc_unc)))
            print("Mean Interpretability (PRIM+GPC+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(res_dim_gpc_qbc)))
            print("-----------------------------------------------------------------------")
            print("Mean Consistency (PRIM_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(cons_prim_n)))
            print("Mean Consistency (PRIM+RF_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(cons_prim_r)))
            print("Mean Consistency (PRIM+RF+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(cons_prim_rf_unc)))
            print("Mean Consistency (PRIM+RF+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(cons_prim_rf_qbc)))
            print("Mean Consistency (PRIM+SVM+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(cons_prim_svm_unc)))
            print("Mean Consistency (PRIM+SVM+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(cons_prim_svm_qbc)))
            print("Mean Consistency (PRIM+GPC+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(cons_prim_gpc_unc)))
            print("Mean Consistency (PRIM+GPC+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(cons_prim_gpc_qbc)))
            print("-----------------------------------------------------------------------")
            print("Mean Density (PRIM_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(dens_prim_n)))
            print("Mean Density (PRIM+RF_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(dens_prim_r)))
            print("Mean Density (PRIM+RF+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(dens_prim_rf_unc)))
            print("Mean Density (PRIM+RF+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(dens_prim_rf_qbc)))
            print("Mean Density (PRIM+SVM+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(dens_prim_svm_qbc)))
            print("Mean Density (PRIM+SVM+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(dens_prim_svm_unc)))
            print("Mean Density (PRIM+GPC+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(dens_prim_gpc_unc)))
            print("Mean Density (PRIM+GPC+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(dens_prim_gpc_qbc)))
            print("-----------------------------------------------------------------------")
            print("Mean Accuracy (PRIM+RF_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(acc_rf)))
            print("Mean Accuracy (PRIM+RF+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(acc_rf_unc)))
            print("Mean Accuracy (PRIM+RF+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(acc_rf_qbc)))
            print("Mean Accuracy (PRIM+SVM+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(acc_svm_qbc)))
            print("Mean Accuracy (PRIM+SVM+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(acc_svm_unc)))
            print("Mean Accuracy (PRIM+GPC+UNC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(acc_gpc_unc)))
            print("Mean Accuracy (PRIM+GPC+QBC_"+str(pool_size)+")_"+dts+" : " +str(np.nanmean(acc_gpc_qbc)))

            avg_auc_prim.append(np.nanmean(auc_n))
            avg_auc_prim_rf.append(np.nanmean(auc_r))
            avg_auc_prim_rf_unc.append(np.nanmean(auc_rf_unc))
            avg_auc_prim_rf_qbc.append(np.nanmean(auc_rf_qbc))
            avg_auc_prim_svm_qbc.append(np.nanmean(auc_svm_qbc))
            avg_auc_prim_svm_unc.append(np.nanmean(auc_svm_unc))
            avg_auc_prim_gpc_unc.append(np.nanmean(auc_gpc_unc))
            avg_auc_prim_gpc_qbc.append(np.nanmean(auc_gpc_qbc))

            avg_res_dim_prim.append(np.nanmean(res_dim_primn))
            avg_res_dim_prim_rf.append(np.nanmean(res_dim_primr))
            avg_res_dim_prim_rf_unc.append(np.nanmean(res_dim_rf_unc))
            avg_res_dim_prim_rf_qbc.append(np.nanmean(res_dim_rf_qbc))
            avg_res_dim_prim_svm_qbc.append(np.nanmean(res_dim_svm_qbc))
            avg_res_dim_prim_svm_unc.append(np.nanmean(res_dim_svm_unc))
            avg_res_dim_prim_gpc_unc.append(np.nanmean(res_dim_gpc_unc))
            avg_res_dim_prim_gpc_qbc.append(np.nanmean(res_dim_gpc_qbc))

            avg_cons_prim.append(np.nanmean(cons_prim_n))
            avg_cons_prim_rf.append(np.nanmean(cons_prim_r))
            avg_cons_prim_rf_unc.append(np.nanmean(cons_prim_rf_unc))
            avg_cons_prim_rf_qbc.append(np.nanmean(cons_prim_rf_qbc))
            avg_cons_prim_svm_unc.append(np.nanmean(cons_prim_svm_unc))
            avg_cons_prim_svm_qbc.append(np.nanmean(cons_prim_svm_qbc))
            avg_cons_prim_gpc_unc.append(np.nanmean(cons_prim_gpc_unc))
            avg_cons_prim_gpc_qbc.append(np.nanmean(cons_prim_gpc_qbc))

            avg_dens_prim.append(np.nanmean(dens_prim_n))
            avg_dens_prim_rf.append(np.nanmean(dens_prim_r))
            avg_dens_prim_rf_unc.append(np.nanmean(dens_prim_rf_unc))
            avg_dens_prim_rf_qbc.append(np.nanmean(dens_prim_rf_qbc))
            avg_dens_prim_svm_unc.append(np.nanmean(dens_prim_svm_unc))
            avg_dens_prim_svm_qbc.append(np.nanmean(dens_prim_svm_qbc))
            avg_dens_prim_gpc_unc.append(np.nanmean(dens_prim_gpc_unc))
            avg_dens_prim_gpc_qbc.append(np.nanmean(dens_prim_gpc_qbc))

            avg_acc_rf.append(np.nanmean(acc_rf))
            avg_acc_rf_unc.append(np.nanmean(acc_rf_unc))
            avg_acc_rf_qbc.append(np.nanmean(acc_rf_qbc))
            avg_acc_svm_qbc.append(np.nanmean(acc_svm_qbc))
            avg_acc_svm_unc.append(np.nanmean(acc_svm_unc))
            avg_acc_gpc_unc.append(np.nanmean(acc_gpc_unc))
            avg_acc_gpc_qbc.append(np.nanmean(acc_gpc_qbc))

            fig_peeling_curve = plt.figure()
            ax_peeling_curve = fig_peeling_curve.add_subplot()
            ax_peeling_curve.plot(rec_n_avg, prec_n_avg, label = "PRIM")
            ax_peeling_curve.plot(rec_r_avg, prec_r_avg, label = "PRIM+RF")
            ax_peeling_curve.plot(rec_rf_unc_avg, prec_rf_unc_avg, label = "PRIM+RF+UNC")
            ax_peeling_curve.plot(rec_rf_qbc_avg, prec_rf_qbc_avg, label = "PRIM+RF+QBC")
            ax_peeling_curve.plot(rec_svm_qbc_avg, prec_svm_qbc_avg, label = "PRIM+SVM+QBC")
            ax_peeling_curve.plot(rec_svm_unc_avg, prec_svm_unc_avg, label = "PRIM+SVM+UNC")
            ax_peeling_curve.plot(rec_gpc_unc_avg, prec_gpc_unc_avg, label = "PRIM+GPC+UNC")
            ax_peeling_curve.plot(rec_gpc_qbc_avg, prec_gpc_qbc_avg, label = "PRIM+GPC+QBC")
            ax_peeling_curve.set(title= 'peeling curve |d|='+str(pool_size))
            ax_peeling_curve.set_xlabel('coverage')
            ax_peeling_curve.set_ylabel('density')
            ax_peeling_curve.legend()
            fig_peeling_curve.savefig('peeling-curve_|d|='+str(pool_size)+ "_" + dts +".png")

            fig_auc = plt.figure(figsize=(20, 10))
            ax_auc = fig_auc.add_subplot()
            bp_auc = ax_auc.boxplot(aucs)
            ax_auc.set_xticklabels(['PRIM','PRIM+RF','PRIM+RF+UNC','PRIM+RF+QBC','PRIM+SVM+UNC','PRIM+SVM+QBC','PRIM+GPC+UNC','PRIM+GPC+QBC'])
            ax_auc.get_xaxis().tick_bottom()
            ax_auc.get_yaxis().tick_left()
            ax_auc.set(title= 'AUC |d|='+str(pool_size)+ " " + dts)
            fig_auc.savefig('auc_|d|='+str(pool_size)+ "_" + dts +".png", bbox_inches='tight')

            fig_cons = plt.figure(figsize=(20, 10))
            ax_cons = fig_cons.add_subplot()
            bp_cons = ax_cons.boxplot(interpretabilities)
            ax_cons.set_xticklabels(['PRIM','PRIM+RF','PRIM+RF+UNC','PRIM+RF+QBC','PRIM+SVM+UNC','PRIM+SVM+QBC','PRIM+GPC+UNC','PRIM+GPC+QBC'])
            ax_cons.get_xaxis().tick_bottom()
            ax_cons.get_yaxis().tick_left()
            ax_cons.set(title= 'Interpretability |d|='+str(pool_size)+ " " + dts)
            fig_cons.savefig('interpretability_|d|='+str(pool_size)+ "_" + dts +".png", bbox_inches='tight')

            fig_cons = plt.figure(figsize=(20, 10))
            ax_cons = fig_cons.add_subplot()
            bp_cons = ax_cons.boxplot(consistencies)
            ax_cons.set_xticklabels(['PRIM','PRIM+RF','PRIM+RF+UNC','PRIM+RF+QBC','PRIM+SVM+UNC','PRIM+SVM+QBC','PRIM+GPC+UNC','PRIM+GPC+QBC'])
            ax_cons.get_xaxis().tick_bottom()
            ax_cons.get_yaxis().tick_left()
            ax_cons.set(title= 'Consistency |d|='+str(pool_size)+ " " + dts)
            fig_cons.savefig('consistency_|d|='+str(pool_size)+ "_" + dts +".png", bbox_inches='tight')

            fig_dens = plt.figure(figsize=(20, 10))
            ax_dens = fig_dens.add_subplot()
            bp_dens = ax_dens.boxplot(densities)
            ax_dens.set_xticklabels(['PRIM','PRIM+RF','PRIM+RF+UNC','PRIM+RF+QBC','PRIM+SVM+UNC','PRIM+SVM+QBC','PRIM+GPC+UNC','PRIM+GPC+QBC'])
            ax_dens.get_xaxis().tick_bottom()
            ax_dens.get_yaxis().tick_left()
            ax_dens.set(title= 'Density |d|='+str(pool_size)+ " " + dts)
            fig_dens.savefig('density_|d|='+str(pool_size)+ "_" + dts +".png", bbox_inches='tight')

            fig_acc = plt.figure(figsize=(20, 10))
            ax_acc = fig_acc.add_subplot()
            bp_acc = ax_acc.boxplot(accuracies)
            ax_acc.set_xticklabels(['PRIM+RF','PRIM+RF+UNC','PRIM+RF+QBC','PRIM+SVM+UNC','PRIM+SVM+QBC','PRIM+GPC+UNC','PRIM+GPC+QBC'])
            ax_acc.get_xaxis().tick_bottom()
            ax_acc.get_yaxis().tick_left()
            ax_acc.set(title= 'Accuracy |d|='+str(pool_size)+ " " + dts)
            fig_acc.savefig('accuracy_|d|='+str(pool_size)+ "_" + dts +".png", bbox_inches='tight')

            fig_acc_curve = plt.figure()
            ax_acc_curve = fig_acc_curve.add_subplot()
            ax_acc_curve.plot(acc_rf, 'o', label = "RF")
            ax_acc_curve.plot(acc_rf_unc, 'o', label = "RF+UNC")
            ax_acc_curve.plot(acc_rf_qbc, 'o', label = "RF+QBC")
            ax_acc_curve.plot(acc_svm_unc, 'o', label = "SVM+UNC")
            ax_acc_curve.plot(acc_svm_qbc, 'o', label = "SVM+QBC")
            ax_acc_curve.plot(acc_gpc_unc, 'o', label = "GPC+UNC")
            ax_acc_curve.plot(acc_gpc_qbc, 'o', label = "GPC+QBC")
            ax_acc_curve.set(title= 'Accuracy |d|='+str(pool_size)+ " " + dts)
            ax_acc_curve.set_ylabel('acc')
            ax_acc_curve.legend()
            fig_acc_curve.savefig('acc-curve_|d|='+str(pool_size)+ dts +".png")

            print("---------------------------- Experiment Finished --------------------------")

            prec_n = []
            prec_r = []
            prec_rf_unc = []
            prec_rf_qbc = []
            prec_svm_qbc = []
            prec_svm_unc = []
            prec_gpc_unc = []
            prec_gpc_qbc = []
            rec_n = []
            rec_r = []
            rec_rf_unc = []
            rec_rf_qbc = []
            rec_svm_qbc = []
            rec_svm_unc = []
            rec_gpc_unc = []
            rec_gpc_qbc = []
            auc_n = []
            auc_r = []
            auc_rf_unc = []
            auc_rf_qbc = []
            auc_svm_qbc = []
            auc_svm_unc = []
            auc_gpc_unc = []
            auc_gpc_qbc = []
            res_dim_primn = []
            res_dim_primr = []
            res_dim_rf_unc = []
            res_dim_rf_qbc = []
            res_dim_svm_qbc = []
            res_dim_svm_unc = []
            res_dim_gpc_unc = []
            res_dim_gpc_qbc = []
            cons_prim_n = []
            cons_prim_r = []
            cons_prim_rf_unc = []
            cons_prim_rf_qbc = []
            cons_prim_svm_unc = []
            cons_prim_svm_qbc = []
            cons_prim_gpc_unc = []
            cons_prim_gpc_qbc = []
            dens_prim_n = []
            dens_prim_r = []
            dens_prim_rf_unc = []
            dens_prim_rf_qbc = []
            dens_prim_svm_unc = []
            dens_prim_svm_qbc = []
            dens_prim_gpc_unc = []
            dens_prim_gpc_qbc = []
            acc_rf = []
            acc_rf_unc = []
            acc_rf_qbc = []
            acc_svm_qbc = []
            acc_svm_unc = []
            acc_gpc_unc = []
            acc_gpc_qbc = []

            d = d + 1

            plt.close('all')

        avg_aucs = [avg_auc_prim,avg_auc_prim_rf,avg_auc_prim_rf_unc,avg_auc_prim_rf_qbc,avg_auc_prim_svm_unc,avg_auc_prim_svm_qbc,avg_auc_prim_gpc_unc,avg_auc_prim_gpc_qbc]
        avg_res_dim = [avg_res_dim_prim,avg_res_dim_prim_rf,avg_res_dim_prim_rf_unc,avg_res_dim_prim_rf_qbc,avg_res_dim_prim_svm_unc,avg_res_dim_prim_svm_qbc,avg_res_dim_prim_gpc_unc,avg_res_dim_prim_gpc_qbc]
        avg_cons = [avg_cons_prim,avg_cons_prim_rf,avg_cons_prim_rf_unc,avg_cons_prim_rf_qbc,avg_cons_prim_svm_unc,avg_cons_prim_svm_qbc,avg_cons_prim_gpc_unc,avg_cons_prim_gpc_qbc]
        avg_dens = [avg_dens_prim,avg_dens_prim_rf,avg_dens_prim_rf_unc,avg_dens_prim_rf_qbc,avg_dens_prim_svm_unc,avg_dens_prim_svm_qbc,avg_dens_prim_gpc_unc,avg_dens_prim_gpc_qbc]
        avg_accs = [avg_acc_rf,avg_acc_rf_unc,avg_acc_rf_qbc,avg_acc_svm_unc,avg_acc_svm_qbc,avg_acc_gpc_unc,avg_acc_gpc_qbc]


        fig_avg_auc = plt.figure(figsize=(20, 10))
        ax_avg_auc = fig_avg_auc.add_subplot()
        bp_avg_auc = ax_avg_auc.boxplot(avg_aucs)
        ax_avg_auc.set_xticklabels(['PRIM','PRIM+RF','PRIM+RF+UNC','PRIM+RF+QBC','PRIM+SVM+UNC','PRIM+SVM+QBC','PRIM+GPC+UNC','PRIM+GPC+QBC'])
        ax_avg_auc.get_xaxis().tick_bottom()
        ax_avg_auc.get_yaxis().tick_left()
        ax_avg_auc.set(title= 'Average AUC (Total)')
        fig_avg_auc.savefig('avg-auc_'+str(pool_size)+'.png', bbox_inches='tight')

        fig_avg_res_dim = plt.figure(figsize=(20, 10))
        ax_avg_res_dim = fig_avg_res_dim.add_subplot()
        bp_avg_res_dim = ax_avg_res_dim.boxplot(avg_res_dim)
        ax_avg_res_dim.set_xticklabels(['PRIM','PRIM+RF','PRIM+RF+UNC','PRIM+RF+QBC','PRIM+SVM+UNC','PRIM+SVM+QBC','PRIM+GPC+UNC','PRIM+GPC+QBC'])
        ax_avg_res_dim.get_xaxis().tick_bottom()
        ax_avg_res_dim.get_yaxis().tick_left()
        ax_avg_res_dim.set(title= 'Average Interpretability (Total)')
        fig_avg_res_dim.savefig('avg-res_dim_'+str(pool_size)+'.png', bbox_inches='tight')

        fig_avg_cons = plt.figure(figsize=(20, 10))
        ax_avg_cons = fig_avg_cons.add_subplot()
        bp_avg_cons = ax_avg_cons.boxplot(avg_cons)
        ax_avg_cons.set_xticklabels(['PRIM','PRIM+RF','PRIM+RF+UNC','PRIM+RF+QBC','PRIM+SVM+UNC','PRIM+SVM+QBC','PRIM+GPC+UNC','PRIM+GPC+QBC'])
        ax_avg_cons.get_xaxis().tick_bottom()
        ax_avg_cons.get_yaxis().tick_left()
        ax_avg_cons.set(title= 'Average Consistency (Total)')
        fig_avg_cons.savefig('avg-cons_'+str(pool_size)+'.png', bbox_inches='tight')

        fig_avg_dens = plt.figure(figsize=(20, 10))
        ax_avg_dens = fig_avg_dens.add_subplot()
        bp_avg_dens = ax_dens.boxplot(avg_dens)
        ax_avg_dens.set_xticklabels(['PRIM','PRIM+RF','PRIM+RF+UNC','PRIM+RF+QBC','PRIM+SVM+UNC','PRIM+SVM+QBC','PRIM+GPC+UNC','PRIM+GPC+QBC'])
        ax_avg_dens.get_xaxis().tick_bottom()
        ax_avg_dens.get_yaxis().tick_left()
        ax_avg_dens.set(title= 'Average Density (Total)')
        fig_avg_dens.savefig('avg-dens_'+str(pool_size)+'.png', bbox_inches='tight')

        fig_avg_acc = plt.figure(figsize=(20, 10))
        ax_avg_acc = fig_avg_acc.add_subplot()
        bp_avg_acc = ax_acc.boxplot(avg_accs)
        ax_avg_acc.set_xticklabels(['PRIM+RF','PRIM+RF+UNC','PRIM+RF+QBC','PRIM+SVM+UNC','PRIM+SVM+QBC','PRIM+GPC+UNC','PRIM+GPC+QBC'])
        ax_avg_acc.get_xaxis().tick_bottom()
        ax_avg_acc.get_yaxis().tick_left()
        ax_avg_acc.set(title= 'Average Accuracy (Total)')
        fig_avg_acc.savefig('avg-acc_'+str(pool_size)+'.png', bbox_inches='tight')

        print("-------------------------------------- - Done! - ----------------------------------")

        plt.close('all')

        avg_auc_prim = []
        avg_auc_prim_rf = []
        avg_auc_prim_rf_unc = []
        avg_auc_prim_rf_qbc = []
        avg_auc_prim_svm_qbc = []
        avg_auc_prim_svm_unc = []
        avg_auc_prim_gpc_unc = []
        avg_auc_prim_gpc_qbc = []
        avg_res_dim_prim = []
        avg_res_dim_prim_rf = []
        avg_res_dim_prim_rf_unc = []
        avg_res_dim_prim_rf_qbc = []
        avg_res_dim_prim_svm_unc = []
        avg_res_dim_prim_svm_qbc = []
        avg_res_dim_prim_gpc_unc = []
        avg_res_dim_prim_gpc_qbc = []
        avg_cons_prim = []
        avg_cons_prim_rf = []
        avg_cons_prim_rf_unc = []
        avg_cons_prim_rf_qbc = []
        avg_cons_prim_svm_qbc = []
        avg_cons_prim_svm_unc = []
        avg_cons_prim_gpc_unc = []
        avg_cons_prim_gpc_qbc = []
        avg_dens_prim = []
        avg_dens_prim_rf = []
        avg_dens_prim_rf_unc = []
        avg_dens_prim_rf_qbc = []
        avg_dens_prim_svm_qbc = []
        avg_dens_prim_svm_unc = []
        avg_dens_prim_gpc_unc = []
        avg_dens_prim_gpc_qbc = []
        avg_acc_rf = []
        avg_acc_rf_unc = []
        avg_acc_rf_qbc = []
        avg_acc_svm_qbc = []
        avg_acc_svm_unc = []
        avg_acc_gpc_unc = []
        avg_acc_gpc_qbc = []

if __name__ == '__main__':
    main()
