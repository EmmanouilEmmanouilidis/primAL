import numpy as np
import pandas as pd
import matplotlib as mpl
import sklearn
import modAL
import collections
import copy
import cvxpy
import matplotlib.pyplot as plt
from _variance_reduction import estVar
from customGPC import GPClassifier
from multiprocessing import Pool
from functools import partial
from dataset import Dataset
from modAL.expected_error import expected_error_reduction
from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner, Committee
from modAL.utils.selection import multi_argmax
from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.affine.sum import sum
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels, rbf_kernel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone

RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)

def expeced_model_change(classifier, X, X_train, y_train):
    predictions = classifier.predict(X)

    scores = []
    for i,x in enumerate(X):
        X_temp = np.append(X_train, [x])
        clf = copy.deepcopy(classifier)
        y_temp = np.append(y_train, predictions[i])
        X_temp = X_temp.reshape(-1,X.shape[1])
        clf.fit(X_temp, y_temp)
        new_predictions = clf.predict(X)

        label_change = np.abs(new_predictions - predictions)
        score = 0

        score = np.sum(label_change)
        scores.append(score)

    return scores


def variance_reduction(model, X, y_unlabeled, X_train, y_train, sigma = 100, n_jobs=1):
    label_count = np.sum(y_unlabeled) + np.sum(y_train)
    clf = copy.deepcopy(model)
    clf.fit(X_train, y_train)
    p = Pool(n_jobs)
    errors = p.map(_E, [(X_train, y_train, x, clf, label_count, sigma, model) for x in X])
    p.terminate()

    return errors


def _Phi(sigma, PI, X, epi, ex, label_count, feature_count):
    ret = estVar(sigma, PI, X, epi, ex)
    return ret


def _E(args):
    X, y, qx, clf, label_count, sigma, model = args
    print(X)
    print(y)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    query_point = sigmoid(clf.predict_real([qx]))
    feature_count = len(X[0])
    ret = 0.0
    for i in range(label_count):
        clf_ = copy.copy(model)
        clf_.train(Dataset(np.vstack((X, [qx])), np.append(y, i)))
        PI = sigmoid(clf_.predict_real(np.vstack((X, [qx]))))
        ret += query_point[-1][i] * _Phi(sigma, PI[:-1], X, PI[-1], qx, label_count, feature_count)
    return ret


def _is_arraylike(x):
    return (hasattr(x, '__len__') or
            hasattr(x, 'shape') or
            hasattr(x, '__array__'))


def nlargestarg(a, n):
    assert (_is_arraylike(a))
    assert (n > 0)
    argret = np.argsort(a)
    # ascent
    return argret[argret.size - n:]


class QUIRE:
    def __init__(self, X, y, train_idx, **kwargs):
        X = np.asarray(X)[train_idx]
        y = np.asarray(y)[train_idx]
        self._train_idx = np.asarray(train_idx)
        #print("-----------")
        #print(self._train_idx)
        #print("-----------")
        self.y = np.array(y)
        self.lmbda = kwargs.pop('lambda', 1.)
        self.kernel = kwargs.pop('kernel', 'rbf')
        self.K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma', 1.))

        self.L = np.linalg.inv(self.K + self.lmbda * np.eye(len(X)))

    def select(self, label_index, unlabel_index, **kwargs):
        if len(unlabel_index) <= 1:
            return list(unlabel_index)
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)

        # build map from value to index
        #print(unlabel_index)
        #print(label_index)
        #print(self._train_idx)
        #print("-----------")
        label_index_in_train = [np.where(self._train_idx == i)[0][0] for i in label_index]
        #print(label_index_in_train)
        #print([np.where(self._train_idx == i) for i in unlabel_index])
        unlabel_index_in_train = [np.where(self._train_idx == i)[0][0] for i in unlabel_index]
        # end

        L = self.L
        Lindex = list(label_index_in_train)
        Uindex = list(unlabel_index_in_train)
        query_index = -1
        min_eva = np.inf
        # y_labeled = np.array([label for label in self.y if label is not None])
        y_labeled = self.y[Lindex]
        det_Laa = np.linalg.det(L[np.ix_(Uindex, Uindex)])
        # efficient computation of inv(Laa)
        M3 = np.dot(self.K[np.ix_(Uindex, Lindex)],
                    np.linalg.inv(self.lmbda * np.eye(len(Lindex))))
        M2 = np.dot(M3, self.K[np.ix_(Lindex, Uindex)])
        M1 = self.lmbda * np.eye(len(Uindex)) + self.K[np.ix_(Uindex, Uindex)]
        inv_Laa = M1 - M2
        iList = list(range(len(Uindex)))
        if len(iList) == 1:
            return Uindex[0]
        for i, each_index in enumerate(Uindex):
            # go through all unlabeled instances and compute their evaluation
            # values one by one
            Uindex_r = Uindex[:]
            Uindex_r.remove(each_index)
            iList_r = iList[:]
            iList_r.remove(i)
            inv_Luu = inv_Laa[np.ix_(iList_r, iList_r)] - 1 / inv_Laa[i, i] * \
                                                          np.dot(inv_Laa[iList_r, i], inv_Laa[iList_r, i].T)
            tmp = np.dot(
                L[each_index][Lindex] -
                np.dot(
                    np.dot(
                        L[each_index][Uindex_r],
                        inv_Luu
                    ),
                    L[np.ix_(Uindex_r, Lindex)]
                ),
                y_labeled,
            )
            eva = L[each_index][each_index] - \
                  det_Laa / L[each_index][each_index] + 2 * np.abs(tmp)

            if eva < min_eva:
                query_index = each_index
                min_eva = eva
        return [self._train_idx[query_index]]


class BMDR:
    def __init__(self, X, y, beta=1000, gamma=0.1, rho=1, **kwargs):

        # K: kernel matrix
        super(BMDR, self).__init__()
        self.X = X
        self.y = y
        ul = unique_labels(self.y)

        if len(ul) == 2 and {1, -1} != set(ul):
            y_temp = np.array(copy.deepcopy(self.y))
            y_temp[y_temp == ul[0]] = 1
            y_temp[y_temp == ul[1]] = -1
            self.y = y_temp

        self._beta = beta
        self._gamma = gamma
        self._rho = rho

        # calc kernel
        self._kernel = kwargs.pop('kernel', 'rbf')
        self._K = rbf_kernel(X=self.X, Y=self.X, gamma=kwargs.pop('gamma_ker', 1.))


    def select(self, label_index, unlabel_index, batch_size=5, qp_solver='ECOS', **kwargs):

        KLL = self._K[np.ix_(label_index, label_index)]
        KLU = self._K[np.ix_(label_index, unlabel_index)]
        KUU = self._K[np.ix_(unlabel_index, unlabel_index)]

        L_len = len(label_index)
        U_len = len(unlabel_index)
        N = L_len + U_len

        # precision of ADMM
        MAX_ITER = 1000
        ABSTOL = 1e-4
        RELTOL = 1e-2

        # train a linear model in kernel form for
        tau = np.linalg.inv(KLL + self._gamma * np.eye(L_len)).dot(self.y[label_index])

        # start optimization
        last_round_selected = []
        iter_round = 0
        while 1:
            iter_round += 1
            # solve QP
            P = 0.5 * self._beta * KUU
            pred_of_unlab = tau.dot(KLU)
            a = pred_of_unlab * pred_of_unlab + 2 * np.abs(pred_of_unlab)
            q = self._beta * (
                (U_len - batch_size) / N * np.ones(L_len).dot(KLU) - (L_len + batch_size) / N * np.ones(U_len).dot(
                    KUU)) + a

            # cvx
            x = cvxpy.Variable(U_len)
            objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q.T * x)
            constraints = [0 <= x, x <= 1, sum(x) == batch_size]
            prob = cvxpy.Problem(objective, constraints)
            # The optimal objective value is returned by `prob.solve()`.
            # print(prob.is_qp())
            try:
                result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS)
            except cvxpy.error.DCPError:
                # cvx
                x = cvxpy.Variable(U_len)
                objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q.T * x)
                constraints = [0 <= x, x <= 1]
                prob = cvxpy.Problem(objective, constraints)
                # The optimal objective value is returned by `prob.solve()`.
                try:
                    result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS)
                except cvxpy.error.DCPError:
                    result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS, gp=True)

            # Sometimes the constraints can not be satisfied,
            # thus we relax the constraints to get an approximate solution.
            if not (type(result) == float and result != float('inf') and result != float('-inf')):
                # cvx
                x = cvxpy.Variable(U_len)
                objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q.T * x)
                constraints = [0 <= x, x <= 1]
                prob = cvxpy.Problem(objective, constraints)
                # The optimal objective value is returned by `prob.solve()`.
                try:
                    result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS)
                except cvxpy.error.DCPError:
                    result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS, gp=True)

            # The optimal value for x is stored in `x.value`.
            # print(x.value)
            dr_weight = np.array(x.value)
            if len(np.shape(dr_weight)) == 2:
                dr_weight = dr_weight.T[0]
            # end cvx

            # record selected indexes and judge convergence
            dr_largest = nlargestarg(dr_weight, batch_size)
            select_ind = np.asarray(unlabel_index)[dr_largest]
            if set(last_round_selected) == set(select_ind) or iter_round > 15:
                return select_ind
            else:
                last_round_selected = copy.copy(select_ind)
            # print(dr_weight[dr_largest])

            # ADMM optimization process
            delta = np.zeros(batch_size)  # dual variable in ADMM
            KLQ = self._K[np.ix_(label_index, select_ind)]
            z = tau.dot(KLQ)

            for solver_iter in range(MAX_ITER):
                # tau update
                A = KLL.dot(KLL) + self._rho / 2 * KLQ.dot(KLQ.T) + self._gamma * KLL
                r = self.y[label_index].dot(KLL) + 0.5 * delta.dot(KLQ.T) + self._rho / 2 * z.dot(KLQ.T)
                tau = np.linalg.pinv(A).dot(r)

                # z update
                zold = z
                v = (self._rho * tau.dot(KLQ) - delta) / (self._rho + 2)
                ita = 2 / (self._rho + 2)
                z_sign = np.sign(v)
                z_sign[z_sign == 0] = 1
                ztp = (np.abs(v) - ita * np.ones(len(v)))
                ztp[ztp < 0] = 0
                z = z_sign * ztp

                # delta update
                delta += self._rho * (z - tau.dot(KLQ))

                # judge convergence
                r_norm = np.linalg.norm((tau.dot(KLQ) - z))
                s_norm = np.linalg.norm(-self._rho * (z - zold))
                eps_pri = np.sqrt(batch_size) * ABSTOL + RELTOL * max(np.linalg.norm(z), np.linalg.norm(tau.dot(KLQ)))
                eps_dual = np.sqrt(batch_size) * ABSTOL + RELTOL * np.linalg.norm(delta)
                if r_norm < eps_pri and s_norm < eps_dual:
                    break


class SPAL:
    def __init__(self, X, y, mu=0.1, gamma=0.1, rho=1, lambda_init=0.1, lambda_pace=0.01, **kwargs):

        # K: kernel matrix
        super(SPAL, self).__init__()
        self.X = X
        self.y = y
        ul = unique_labels(self.y)
        if len(ul) == 2 and {1, -1} != set(ul):
            y_temp = np.array(copy.deepcopy(self.y))
            y_temp[y_temp == ul[0]] = 1
            y_temp[y_temp == ul[1]] = -1
            self.y = y_temp

        self._mu = mu
        self._gamma = gamma
        self._rho = rho
        self._lambda_init = lambda_init
        self._lambda_pace = lambda_pace
        self._lambda = lambda_init

        # calc kernel
        self._kernel = kwargs.pop('kernel', 'rbf')
        self._K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma_ker', 1.))

    def select(self, label_index, unlabel_index, batch_size=5, qp_solver='ECOS', **kwargs):
        KLL = self._K[np.ix_(label_index, label_index)]
        KLU = self._K[np.ix_(label_index, unlabel_index)]
        KUU = self._K[np.ix_(unlabel_index, unlabel_index)]

        L_len = len(label_index)
        U_len = len(unlabel_index)
        N = L_len + U_len

        # precision of ADMM
        MAX_ITER = 1000
        ABSTOL = 1e-4
        RELTOL = 1e-2

        # train a linear model in kernel form for
        theta = np.linalg.inv(KLL + self._gamma * np.eye(L_len)).dot(self.y[label_index])

        # start optimization
        dr_weight = np.ones(U_len)  # informativeness % representativeness
        es_weight = np.ones(U_len)  # easiness
        last_round_selected = []
        iter_round = 0
        while 1:
            iter_round += 1
            # solve QP
            P = 0.5 * self._mu * KUU
            pred_of_unlab = theta.dot(KLU)
            a = es_weight * (pred_of_unlab * pred_of_unlab + 2 * np.abs(pred_of_unlab))
            q = self._mu * (
                (U_len - batch_size) / N * np.ones(L_len).dot(KLU) - (L_len + batch_size) / N * np.ones(U_len).dot(
                    KUU)) + a
            # cvx
            x = cvxpy.Variable(U_len)
            objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q.T * x)
            constraints = [0 <= x, x <= 1, es_weight * x == batch_size]
            prob = cvxpy.Problem(objective, constraints)
            # The optimal objective value is returned by `prob.solve()`.
            # result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS)
            try:
                result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS)
            except cvxpy.error.DCPError:
                result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS, gp=True)
            # Sometimes the constraints can not be satisfied,
            # thus we relax the constraints to get an approximate solution.
            if not (type(result) == float and result != float('inf') and result != float('-inf')):
                # P = 0.5 * self._mu * KUU
                # pred_of_unlab = theta.dot(KLU)
                # a = es_weight * (pred_of_unlab * pred_of_unlab + 2 * np.abs(pred_of_unlab))
                # q = self._mu * (
                #     (U_len - batch_size) / N * np.ones(L_len).dot(KLU) - (L_len + batch_size) / N * np.ones(U_len).dot(
                #         KUU)) + a
                # cvx
                x = cvxpy.Variable(U_len)
                objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q.T * x)
                constraints = [0 <= x, x <= 1]
                prob = cvxpy.Problem(objective, constraints)
                # The optimal objective value is returned by `prob.solve()`.
                try:
                    result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS)
                except cvxpy.error.DCPError:
                    result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS, gp=True)

            # The optimal value for x is stored in `x.value`.
            # print(x.value)
            dr_weight = np.array(x.value)
            # print(dr_weight)
            # print(result)
            if len(np.shape(dr_weight)) == 2:
                dr_weight = dr_weight.T[0]
            # end cvx

            # update easiness weight
            worst_loss = dr_weight * (pred_of_unlab * pred_of_unlab + 2 * np.abs(pred_of_unlab))
            es_weight = np.zeros(U_len)
            es_weight_tmp = 1 - (worst_loss / self._lambda)
            update_indices = np.nonzero(worst_loss < self._lambda)[0]
            es_weight[update_indices] = es_weight_tmp[update_indices]

            # record selected indexes and judge convergence
            dr_largest = nlargestarg(dr_weight * es_weight, batch_size)
            select_ind = np.asarray(unlabel_index)[dr_largest]
            if set(last_round_selected) == set(select_ind) or iter_round > 15:
                return select_ind
            else:
                last_round_selected = copy.copy(select_ind)
                # print(dr_largest)
                # print(dr_weight[dr_largest])

            # ADMM optimization process
            # Filter less important instances for efficiency
            mix_weight = dr_weight * es_weight
            mix_weight[mix_weight < 0] = 0
            validind = np.nonzero(mix_weight > 0.001)[0]
            if len(validind) < 1:
                validind = nlargestarg(mix_weight, 1)
            vKlu = KLU[:, validind]

            delta = np.zeros(len(validind))  # dual variable in ADMM
            z = theta.dot(vKlu)  # auxiliary variable in ADMM

            # pre-computed constants in ADMM
            A = 2 * KLL.dot(KLL) + self._rho * vKlu.dot(vKlu.T) + 2 * self._gamma * KLL
            pinvA = np.linalg.pinv(A)
            rz = self._rho * vKlu
            rc = 2 * KLL.dot(self.y[label_index])
            kdenom = np.sqrt(mix_weight[validind] + self._rho / 2)
            ci = mix_weight[validind] / kdenom

            for solver_iter in range(MAX_ITER):
                # theta update
                r = rz.dot(z.T) + vKlu.dot(delta) + rc
                theta = pinvA.dot(r)

                # z update
                zold = z
                vud = self._rho * theta.dot(vKlu)
                vi = (vud - delta) / (2 * kdenom)
                ztmp = np.abs(vi) - ci
                ztmp[ztmp < 0] = 0
                ksi = np.sign(vi) * ztmp
                z = ksi / kdenom

                # delta update
                delta += self._rho * (z - theta.dot(vKlu))

                # judge convergence
                r_norm = np.linalg.norm((theta.dot(vKlu) - z))
                s_norm = np.linalg.norm(-self._rho * (z - zold))
                eps_pri = np.sqrt(len(validind)) * ABSTOL + RELTOL * max(np.linalg.norm(z),
                                                                         np.linalg.norm(theta.dot(vKlu)))
                eps_dual = np.sqrt(len(validind)) * ABSTOL + RELTOL * np.linalg.norm(delta)
                if r_norm < eps_pri and s_norm < eps_dual:
                    break


def emc_strategy(model, X, x_train, y_train):
    utility = expeced_model_change(model, X, x_train, y_train)
    query_idx = multi_argmax(np.array(utility), n_instances=1)
    return query_idx, X[query_idx]

def var_redu_strategy(model, X, y, x_train, y_train):
    print(x_train)
    print(y_train)
    utility = variance_reduction(model.estimator, X, y, x_train, y_train)
    query_idx = multi_argmax(np.invert(utility), n_instances=1)
    return query_idx, X[query_idx]

def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]

def get_label(D, p=0):
    landmark = np.zeros((2, D.shape[1]))
    landmark[1,0] += 3.0
    threshold = 1.
    L = np.min(pairwise_distances(D, landmark), axis=1) < threshold
    L = L*2-1
    L *= (-1)**np.random.binomial(1, p, D.shape[0]) # add noise
    return L

def delete_idxs(a,b):
    return list(set(a)-set(b))


def experiment_rf(features, labels, size, repetitions, nsamples):
    init = 10

    acc_unc_all = []
    f1_unc_all = []
    auc_unc_all = []
    acc_qbc_all = []
    f1_qbc_all = []
    auc_qbc_all = []
    acc_eer_all = []
    f1_eer_all = []
    auc_eer_all = []
    acc_rbm_all = []
    f1_rbm_all = []
    auc_rbm_all = []
    acc_rnd_all = []
    f1_rnd_all = []
    auc_rnd_all = []

    for i in range(repetitions):
        print("--- Started " + str(i+1) +"th iteration ---")
        init_indices = np.random.randint(low=0, high=features.shape[0], size=size)
        found = False
        while found is False:
            init_indices = np.random.randint(low=0, high=features.shape[0], size=size)
            uniqueValues, occurCount = np.unique(labels[init_indices], return_counts=True)
            if uniqueValues.shape[0]==2:
                if occurCount[0] > 10 and occurCount[1] > 10:
                    found = True

        X_raw = features[init_indices]
        y_raw = labels[init_indices]
        initsize =  X_raw.shape[0] // init

        n_labeled_examples = X_raw.shape[0]
        training_indices = np.random.randint(low=0, high=n_labeled_examples, size=20)
        found = False
        while found is False:
                training_indices = np.random.randint(low=0, high=n_labeled_examples, size=20)
                uniqueValues, occurCount = np.unique(y_raw[training_indices], return_counts=True)
                if uniqueValues.shape[0]==2:
                    if occurCount[0] == 10 and occurCount[1] == 10:
                        found = True

        X_train = X_raw[training_indices]
        y_train = y_raw[training_indices]


        #Uncertinty Sampling
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)

        rfc = RandomForestClassifier(random_state=i)

        rfc_unc = clone(rfc)
        learner = ActiveLearner(estimator=rfc_unc, X_training=X_train, y_training=y_train)
        predictions = learner.predict(X_raw)
        performance_acc_unc = [accuracy_score(y_raw, predictions)]
        #print(performance_acc_unc)
        performance_f1_unc  =[f1_score(y_raw, predictions)]
        performance_auc_unc = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd = learner.predict(X_raw)
            performance_acc_unc.append(accuracy_score(y_raw, prd))
            performance_f1_unc.append(f1_score(y_raw, prd))
            performance_auc_unc.append(roc_auc_score(y_raw, prd))

        acc_unc_all.append(performance_acc_unc)
        f1_unc_all.append(performance_f1_unc)
        auc_unc_all.append(performance_auc_unc)


        # QBC
        n_members = 2
        learner_list = list()

        for member_idx in range(n_members):
            train_idx = np.random.randint(low=0, high=n_labeled_examples, size=initsize)
            while np.unique(y_raw[train_idx]).shape[0] != 2 :
                train_idx = np.random.randint(low=0, high=n_labeled_examples, size=initsize)
            X_train_qcb = X_raw[train_idx]
            y_train_qcb = y_raw[train_idx]

            X_pool = np.delete(X_raw, train_idx, axis=0)
            y_pool = np.delete(y_raw, train_idx)

            learner = ActiveLearner(estimator=RandomForestClassifier(random_state=i), X_training=X_train_qcb, y_training=y_train_qcb)
            learner_list.append(learner)

        committee = Committee(learner_list=learner_list)

        pred  = committee.predict(X_raw)
        performance_acc_qbc = [accuracy_score(y_raw, pred)]
        performance_f1_qbc = [f1_score(y_raw, pred)]
        performance_auc_qbc  =[roc_auc_score(y_raw, pred)]

        for index in range(nsamples):
            query_idx, query_instance = committee.query(X_pool)
            committee.teach(
                X=X_pool[query_idx].reshape(1, -1),
                y=y_pool[query_idx].reshape(1, )
            )
            pred  = committee.predict(X_raw)
            performance_acc_qbc.append(accuracy_score(y_raw, pred))
            performance_f1_qbc.append(f1_score(y_raw, pred))
            performance_auc_qbc.append(roc_auc_score(y_raw, pred))
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)

        acc_qbc_all.append(performance_acc_qbc)
        f1_qbc_all.append(performance_f1_qbc)
        auc_qbc_all.append(performance_auc_qbc)


        #EER
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)
        inst_lim = np.random.randint(low=0, high=X_pool.shape[0], size=110)
        X_pool = X_pool[inst_lim]
        y_pool = y_pool[inst_lim]

        rfc_eer = clone(rfc)
        learner = ActiveLearner(estimator=rfc_eer, X_training=X_train, y_training=y_train, query_strategy=expected_error_reduction)
        predictions = learner.predict(X_raw)
        performance_acc_eer = [accuracy_score(y_raw, predictions)]
        performance_f1_eer  =[f1_score(y_raw, predictions)]
        performance_auc_eer = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd= learner.predict(X_raw)
            performance_acc_eer.append(accuracy_score(y_raw, prd))
            performance_f1_eer.append(f1_score(y_raw, prd))
            performance_auc_eer.append(roc_auc_score(y_raw, prd))

        acc_eer_all.append(performance_acc_eer)
        f1_eer_all.append(performance_f1_eer)
        auc_eer_all.append(performance_auc_eer)


        #RBMAL
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)

        rfc_rbmal = clone(rfc)

        BATCH_SIZE = 1
        preset_batch = partial(uncertainty_batch_sampling, n_instances=BATCH_SIZE)

        learner = ActiveLearner(estimator=rfc_rbmal, X_training=X_train, y_training=y_train, query_strategy=preset_batch)
        predictions = learner.predict(X_raw)
        performance_acc_rbm = [accuracy_score(y_raw, predictions)]
        performance_f1_rbm  =[f1_score(y_raw, predictions)]
        performance_auc_rbm = [roc_auc_score(y_raw, predictions)]

        N_QUERIES = nsamples // BATCH_SIZE

        for index in range(N_QUERIES):
            query_index, query_instance = learner.query(X_pool)

            X, y = X_pool[query_index], y_pool[query_index]
            learner.teach(X=X, y=y)

            X_pool = np.delete(X_pool, query_index, axis=0)
            y_pool = np.delete(y_pool, query_index)
            prd= learner.predict(X_raw)
            performance_acc_rbm.append(accuracy_score(y_raw, prd))
            performance_f1_rbm.append(f1_score(y_raw, prd))
            performance_auc_rbm.append(roc_auc_score(y_raw, prd))

        acc_rbm_all.append(performance_acc_rbm)
        f1_rbm_all.append(performance_f1_rbm)
        auc_rbm_all.append(performance_auc_rbm)


        # Random Sampling
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)

        rfc_rand = clone(rfc)
        learner = ActiveLearner(estimator=rfc_rand, X_training=X_train, y_training=y_train, query_strategy=random_sampling)
        predictions = learner.predict(X_raw)
        performance_acc_rnd = [accuracy_score(y_raw, predictions)]
        performance_f1_rnd  =[f1_score(y_raw, predictions)]
        performance_auc_rnd = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd= learner.predict(X_raw)
            performance_acc_rnd.append(accuracy_score(y_raw, prd))
            performance_f1_rnd.append(f1_score(y_raw, prd))
            performance_auc_rnd.append(roc_auc_score(y_raw, prd))

        acc_rnd_all.append(performance_acc_rnd)
        f1_rnd_all.append(performance_f1_rnd)
        auc_rnd_all.append(performance_auc_rnd)

        print("--- Done ---")

    unc_avg_acc = np.mean(np.asarray(acc_unc_all), axis = 0)
    unc_avg_f1 = np.mean(np.asarray(f1_unc_all), axis = 0)
    unc_avg_auc = np.mean(np.asarray(auc_unc_all), axis = 0)
    qbc_avg_acc = np.mean(np.asarray(acc_qbc_all), axis = 0)
    qbc_avg_f1 = np.mean(np.asarray(f1_qbc_all), axis = 0)
    qbc_avg_auc = np.mean(np.asarray(auc_qbc_all), axis = 0)
    eer_avg_acc = np.mean(np.asarray(acc_eer_all), axis = 0)
    eer_avg_f1 = np.mean(np.asarray(f1_eer_all), axis = 0)
    eer_avg_auc = np.mean(np.asarray(auc_eer_all), axis = 0)
    rbm_avg_acc = np.mean(np.asarray(acc_rbm_all), axis = 0)
    rbm_avg_f1 = np.mean(np.asarray(f1_rbm_all), axis = 0)
    rbm_avg_auc = np.mean(np.asarray(auc_rbm_all), axis = 0)
    rnd_avg_acc = np.mean(np.asarray(acc_rnd_all), axis = 0)
    rnd_avg_f1 = np.mean(np.asarray(f1_rnd_all), axis = 0)
    rnd_avg_auc = np.mean(np.asarray(auc_rnd_all), axis = 0)

    avg_acc = [unc_avg_acc, qbc_avg_acc, eer_avg_acc, rbm_avg_acc, rnd_avg_acc]
    avg_f1 = [unc_avg_f1, qbc_avg_f1, eer_avg_f1, rbm_avg_f1, rnd_avg_f1]
    avg_auc = [unc_avg_auc, qbc_avg_auc, eer_avg_auc, rbm_avg_auc, rnd_avg_auc]

    return avg_acc, avg_f1, avg_auc


def experiment_svm(features, labels, size, repetitions, nsamples):
    init = 10

    acc_unc_all = []
    f1_unc_all = []
    auc_unc_all = []
    acc_qbc_all = []
    f1_qbc_all = []
    auc_qbc_all = []
    acc_eer_all = []
    f1_eer_all = []
    auc_eer_all = []
    acc_emc_all = []
    f1_emc_all = []
    auc_emc_all = []
    acc_quire_all = []
    f1_quire_all = []
    auc_quire_all = []
    acc_rbm_all = []
    f1_rbm_all = []
    auc_rbm_all = []
    acc_bmdr_all = []
    f1_bmdr_all = []
    auc_bmdr_all = []
    acc_spal_all = []
    f1_spal_all = []
    auc_spal_all = []
    acc_rnd_all = []
    f1_rnd_all = []
    auc_rnd_all = []

    for i in range(repetitions):
        print("--- Started " + str(i+1) +"th iteration ---")
        init_indices = np.random.randint(low=0, high=features.shape[0], size=size)
        found = False
        while found is False:
            init_indices = np.random.randint(low=0, high=features.shape[0], size=size)
            uniqueValues, occurCount = np.unique(labels[init_indices], return_counts=True)
            if uniqueValues.shape[0]==2:
                if occurCount[0] > 10 and occurCount[1] > 10:
                    found = True

        X_raw_init = features[init_indices]
        y_raw = labels[init_indices]

        scaler = MinMaxScaler(copy=False)
        X_raw = scaler.fit_transform(X_raw_init)

        initsize =  X_raw.shape[0] // init
        n_labeled_examples = X_raw.shape[0]

        training_indices = np.random.randint(low=0, high=n_labeled_examples, size=20)
        found = False
        while found is False:
                training_indices = np.random.randint(low=0, high=n_labeled_examples, size=20)
                uniqueValues, occurCount = np.unique(y_raw[training_indices], return_counts=True)
                if uniqueValues.shape[0]==2:
                    if occurCount[0] == 10 and occurCount[1] == 10:
                        found = True

        X_train = X_raw[training_indices]
        y_train = y_raw[training_indices]

        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)

        svc_init = svm.SVC(kernel='rbf', probability=True, random_state=i)

        #uncertinty sampling
        svc_unc = clone(svc_init)
        learner = ActiveLearner(estimator=svc_unc, X_training=X_train, y_training=y_train)
        predictions = learner.predict(X_raw)
        performance_acc_unc = [accuracy_score(y_raw, predictions)]
        performance_f1_unc  =[f1_score(y_raw, predictions)]
        performance_auc_unc = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd= learner.predict(X_raw)
            performance_acc_unc.append(accuracy_score(y_raw, prd))
            performance_f1_unc.append(f1_score(y_raw, prd))
            performance_auc_unc.append(roc_auc_score(y_raw, prd))

        acc_unc_all.append(performance_acc_unc)
        f1_unc_all.append(performance_f1_unc)
        auc_unc_all.append(performance_auc_unc)


        # QBC
        n_members = 2
        learner_list = list()

        for member_idx in range(n_members):
            train_idx = np.random.randint(low=0, high=n_labeled_examples, size=initsize)
            while np.unique(y_raw[train_idx]).shape[0] != 2 :
                train_idx = np.random.randint(low=0, high=n_labeled_examples, size=initsize)
            X_train_qbc = X_raw[train_idx]
            y_train_qbc = y_raw[train_idx]
            X_pool = np.delete(X_raw, train_idx, axis=0)
            y_pool = np.delete(y_raw, train_idx)

            learner = ActiveLearner(estimator=svm.SVC(kernel='rbf', probability=True, random_state=i), X_training=X_train_qbc, y_training=y_train_qbc)
            learner_list.append(learner)

        committee = Committee(learner_list=learner_list)

        pred  = committee.predict(X_raw)
        performance_acc_qbc = [accuracy_score(y_raw, pred)]
        performance_f1_qbc = [f1_score(y_raw, pred)]
        performance_auc_qbc  =[roc_auc_score(y_raw, pred)]

        for index in range(nsamples):
            query_idx, query_instance = committee.query(X_pool)
            committee.teach(
                X=X_pool[query_idx].reshape(1, -1),
                y=y_pool[query_idx].reshape(1, )
            )
            pred  = committee.predict(X_raw)
            performance_acc_qbc.append(accuracy_score(y_raw, pred))
            performance_f1_qbc.append(f1_score(y_raw, pred))
            performance_auc_qbc.append(roc_auc_score(y_raw, pred))
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)

        acc_qbc_all.append(performance_acc_qbc)
        f1_qbc_all.append(performance_f1_qbc)
        auc_qbc_all.append(performance_auc_qbc)


        #EER
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)
        inst_lim = np.random.randint(low=0, high=X_pool.shape[0], size=110)
        X_pool = X_pool[inst_lim]
        y_pool = y_pool[inst_lim]

        svc_eer = clone(svc_init)
        learner = ActiveLearner(estimator=svc_eer, X_training=X_train, y_training=y_train, query_strategy=expected_error_reduction)
        predictions = learner.predict(X_raw)
        performance_acc_eer = [accuracy_score(y_raw, predictions)]
        performance_f1_eer  = [f1_score(y_raw, predictions)]
        performance_auc_eer = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd = learner.predict(X_raw)
            performance_acc_eer.append(accuracy_score(y_raw, prd))
            performance_f1_eer.append(f1_score(y_raw, prd))
            performance_auc_eer.append(roc_auc_score(y_raw, prd))

        acc_eer_all.append(performance_acc_eer)
        f1_eer_all.append(performance_f1_eer)
        auc_eer_all.append(performance_auc_eer)


        #EMC
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)
        inst_lim = np.random.randint(low=0, high=X_pool.shape[0], size=110)
        X_pool = X_pool[inst_lim]
        y_pool = y_pool[inst_lim]

        svc_emc = clone(svc_init)
        learner = ActiveLearner(estimator=svc_emc, X_training=X_train, y_training=y_train, query_strategy=emc_strategy)
        predictions = learner.predict(X_raw)
        performance_acc_emc = [accuracy_score(y_raw, predictions)]
        performance_f1_emc  =[f1_score(y_raw, predictions)]
        performance_auc_emc = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool, X_train, y_train)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd= learner.predict(X_raw)
            performance_acc_emc.append(accuracy_score(y_raw, prd))
            performance_f1_emc.append(f1_score(y_raw, prd))
            performance_auc_emc.append(roc_auc_score(y_raw, prd))

        acc_emc_all.append(performance_acc_emc)
        f1_emc_all.append(performance_f1_emc)
        auc_emc_all.append(performance_auc_emc)


        #QUIRE
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)
        trn_idxs = training_indices.tolist()
        raw_idxs = list(range(0,X_raw.shape[0]))
        pool_idxs = delete_idxs(raw_idxs, trn_idxs)
        quire = QUIRE(X_raw, y_raw, raw_idxs)

        svc_quire = clone(svc_init)
        learner  = ActiveLearner(estimator=svc_quire, X_training=X_train, y_training=y_train, query_strategy=quire.select)
        predictions = learner.predict(X_raw)
        performance_acc_quire = [accuracy_score(y_raw, predictions)]
        performance_f1_quire  =[f1_score(y_raw, predictions)]
        performance_auc_quire = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index = quire.select(trn_idxs, pool_idxs)
            X, y = X_raw[query_index], y_raw[query_index]
            learner.teach(X=X, y=y)
            pool_idxs = delete_idxs(pool_idxs, query_index)
            trn_idxs.extend(query_index)
            prd= learner.predict(X_raw)
            performance_acc_quire.append(accuracy_score(y_raw, prd))
            performance_f1_quire.append(f1_score(y_raw, prd))
            performance_auc_quire.append(roc_auc_score(y_raw, prd))

        acc_quire_all.append(performance_acc_quire)
        f1_quire_all.append(performance_f1_quire)
        auc_quire_all.append(performance_auc_quire)


        #RBMAL
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)

        BATCH_SIZE = 1
        preset_batch = partial(uncertainty_batch_sampling, n_instances=BATCH_SIZE)

        svc_rbmal = clone(svc_init)
        learner = ActiveLearner(estimator=svc_rbmal, X_training=X_train, y_training=y_train, query_strategy=preset_batch)
        predictions = learner.predict(X_raw)
        performance_acc_rbm = [accuracy_score(y_raw, predictions)]
        performance_f1_rbm  =[f1_score(y_raw, predictions)]
        performance_auc_rbm = [roc_auc_score(y_raw, predictions)]

        N_QUERIES = nsamples // BATCH_SIZE

        for index in range(N_QUERIES):
            query_index, query_instance = learner.query(X_pool)
            X, y = X_pool[query_index], y_pool[query_index]
            learner.teach(X=X, y=y)
            X_pool = np.delete(X_pool, query_index, axis=0)
            y_pool = np.delete(y_pool, query_index)
            prd = learner.predict(X_raw)
            performance_acc_rbm.append(accuracy_score(y_raw, prd))
            performance_f1_rbm.append(f1_score(y_raw, prd))
            performance_auc_rbm.append(roc_auc_score(y_raw, prd))

        acc_rbm_all.append(performance_acc_rbm)
        f1_rbm_all.append(performance_f1_rbm)
        auc_rbm_all.append(performance_auc_rbm)

        #BMDR
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)

        BATCH_SIZE = 1
        bmdr = BMDR(X_raw, y_raw)

        svc_bmdr = clone(svc_init)
        learner  = ActiveLearner(estimator=svc_bmdr, X_training=X_train, y_training=y_train, query_strategy=bmdr.select)
        predictions = learner.predict(X_raw)
        performance_acc_bmdr = [accuracy_score(y_raw, predictions)]
        performance_f1_bmdr  =[f1_score(y_raw, predictions)]
        performance_auc_bmdr = [roc_auc_score(y_raw, predictions)]

        N_QUERIES = nsamples // BATCH_SIZE
        trn_idxs = training_indices.tolist()
        raw_idxs = list(range(0,X_raw.shape[0]))
        pool_idxs = delete_idxs(raw_idxs, trn_idxs)
        for index in range(N_QUERIES):
            query_index = bmdr.select(trn_idxs, pool_idxs, BATCH_SIZE)
            X, y = X_raw[query_index], y_raw[query_index]
            learner.teach(X=X, y=y)
            pool_idxs = delete_idxs(pool_idxs, query_index)
            trn_idxs.extend(query_index.tolist())
            prd = learner.predict(X_raw)
            performance_acc_bmdr.append(accuracy_score(y_raw, prd))
            performance_f1_bmdr.append(f1_score(y_raw, prd))
            performance_auc_bmdr.append(roc_auc_score(y_raw, prd))

        acc_bmdr_all.append(performance_acc_bmdr)
        f1_bmdr_all.append(performance_f1_bmdr)
        auc_bmdr_all.append(performance_auc_bmdr)


        #SP-AL
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)

        BATCH_SIZE = 1
        spal = SPAL(X_raw, y_raw)

        svc_spal = clone(svc_init)
        learner  = ActiveLearner(estimator=svc_spal, X_training=X_train, y_training=y_train, query_strategy=spal.select)
        predictions = learner.predict(X_raw)
        performance_acc_spal = [accuracy_score(y_raw, predictions)]
        performance_f1_spal  =[f1_score(y_raw, predictions)]
        performance_auc_spal = [roc_auc_score(y_raw, predictions)]

        N_QUERIES = nsamples // BATCH_SIZE
        trn_idxs = training_indices.tolist()
        raw_idxs = list(range(0,X_raw.shape[0]))
        pool_idxs = delete_idxs(raw_idxs, trn_idxs)
        for index in range(N_QUERIES):
            query_index = spal.select(trn_idxs, pool_idxs, BATCH_SIZE)
            X, y = X_raw[query_index], y_raw[query_index]
            learner.teach(X=X, y=y)
            pool_idxs = delete_idxs(pool_idxs, query_index)
            trn_idxs.extend(query_index.tolist())
            prd = learner.predict(X_raw)
            performance_acc_spal.append(accuracy_score(y_raw, prd))
            performance_f1_spal.append(f1_score(y_raw, prd))
            performance_auc_spal.append(roc_auc_score(y_raw, prd))

        acc_spal_all.append(performance_acc_spal)
        f1_spal_all.append(performance_f1_spal)
        auc_spal_all.append(performance_auc_spal)


        # Random Sampling
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)

        svc_rnd = clone(svc_init)
        learner = ActiveLearner(estimator=svc_rnd, X_training=X_train, y_training=y_train, query_strategy=random_sampling)
        predictions = learner.predict(X_raw)
        performance_acc_rnd = [accuracy_score(y_raw, predictions)]
        performance_f1_rnd  =[f1_score(y_raw, predictions)]
        performance_auc_rnd = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd= learner.predict(X_raw)
            performance_acc_rnd.append(accuracy_score(y_raw, prd))
            performance_f1_rnd.append(f1_score(y_raw, prd))
            performance_auc_rnd.append(roc_auc_score(y_raw, prd))

        acc_rnd_all.append(performance_acc_rnd)
        f1_rnd_all.append(performance_f1_rnd)
        auc_rnd_all.append(performance_auc_rnd)

        print("--- Done ---")

    unc_avg_acc = np.mean(np.asarray(acc_unc_all), axis = 0)
    unc_avg_f1 = np.mean(np.asarray(f1_unc_all), axis = 0)
    unc_avg_auc = np.mean(np.asarray(auc_unc_all), axis = 0)
    qbc_avg_acc = np.mean(np.asarray(acc_qbc_all), axis = 0)
    qbc_avg_f1 = np.mean(np.asarray(f1_qbc_all), axis = 0)
    qbc_avg_auc = np.mean(np.asarray(auc_qbc_all), axis = 0)
    eer_avg_acc = np.mean(np.asarray(acc_eer_all), axis = 0)
    eer_avg_f1 = np.mean(np.asarray(f1_eer_all), axis = 0)
    eer_avg_auc = np.mean(np.asarray(auc_eer_all), axis = 0)
    emc_avg_acc = np.mean(np.asarray(acc_emc_all), axis = 0)
    emc_avg_f1 = np.mean(np.asarray(f1_emc_all), axis = 0)
    emc_avg_auc = np.mean(np.asarray(auc_emc_all), axis = 0)
    rbm_avg_acc = np.mean(np.asarray(acc_rbm_all), axis = 0)
    rbm_avg_f1 = np.mean(np.asarray(f1_rbm_all), axis = 0)
    rbm_avg_auc = np.mean(np.asarray(auc_rbm_all), axis = 0)
    bmdr_avg_acc = np.mean(np.asarray(acc_bmdr_all), axis = 0)
    bmdr_avg_f1 = np.mean(np.asarray(f1_bmdr_all), axis = 0)
    bmdr_avg_auc = np.mean(np.asarray(auc_bmdr_all), axis = 0)
    quire_avg_acc = np.mean(np.asarray(acc_quire_all), axis = 0)
    quire_avg_f1 = np.mean(np.asarray(f1_quire_all), axis = 0)
    quire_avg_auc = np.mean(np.asarray(auc_quire_all), axis = 0)
    spal_avg_acc = np.mean(np.asarray(acc_spal_all), axis = 0)
    spal_avg_f1 = np.mean(np.asarray(f1_spal_all), axis = 0)
    spal_avg_auc = np.mean(np.asarray(auc_spal_all), axis = 0)
    rnd_avg_acc = np.mean(np.asarray(acc_rnd_all), axis = 0)
    rnd_avg_f1 = np.mean(np.asarray(f1_rnd_all), axis = 0)
    rnd_avg_auc = np.mean(np.asarray(auc_rnd_all), axis = 0)

    avg_acc = [unc_avg_acc, qbc_avg_acc, eer_avg_acc, emc_avg_acc, rbm_avg_acc, bmdr_avg_acc, quire_avg_acc, spal_avg_acc, rnd_avg_acc]
    avg_f1 = [unc_avg_f1, qbc_avg_f1, eer_avg_f1, emc_avg_f1, rbm_avg_f1, bmdr_avg_f1, quire_avg_f1, spal_avg_f1, rnd_avg_f1]
    avg_auc = [unc_avg_auc, qbc_avg_auc, eer_avg_auc, emc_avg_auc, rbm_avg_auc, bmdr_avg_auc, quire_avg_auc, spal_avg_auc, rnd_avg_auc]

    return avg_acc, avg_f1, avg_auc


def experiment_gpc(features, labels, size, repetitions, nsamples):
    init = 10
    kernel = 1.0 * RBF(1.0)

    acc_unc_all = []
    f1_unc_all = []
    auc_unc_all = []
    acc_qbc_all = []
    f1_qbc_all = []
    auc_qbc_all = []
    acc_eer_all = []
    f1_eer_all = []
    auc_eer_all = []
    acc_emc_all = []
    f1_emc_all = []
    auc_emc_all = []
    acc_varredu_all = []
    f1_varredu_all = []
    auc_varredu_all = []
    acc_quire_all = []
    f1_quire_all = []
    auc_quire_all = []
    acc_rbm_all = []
    f1_rbm_all = []
    auc_rbm_all = []
    acc_bmdr_all = []
    f1_bmdr_all = []
    auc_bmdr_all = []
    acc_spal_all = []
    f1_spal_all = []
    auc_spal_all = []
    acc_rnd_all = []
    f1_rnd_all = []
    auc_rnd_all = []

    for i in range(repetitions):
        print("--- Started " + str(i+1) +"th iteration")
        init_indices = np.random.randint(low=0, high=features.shape[0], size=size)
        found = False
        while found is False:
            init_indices = np.random.randint(low=0, high=features.shape[0], size=size)
            uniqueValues, occurCount = np.unique(labels[init_indices], return_counts=True)
            if uniqueValues.shape[0]==2:
                if occurCount[0] > 10 and occurCount[1] > 10:
                    found = True

        X_raw = features[init_indices]
        y_raw = labels[init_indices]
        initsize =  X_raw.shape[0] // init

        n_labeled_examples = X_raw.shape[0]
        training_indices = np.random.randint(low=0, high=n_labeled_examples, size=20)
        found = False
        while found is False:
                training_indices = np.random.randint(low=0, high=n_labeled_examples, size=20)
                uniqueValues, occurCount = np.unique(y_raw[training_indices], return_counts=True)
                if uniqueValues.shape[0]==2:
                    if occurCount[0] == 10 and occurCount[1] == 10:
                        found = True

        X_train = X_raw[training_indices]
        y_train = y_raw[training_indices]
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)

        gpc_init = GaussianProcessClassifier(kernel=kernel, random_state=i)


        #uncertinty sampling
        gpc_unc = clone(gpc_init)
        learner = ActiveLearner(estimator=gpc_unc, X_training=X_train, y_training=y_train)
        predictions = learner.predict(X_raw)
        performance_acc_unc = [accuracy_score(y_raw, predictions)]
        performance_f1_unc  =[f1_score(y_raw, predictions)]
        performance_auc_unc = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd = learner.predict(X_raw)
            performance_acc_unc.append(accuracy_score(y_raw, prd))
            performance_f1_unc.append(f1_score(y_raw, prd))
            performance_auc_unc.append(roc_auc_score(y_raw, prd))

        acc_unc_all.append(performance_acc_unc)
        f1_unc_all.append(performance_f1_unc)
        auc_unc_all.append(performance_auc_unc)


        # QBC
        n_members = 2
        learner_list = list()

        for member_idx in range(n_members):
            train_idx = np.random.randint(low=0, high=n_labeled_examples, size=initsize)
            while np.unique(y_raw[train_idx]).shape[0] != 2 :
                train_idx = np.random.randint(low=0, high=n_labeled_examples, size=initsize)

            X_train_qbc = X_raw[train_idx]
            y_train_qbc = y_raw[train_idx]
            X_pool = np.delete(X_raw, train_idx, axis=0)
            y_pool = np.delete(y_raw, train_idx)
            learner = ActiveLearner(estimator=GaussianProcessClassifier(kernel=kernel, random_state=i), X_training=X_train_qbc, y_training=y_train_qbc)
            learner_list.append(learner)

        committee = Committee(learner_list=learner_list)
        pred  = committee.predict(X_raw)
        performance_acc_qbc = [accuracy_score(y_raw, pred)]
        performance_f1_qbc = [f1_score(y_raw, pred)]
        performance_auc_qbc  =[roc_auc_score(y_raw, pred)]

        for index in range(nsamples):
            query_idx, query_instance = committee.query(X_pool)
            committee.teach(
                X=X_pool[query_idx].reshape(1, -1),
                y=y_pool[query_idx].reshape(1, )
            )
            pred  = committee.predict(X_raw)
            performance_acc_qbc.append(accuracy_score(y_raw, pred))
            performance_f1_qbc.append(f1_score(y_raw, pred))
            performance_auc_qbc.append(roc_auc_score(y_raw, pred))
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)

        acc_qbc_all.append(performance_acc_qbc)
        f1_qbc_all.append(performance_f1_qbc)
        auc_qbc_all.append(performance_auc_qbc)


        #EER
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)
        inst_lim = np.random.randint(low=0, high=X_pool.shape[0], size=110)
        X_pool = X_pool[inst_lim]
        y_pool = y_pool[inst_lim]
        gpc_eer = clone(gpc_init)
        learner = ActiveLearner(estimator=gpc_eer, X_training=X_train, y_training=y_train, query_strategy=expected_error_reduction)
        predictions = learner.predict(X_raw)
        performance_acc_eer = [accuracy_score(y_raw, predictions)]
        performance_f1_eer  =[f1_score(y_raw, predictions)]
        performance_auc_eer = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd= learner.predict(X_raw)
            performance_acc_eer.append(accuracy_score(y_raw, prd))
            performance_f1_eer.append(f1_score(y_raw, prd))
            performance_auc_eer.append(roc_auc_score(y_raw, prd))

        acc_eer_all.append(performance_acc_eer)
        f1_eer_all.append(performance_f1_eer)
        auc_eer_all.append(performance_auc_eer)


        #EMC
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)
        inst_lim = np.random.randint(low=0, high=X_pool.shape[0], size=110)
        X_pool = X_pool[inst_lim]
        y_pool = y_pool[inst_lim]
        gpc_emc = clone(gpc_init)
        learner = ActiveLearner(estimator=gpc_emc, X_training=X_train, y_training=y_train, query_strategy=emc_strategy)
        predictions = learner.predict(X_raw)
        performance_acc_emc = [accuracy_score(y_raw, predictions)]
        performance_f1_emc  =[f1_score(y_raw, predictions)]
        performance_auc_emc = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool, X_train, y_train)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd= learner.predict(X_raw)
            performance_acc_emc.append(accuracy_score(y_raw, prd))
            performance_f1_emc.append(f1_score(y_raw, prd))
            performance_auc_emc.append(roc_auc_score(y_raw, prd))

        acc_emc_all.append(performance_acc_emc)
        f1_emc_all.append(performance_f1_emc)
        auc_emc_all.append(performance_auc_emc)

        #Variance Reduction
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)
        inst_lim = np.random.randint(low=0, high=X_pool.shape[0], size=110)
        X_pool = X_pool[inst_lim]
        y_pool = y_pool[inst_lim]
        gpc_vr = clone(gpc_init)
        learner = ActiveLearner(estimator=GaussianProcessClassifier(kernel=kernel), query_strategy=var_redu_strategy, X_training=X_train, y_training =y_train)
        predictions = learner.predict(X_raw)
        performance_acc_vr = [accuracy_score(y_raw, predictions)]
        performance_f1_vr  =[f1_score(y_raw, predictions)]
        performance_auc_vr = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool, y_pool, x_train, y_train)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd= learner.predict(X_raw)
            performance_acc_vr.append(accuracy_score(y_raw, prd))
            performance_f1_vr.append(f1_score(y_raw, prd))
            performance_auc_vr.append(roc_auc_score(y_raw, prd))

        acc_varredu_all.append(performance_acc_vr)
        f1_varredu_all.append(performance_f1_vr)
        auc_varredu_all.append(performance_auc_vr)


        #QUIRE
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)
        trn_idxs = training_indices.tolist()
        raw_idxs = list(range(0,X_raw.shape[0]))
        pool_idxs = delete_idxs(raw_idxs, trn_idxs)
        quire = QUIRE(X_raw, y_raw, raw_idxs)
        gpc_quire = clone(gpc_init)
        learner  = ActiveLearner(estimator=gpc_quire, X_training=X_train, y_training=y_train, query_strategy=quire.select)
        predictions = learner.predict(X_raw)
        performance_acc_quire = [accuracy_score(y_raw, predictions)]
        performance_f1_quire  =[f1_score(y_raw, predictions)]
        performance_auc_quire = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index = quire.select(trn_idxs, pool_idxs)

            X, y = X_raw[query_index], y_raw[query_index]
            learner.teach(X=X, y=y)

            pool_idxs = delete_idxs(pool_idxs, query_index)
            trn_idxs.extend(query_index)
            prd= learner.predict(X_raw)
            performance_acc_quire.append(accuracy_score(y_raw, prd))
            performance_f1_quire.append(f1_score(y_raw, prd))
            performance_auc_quire.append(roc_auc_score(y_raw, prd))

        acc_quire_all.append(performance_acc_quire)
        f1_quire_all.append(performance_f1_quire)
        auc_quire_all.append(performance_auc_quire)


        #RBMAL
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)

        BATCH_SIZE = 1
        preset_batch = partial(uncertainty_batch_sampling, n_instances=BATCH_SIZE)
        gpc_rbmal = clone(gpc_init)
        learner = ActiveLearner(estimator=gpc_rbmal, X_training=X_train, y_training=y_train, query_strategy=preset_batch)
        predictions = learner.predict(X_raw)
        performance_acc_rbm = [accuracy_score(y_raw, predictions)]
        performance_f1_rbm  =[f1_score(y_raw, predictions)]
        performance_auc_rbm = [roc_auc_score(y_raw, predictions)]

        N_QUERIES = nsamples // BATCH_SIZE

        for index in range(N_QUERIES):
            query_index, query_instance = learner.query(X_pool)

            X, y = X_pool[query_index], y_pool[query_index]
            learner.teach(X=X, y=y)

            X_pool = np.delete(X_pool, query_index, axis=0)
            y_pool = np.delete(y_pool, query_index)
            prd = learner.predict(X_raw)
            performance_acc_rbm.append(accuracy_score(y_raw, prd))
            performance_f1_rbm.append(f1_score(y_raw, prd))
            performance_auc_rbm.append(roc_auc_score(y_raw, prd))

        acc_rbm_all.append(performance_acc_rbm)
        f1_rbm_all.append(performance_f1_rbm)
        auc_rbm_all.append(performance_auc_rbm)

        #BMDR
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)
        BATCH_SIZE = 1
        bmdr = BMDR(X_raw, y_raw)
        gpc_bmdr = clone(gpc_init)
        learner  = ActiveLearner(estimator=gpc_bmdr, X_training=X_train, y_training=y_train, query_strategy=bmdr.select)
        predictions = learner.predict(X_raw)
        performance_acc_bmdr = [accuracy_score(y_raw, predictions)]
        performance_f1_bmdr  =[f1_score(y_raw, predictions)]
        performance_auc_bmdr = [roc_auc_score(y_raw, predictions)]

        N_QUERIES = nsamples // BATCH_SIZE
        trn_idxs = training_indices.tolist()
        raw_idxs = list(range(0,X_raw.shape[0]))
        pool_idxs = delete_idxs(raw_idxs, trn_idxs)
        for index in range(N_QUERIES):
            query_index = bmdr.select(trn_idxs, pool_idxs, BATCH_SIZE)

            X, y = X_raw[query_index], y_raw[query_index]
            learner.teach(X=X, y=y)

            pool_idxs = delete_idxs(pool_idxs, query_index)
            trn_idxs.extend(query_index.tolist())
            prd= learner.predict(X_raw)
            performance_acc_bmdr.append(accuracy_score(y_raw, prd))
            performance_f1_bmdr.append(f1_score(y_raw, prd))
            performance_auc_bmdr.append(roc_auc_score(y_raw, prd))

        acc_bmdr_all.append(performance_acc_bmdr)
        f1_bmdr_all.append(performance_f1_bmdr)
        auc_bmdr_all.append(performance_auc_bmdr)


        #SP-AL
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)
        BATCH_SIZE = 1
        spal = SPAL(X_raw, y_raw)
        gpc_spal = clone(gpc_init)
        learner  = ActiveLearner(estimator=gpc_spal, X_training=X_train, y_training=y_train, query_strategy=spal.select)
        predictions = learner.predict(X_raw)
        performance_acc_spal = [accuracy_score(y_raw, predictions)]
        performance_f1_spal  =[f1_score(y_raw, predictions)]
        performance_auc_spal = [roc_auc_score(y_raw, predictions)]

        N_QUERIES = nsamples // BATCH_SIZE
        trn_idxs = training_indices.tolist()
        raw_idxs = list(range(0,X_raw.shape[0]))
        pool_idxs = delete_idxs(raw_idxs, trn_idxs)
        for index in range(N_QUERIES):
            query_index = spal.select(trn_idxs, pool_idxs, BATCH_SIZE)

            X, y = X_raw[query_index], y_raw[query_index]
            learner.teach(X=X, y=y)

            pool_idxs = delete_idxs(pool_idxs, query_index)
            trn_idxs.extend(query_index.tolist())
            prd= learner.predict(X_raw)
            performance_acc_spal.append(accuracy_score(y_raw, prd))
            performance_f1_spal.append(f1_score(y_raw, prd))
            performance_auc_spal.append(roc_auc_score(y_raw, prd))

        acc_spal_all.append(performance_acc_spal)
        f1_spal_all.append(performance_f1_spal)
        auc_spal_all.append(performance_auc_spal)


        # Random Sampling
        X_pool = np.delete(X_raw, training_indices, axis=0)
        y_pool = np.delete(y_raw, training_indices, axis=0)
        gpc_rnd = clone(gpc_init)
        learner = ActiveLearner(estimator=gpc_rnd,X_training=X_train, y_training=y_train, query_strategy=random_sampling)
        predictions = learner.predict(X_raw)
        performance_acc_rnd = [accuracy_score(y_raw, predictions)]
        performance_f1_rnd  =[f1_score(y_raw, predictions)]
        performance_auc_rnd = [roc_auc_score(y_raw, predictions)]

        for index in range(nsamples):
            query_index, query_instance = learner.query(X_pool)
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            prd= learner.predict(X_raw)
            performance_acc_rnd.append(accuracy_score(y_raw, prd))
            performance_f1_rnd.append(f1_score(y_raw, prd))
            performance_auc_rnd.append(roc_auc_score(y_raw, prd))

        acc_rnd_all.append(performance_acc_rnd)
        f1_rnd_all.append(performance_f1_rnd)
        auc_rnd_all.append(performance_auc_rnd)

        print("--- Done ---")

    unc_avg_acc = np.mean(np.asarray(acc_unc_all), axis = 0)
    unc_avg_f1 = np.mean(np.asarray(f1_unc_all), axis = 0)
    unc_avg_auc = np.mean(np.asarray(auc_unc_all), axis = 0)
    qbc_avg_acc = np.mean(np.asarray(acc_qbc_all), axis = 0)
    qbc_avg_f1 = np.mean(np.asarray(f1_qbc_all), axis = 0)
    qbc_avg_auc = np.mean(np.asarray(auc_qbc_all), axis = 0)
    eer_avg_acc = np.mean(np.asarray(acc_eer_all), axis = 0)
    eer_avg_f1 = np.mean(np.asarray(f1_eer_all), axis = 0)
    eer_avg_auc = np.mean(np.asarray(auc_eer_all), axis = 0)
    emc_avg_acc = np.mean(np.asarray(acc_emc_all), axis = 0)
    emc_avg_f1 = np.mean(np.asarray(f1_emc_all), axis = 0)
    emc_avg_auc = np.mean(np.asarray(auc_emc_all), axis = 0)
    vr_avg_acc = np.mean(np.asarray(acc_varredu_all), axis = 0)
    vr_avg_f1 = np.mean(np.asarray(f1_varredu_all), axis = 0)
    vr_avg_auc = np.mean(np.asarray(auc_varredu_all), axis = 0)
    rbm_avg_acc = np.mean(np.asarray(acc_rbm_all), axis = 0)
    rbm_avg_f1 = np.mean(np.asarray(f1_rbm_all), axis = 0)
    rbm_avg_auc = np.mean(np.asarray(auc_rbm_all), axis = 0)
    bmdr_avg_acc = np.mean(np.asarray(acc_bmdr_all), axis = 0)
    bmdr_avg_f1 = np.mean(np.asarray(f1_bmdr_all), axis = 0)
    bmdr_avg_auc = np.mean(np.asarray(auc_bmdr_all), axis = 0)
    quire_avg_acc = np.mean(np.asarray(acc_quire_all), axis = 0)
    quire_avg_f1 = np.mean(np.asarray(f1_quire_all), axis = 0)
    quire_avg_auc = np.mean(np.asarray(auc_quire_all), axis = 0)
    spal_avg_acc = np.mean(np.asarray(acc_spal_all), axis = 0)
    spal_avg_f1 = np.mean(np.asarray(f1_spal_all), axis = 0)
    spal_avg_auc = np.mean(np.asarray(auc_spal_all), axis = 0)
    rnd_avg_acc = np.mean(np.asarray(acc_rnd_all), axis = 0)
    rnd_avg_f1 = np.mean(np.asarray(f1_rnd_all), axis = 0)
    rnd_avg_auc = np.mean(np.asarray(auc_rnd_all), axis = 0)

    avg_acc = [unc_avg_acc, qbc_avg_acc, eer_avg_acc, emc_avg_acc, vr_avg_acc, rbm_avg_acc, bmdr_avg_acc, quire_avg_acc, spal_avg_acc, rnd_avg_acc]
    avg_f1 = [unc_avg_f1, qbc_avg_f1, eer_avg_f1, emc_avg_f1, vr_avg_f1, rbm_avg_f1, bmdr_avg_f1, quire_avg_f1, spal_avg_f1, rnd_avg_f1]
    avg_auc = [unc_avg_auc, qbc_avg_auc, eer_avg_auc, emc_avg_auc, vr_avg_auc, rbm_avg_auc, quire_avg_auc, quire_avg_auc, spal_avg_auc, rnd_avg_auc]

    return avg_acc, avg_f1, avg_auc


def main():
    dsgc = pd.read_csv("data/sdre.csv")
    morris = pd.read_csv("data/morris.csv")
    willetal06 = pd.read_csv('data/willetal06.csv')

    dp_list = [dsgc,morris,willetal06]
    sizes = [1000]
    repetitions = 50
    samples = 100


    for dataset in dp_list:
        X_raw_full = dataset.iloc[:, 1:-1].values
        y_raw_full = dataset.iloc[:, -1].values
        for size in sizes:

            print("----- Random Forrest Experiment with " + str(size) + " instances and " + str(repetitions) + " repetitions: -----")
            rf_acc, rf_f1, rf_auc = experiment_rf(X_raw_full, y_raw_full, size, repetitions, samples)

            print("-----------------------------RESULTS (RF "+ str(size) + ") -----------------------------")
            print("-----------------------------Accuracy----------------------------")
            print("Unceratinty : " + str(rf_acc[0][-1]))
            print("QBC : " + str(rf_acc[1][-1]))
            print("EER : " + str(rf_acc[2][-1]))
            print("RBMAL: " + str(rf_acc[3][-1]))
            print("Random : " + str(rf_acc[4][-1]))
            print("-----------------------------F1----------------------------------")
            print("Unceratinty : " + str(rf_f1[0][-1]))
            print("QBC : " + str(rf_f1[1][-1]))
            print("EER : " + str(rf_f1[2][-1]))
            print("RBMAL: " + str(rf_f1[3][-1]))
            print("Random : " + str(rf_f1[4][-1]))
            print("-----------------------------ROC_AUC-----------------------------")
            print("Unceratinty : " + str(rf_acc[0][-1]))
            print("QBC : " + str(rf_auc[1][-1]))
            print("EER : " + str(rf_auc[2][-1]))
            print("RBMAL: " + str(rf_auc[3][-1]))
            print("Random : " + str(rf_auc[4][-1]))
            print("-----------------------------------------------------------------------------------------")

            fig_rf, ax_rf = plt.subplots(nrows=3, ncols=1, figsize=(8.5, 6), dpi=130)
            ax_rf[0].plot(rf_acc[0], label="Uncertainty")
            ax_rf[0].plot(rf_acc[1], label="QBC")
            ax_rf[0].plot(rf_acc[2], label="EER")
            ax_rf[0].plot(rf_acc[3], label="RBMAL")
            ax_rf[0].plot(rf_acc[4], label="Random")
            ax_rf[1].plot(rf_f1[0], label="Uncertainty")
            ax_rf[1].plot(rf_f1[1], label="QBC")
            ax_rf[1].plot(rf_f1[2], label="EER")
            ax_rf[1].plot(rf_f1[3], label="RBMAL")
            ax_rf[1].plot(rf_f1[4], label="Random")
            ax_rf[2].plot(rf_auc[0], label="Uncertainty")
            ax_rf[2].plot(rf_auc[1], label="QBC")
            ax_rf[2].plot(rf_auc[2], label="EER")
            ax_rf[2].plot(rf_auc[3], label="RBMAL")
            ax_rf[2].plot(rf_auc[4], label="Random")
            ax_rf[0].grid(True)
            ax_rf[1].grid(True)
            ax_rf[2].grid(True)
            fig_rf.suptitle("Random Forrest |d| = " + str(size), fontsize = 16)
            ax_rf[0].set_ylabel("Accuracy")
            ax_rf[1].set_ylabel("F1")
            ax_rf[2].set_ylabel("ROC_AUC")
            plt.legend()
            plt.savefig("rf_"+str(size)+"_dsgc.png")


            print("----- SVM Experiment with " + str(size) + " instances and " + str(repetitions) + " repetitions: -----")
            svm_acc, svm_f1, svm_auc = experiment_svm(X_raw_full, y_raw_full, 250, repetitions, samples)

            print("-----------------------------RESULTS (SVM "+ str(size) + ") -----------------------------")
            print("-----------------------------Accuracy----------------------------")
            print("Unceratinty : " + str(svm_acc[0][-1]))
            print("QBC : " + str(svm_acc[1][-1]))
            print("EER : " + str(svm_acc[2][-1]))
            print("EMC : " + str(svm_acc[3][-1]))
            print("QUIRE : " + str(svm_acc[6][-1]))
            print("RBMAL: " +str(svm_acc[4][-1]))
            print("BMDR : " + str(svm_acc[5][-1]))
            print("SP-AL : " + str(svm_acc[7][-1]))
            print("Random : " + str(svm_acc[8][-1]))
            print("-----------------------------F1----------------------------------")
            print("Unceratinty : " + str(svm_f1[0][-1]))
            print("QBC : " + str(svm_f1[1][-1]))
            print("EER : " + str(svm_f1[2][-1]))
            print("EMC : " + str(svm_f1[3][-1]))
            print("QUIRE : " + str(svm_f1[6][-1]))
            print("RBMAL: " + str(svm_f1[4][-1]))
            print("BMDR : " + str(svm_f1[5][-1]))
            print("SP-AL : " + str(svm_f1[7][-1]))
            print("Random : " + str(svm_f1[8][-1]))
            print("-----------------------------ROC_AUC-----------------------------")
            print("Unceratinty : " + str(svm_auc[0][-1]))
            print("QBC : " + str(svm_auc[1][-1]))
            print("EER : " + str(svm_auc[2][-1]))
            print("EMC : " + str(svm_auc[3][-1]))
            print("RBMAL: " + str(svm_auc[4][-1]))
            print("QUIRE : " + str(svm_auc[6][-1]))
            print("BMDR : " + str(svm_auc[5][-1]))
            print("SP-AL : " + str(svm_auc[7][-1]))
            print("Random : " + str(svm_auc[8][-1]))
            print("-----------------------------------------------------------------------------------------")

            fig_svm, ax_svm = plt.subplots(nrows=3, ncols=1, figsize=(8.5, 6), dpi=130)
            ax_svm[0].plot(svm_acc[0], label="Uncertainty")
            ax_svm[0].plot(svm_acc[1], label="QBC")
            ax_svm[0].plot(svm_acc[2], label="EER")
            ax_svm[0].plot(svm_acc[3], label="EMC")
            ax_svm[0].plot(svm_acc[4], label="RBMAL")
            ax_svm[0].plot(svm_acc[5], label="BMDR")
            ax_svm[0].plot(svm_acc[6], label="QUIRE")
            ax_svm[0].plot(svm_acc[7], label="SP-AL")
            ax_svm[0].plot(svm_acc[8], label="Random")
            ax_svm[1].plot(svm_f1[0], label="Uncertainty")
            ax_svm[1].plot(svm_f1[1], label="QBC")
            ax_svm[1].plot(svm_f1[2], label="EER")
            ax_svm[1].plot(svm_f1[3], label="EMC")
            ax_svm[1].plot(svm_f1[4], label="RBMAL")
            ax_svm[1].plot(svm_f1[5], label="BMDR")
            ax_svm[1].plot(svm_f1[6], label="QUIRE")
            ax_svm[1].plot(svm_f1[7], label="SP-AL")
            ax_svm[1].plot(svm_f1[8], label="Random")
            ax_svm[2].plot(svm_auc[0], label="Uncertainty")
            ax_svm[2].plot(svm_auc[1], label="QBC")
            ax_svm[2].plot(svm_auc[2], label="EER")
            ax_svm[2].plot(svm_auc[3], label="EMC")
            ax_svm[2].plot(svm_auc[4], label="RBMAL")
            ax_svm[2].plot(svm_auc[5], label="BMDR")
            ax_svm[2].plot(svm_auc[6], label="QUIRE")
            ax_svm[2].plot(svm_auc[7], label="SP-AL")
            ax_svm[2].plot(svm_auc[8], label="Random")

            ax_svm[0].grid(True)
            ax_svm[1].grid(True)
            ax_svm[2].grid(True)
            fig_svm.suptitle("SVM |d| = " + str(size), fontsize = 16)
            ax_svm[0].set_ylabel("Accuracy")
            ax_svm[1].set_ylabel("F1")
            ax_svm[2].set_ylabel("ROC_AUC")
            plt.legend()
            plt.savefig("svm_"+str(size)+"_dsgc.png")


            print("----- GPC Experiment with " + str(size) + " instances and " + str(repetitions) + " repetitions: -----")
            gp_acc, gp_f1, gp_auc = experiment_gpc(X_raw_full, y_raw_full, 250, repetitions, samples)

            print("-----------------------------RESULTS (GPC "+ str(size) + ") -----------------------------")
            print("-----------------------------Accuracy----------------------------")
            print("Unceratinty : " + str(gp_acc[0][-1]))
            print("QBC : " + str(gp_acc[1][-1]))
            print("EER : " + str(gp_acc[2][-1]))
            print("EMC : " + str(gp_acc[3][-1]))
            print("VarRedu: " + str(gp_acc[4][-1]))
            print("QUIRE : " + str(gp_acc[7][-1]))
            print("RBMAL: " + str(gp_acc[5][-1]))
            print("BMDR : " + str(gp_acc[6][-1]))
            print("SP-AL : " + str(gp_acc[8][-1]))
            print("Random : " + str(gp_acc[9][-1]))
            print("-----------------------------F1----------------------------------")
            print("Unceratinty : " + str(gp_f1[0][-1]))
            print("QBC : " + str(gp_f1[1][-1]))
            print("EER : " + str(gp_f1[2][-1]))
            print("EMC : " + str(gp_f1[3][-1]))
            print("VarRedu: " + str(gp_f1[4][-1]))
            print("QUIRE : " + str(gp_f1[7][-1]))
            print("RBMAL: " + str(gp_f1[5][-1]))
            print("BMDR : " + str(gp_f1[6][-1]))
            print("SP-AL : " + str(gp_f1[8][-1]))
            print("Random : " + str(gp_f1[9][-1]))
            print("-----------------------------ROC_AUC-----------------------------")
            print("Unceratinty : " + str(gp_auc[0][-1]))
            print("QBC : " + str(gp_auc[1][-1]))
            print("EER : " + str(gp_auc[2][-1]))
            print("EMC : " + str(gp_auc[3][-1]))
            print("VarRedu : " + str(gp_auc[4][-1]))
            print("QUIRE : " + str(gp_auc[7][-1]))
            print("RBMAL: " + str(gp_auc[5][-1]))
            print("BMDR : " + str(gp_auc[6][-1]))
            print("SP-AL : " + str(gp_auc[8][-1]))
            print("Random : " + str(gp_auc[9][-1]))
            print("-----------------------------------------------------------------------------------------")

            fig_gp, ax_gp = plt.subplots(nrows=3, ncols=1, figsize=(8.5, 6), dpi=130)
            ax_gp[0].plot(gp_acc[0], label="Uncertainty")
            ax_gp[0].plot(gp_acc[1], label="QBC")
            ax_gp[0].plot(gp_acc[2], label="EER")
            ax_gp[0].plot(gp_acc[3], label="EMC")
            ax_gp[0].plot(gp_acc[4], label="VarRedu")
            ax_gp[0].plot(gp_acc[5], label="RBMAL")
            ax_gp[0].plot(gp_acc[6], label="BMDR")
            ax_gp[0].plot(gp_acc[7], label="QUIRE")
            ax_gp[0].plot(gp_acc[8], label="SP-AL")
            ax_gp[0].plot(gp_acc[9], label="Random")
            ax_gp[1].plot(gp_f1[0], label="Uncertainty")
            ax_gp[1].plot(gp_f1[1], label="QBC")
            ax_gp[1].plot(gp_f1[2], label="EER")
            ax_gp[1].plot(gp_f1[3], label="EMC")
            ax_gp[1].plot(gp_f1[4], label="VarRedu")
            ax_gp[1].plot(gp_f1[5], label="RBMAL")
            ax_gp[1].plot(gp_f1[6], label="BMDR")
            ax_gp[1].plot(gp_f1[7], label="QUIRE")
            ax_gp[1].plot(gp_f1[8], label="SP-AL")
            ax_gp[1].plot(gp_f1[9], label="Random")
            ax_gp[2].plot(gp_auc[0], label="Uncertainty")
            ax_gp[2].plot(gp_auc[1], label="QBC")
            ax_gp[2].plot(gp_auc[2], label="EER")
            ax_gp[2].plot(gp_auc[3], label="EMC")
            ax_gp[2].plot(gp_auc[4], label="VarRedu")
            ax_gp[2].plot(gp_auc[5], label="RBMAL")
            ax_gp[2].plot(gp_auc[6], label="BMDR")
            ax_gp[2].plot(gp_auc[7], label="QUIRE")
            ax_gp[2].plot(gp_auc[8], label="SP-AL")
            ax_gp[2].plot(gp_auc[9], label="Random")
            ax_gp[0].grid(True)
            ax_gp[1].grid(True)
            ax_gp[2].grid(True)
            fig_gp.suptitle("GPC |d| = " + str(size), fontsize = 16)
            ax_gp[0].set_ylabel("Accuracy")
            ax_gp[1].set_ylabel("F1")
            ax_gp[2].set_ylabel("ROC_AUC")
            plt.legend()
            plt.savefig("gpc_"+str(size)+"_dsgc.png")



if __name__ == '__main__':
    main()
