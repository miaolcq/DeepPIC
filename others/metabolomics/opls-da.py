
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:34:29 2022

@author: tianmiao
"""

from pyopls import OPLS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from pyopls import OPLS
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score, accuracy_score

#OPLS-DA
data = pd.read_csv("D:/Dpic/data2/leaf_seed/s_python.csv",encoding='gbk')
X = np.array(data.values[:,1:].T,dtype=float)
Y= np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
opls = OPLS(1)
Z = opls.fit_transform(X, Y)
pls = PLSRegression(1)
uncorrected_r2 = r2_score(Y, pls.fit(X, Y).predict(X))
corrected_r2 = r2_score(Y, pls.fit(Z, Y).predict(Z))
uncorrected_q2 = r2_score(Y, cross_val_predict(pls, X, Y, cv=LeaveOneOut()))
corrected_q2 = r2_score(Y, cross_val_predict(pls, Z, Y, cv=LeaveOneOut()))
pls.fit_transform(Z, Y)
plt.figure(1)
df = pd.DataFrame(np.column_stack([pls.x_scores_, opls.T_ortho_[:, 0]]),
                  index=Y,columns=['t', 't_ortho'])                           
pos_df = df[Y==0]
neg_df = df[Y==1]
plt.scatter(neg_df['t'], neg_df['t_ortho'], c='cyan', marker='o',alpha=0.5, label='seed')
plt.scatter(pos_df['t'], pos_df['t_ortho'], c='orangered',marker='^',alpha=0.5, label='leaf')
plt.title('OPLS Scores')
plt.xlabel('t')
plt.ylabel('t_ortho')
plt.legend(loc = 'best')
#plt.show()
plt.savefig("D:/Dpic/wenzhang/plot/oplsda.png", dpi = 1200)

#permutation
import warnings
from sys import stderr
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import is_classifier, clone, ClassifierMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.utils import indexable, check_random_state
def _permutation_test_score(estimator, X, y, groups=None, cv='warn',
                            n_jobs=None, verbose=0, fit_params=None,
                            pre_dispatch='2*n_jobs', method='predict',
                            score_functions=None):
    """Auxiliary function for permutation_test_score"""
    if score_functions is None:
        score_functions = [r2_score]
    y_pred = cross_val_predict(estimator, X, y, groups, cv, n_jobs, verbose, fit_params, pre_dispatch, method)
    cv_scores = [score_function(y, y_pred) for score_function in score_functions]
    return np.array(cv_scores)
def _shuffle(y, groups, random_state):
    """Return a shuffled copy of y eventually shuffle among same groups."""
    if groups is None:
        indices = random_state.permutation(len(y))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = (groups == group)
            indices[this_mask] = random_state.permutation(indices[this_mask])
    return safe_indexing(y, indices)
def safe_indexing(X, indices):
    if hasattr(X, "iloc"):
        # Work-around for indexing with read-only indices in pandas
        indices = indices if indices.flags.writeable else indices.copy()
        # Pandas Dataframes and Series
        try:
            return X.iloc[indices]
        except ValueError:
            # Cython typed memoryviews internally used in pandas do not support
            # readonly buffers.
            warnings.warn("Copying input dataframe for slicing.",
                          DataConversionWarning)
            return X.copy().iloc[indices]
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]
def permutation_test_score(estimator, X, y, groups=None, cv='warn',
                           n_permutations=100, n_jobs=None, random_state=0,
                           verbose=0, pre_dispatch='2*n_jobs', cv_score_functions=None,
                           fit_params=None, method='predict', parallel_by='permutation'):
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    random_state = check_random_state(random_state)
    if cv_score_functions is None:
        if isinstance(estimator, ClassifierMixin):
            cv_score_functions = [accuracy_score]
        else:
            cv_score_functions = [r2_score]
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    score = _permutation_test_score(clone(estimator), X, y, groups, cv,
                                    n_jobs, verbose, fit_params, pre_dispatch,
                                    method, cv_score_functions)
    if parallel_by == 'estimation':
        permutation_scores = np.vstack([
            _permutation_test_score(
                clone(estimator), X, _shuffle(y, groups, random_state),
                groups, cv, n_jobs, verbose, fit_params, pre_dispatch,
                method, cv_score_functions
            ) for _ in range(n_permutations)
        ])
    elif parallel_by == 'permutation':
        permutation_scores = np.vstack(
            Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
                delayed(_permutation_test_score)(
                    clone(estimator), X, _shuffle(y, groups, random_state),
                    groups, cv, fit_params=fit_params, method=method, score_functions=cv_score_functions
                ) for _ in range(n_permutations)
            )
        )
    else:
        raise ValueError(f'Invalid option for parallel_by {parallel_by}')
    pvalue = (np.sum(permutation_scores >= score, axis=0) + 1.0) / (n_permutations + 1)
    return [(score[i], permutation_scores[:, i], pvalue[i]) for i in range(len(score))]
    # return score, permutation_scores, pvalue
cv = GroupKFold(2)
#,shuffle=True, random_state=45
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle=True, random_state=420) 
permutation_scores = permutation_test_score(pls, Z, Y, groups= None , cv=kf,
                           n_permutations=2000, n_jobs=1, random_state=420,
                           verbose=0, fit_params=None)
#pre_dispatch='2*n_jobs',cv_score_functions=None,method='predict',,  parallel_by='permutation'
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.hist(permutation_scores[1], bins=50, density=False, fc="lightpink", ec="magenta")
ax.axvline(permutation_scores[0], ls="--", color="cyan")
# score_label = f"Score on original\ndata: {score:.2f}\n(p-value: {pvalue:.3f})"
# ax.text(0.7, 10, score_label, fontsize=12)
ax.set_xlabel("Q2")
_ = ax.set_ylabel("Frequency")
fig.savefig("D:/Dpic/wenzhang/plot/permutation.png", dpi = 1200)

#permutation
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import StratifiedKFold
random = np.random.RandomState(seed=0)
E = random.normal(size=(len(X), 2000))
X = np.c_[X, E]
cv = StratifiedKFold(10,shuffle=True, random_state=420)
score, permutation_scores, pvalue = permutation_test_score(
pls, Z, Y, cv=cv,n_permutations=2000, n_jobs=1)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.hist(permutation_scores, bins=2000, density=True,fc="lightpink"  ec="magenta")
ax.axvline(score, ls="--", color="cyan")
# score_label = f"Score on original\ndata: {score:.2f}\n(p-value: {pvalue:.3f})"
# ax.text(0.7, 10, score_label, fontsize=12)
ax.set_xlabel("Score")
_ = ax.set_ylabel("Probability")
fig.savefig("D:/Dpic/wenzhang/plot/permutation.png", dpi = 1200)

#vip
def vip(x, y, model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    m, p = x.shape
    _, h = t.shape

    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)

    return vips


import matplotlib.pyplot as plt    
import seaborn as sns  
 
DATA = pd.read_csv("D:/Dpic/data2/leaf_seed/s_python.csv", encoding='gbk')

X = np.array(DATA.values[:,1:].T,dtype=float)
COMs = DATA.values[:,0]

VIPs = vip(Z, Y, pls)   

COM = []
VIP = []

for vv in range(11):
    if VIPs[vv]>=1:
        
        VIP.append(VIPs[vv])
        COM.append(COMs[vv])
 

sorted_vips = sorted(enumerate(VIP), key=lambda x: x[1])
idx = [i[0] for i in sorted_vips]
vips = [i[1] for i in sorted_vips]
COMS=[]
for jj in range(len(COM)):
    COMS.append(COM[idx[jj]])

plt.figure(figsize=(6,6))
plt.grid(axis="y",linestyle='-.')
plt.scatter(vips,range(len(COMS)),c="#88c999",s=100,marker="o")
plt.yticks(range(len(COMS)),COMS,size=16)
plt.xticks(size=15)
plt.savefig("D:/Dpic/wenzhang/plot/vip.png", dpi = 1200)

#HCA

import matplotlib.pyplot as plt    
import seaborn as sns  

DATA = pd.read_csv("D:/Dpic/data2/leaf_seed/hca_python.csv", encoding='gbk')

X = np.array(DATA.values[:,1:].T,dtype=float)
COM_vip = ['V2','V3','V4','V5','V6','V7','V8','V9','V10']        
sns.set(font_scale=1.4)
df = pd.DataFrame(X,index=None,columns = COM_vip)
cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
g = sns.clustermap(df.T,cmap="YlGnBu",col_cluster=False,row_cluster=True,annot_kws={"size": 30},standard_scale=1, linewidths = 0.5,  
                   cbar_kws=dict(orientation='horizontal'),figsize=(15,8),xticklabels=False)

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.26, 0.88, 0.64, 0.02])
g.ax_cbar.tick_params(axis='x', length=10)
for spine in g.ax_cbar.spines:
    g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(2)

plt.savefig("D:/Dpic/wenzhang/plot/hca.png", dpi = 1200)

