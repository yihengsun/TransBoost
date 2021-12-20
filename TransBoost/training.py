# coding: utf-8
# pylint: disable=too-many-locals, too-many-arguments, invalid-name
# pylint: disable=too-many-branches, too-many-statements
"""Training Library containing training routines."""
import warnings
import copy
import numpy as np
from .core import Booster, XGBoostError, _get_booster_layer_trees
from .compat import (SKLEARN_INSTALLED, XGBStratifiedKFold)
from . import callback
from scipy.special import expit,logsumexp
import pandas as pd


def _configure_deprecated_callbacks(
        verbose_eval, early_stopping_rounds, maximize, start_iteration,
        num_boost_round, feval, evals_result, callbacks, show_stdv, cvfolds):
    link = 'https://xgboost.readthedocs.io/en/latest/python/callbacks.html'
    warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
    # Most of legacy advanced options becomes callbacks
    if early_stopping_rounds is not None:
        callbacks.append(callback.early_stop(early_stopping_rounds,
                                             maximize=maximize,
                                             verbose=bool(verbose_eval)))
    if isinstance(verbose_eval, bool) and verbose_eval:
        callbacks.append(callback.print_evaluation(show_stdv=show_stdv))
    else:
        if isinstance(verbose_eval, int):
            callbacks.append(callback.print_evaluation(verbose_eval,
                                                       show_stdv=show_stdv))
    if evals_result is not None:
        callbacks.append(callback.record_evaluation(evals_result))
    callbacks = callback.LegacyCallbacks(
        callbacks, start_iteration, num_boost_round, feval, cvfolds=cvfolds)
    return callbacks


def _is_new_callback(callbacks):
    return any(isinstance(c, callback.TrainingCallback)
               for c in callbacks) or not callbacks



def _train_internal(params, dtrain,
                    num_boost_round=10, evals=(),
                    obj=None, feval=None,
                    xgb_model=None, callbacks=None,
                    evals_result=None, maximize=None,
                    verbose_eval=None, early_stopping_rounds=None):
    """internal training function"""
    callbacks = [] if callbacks is None else copy.copy(callbacks)
    evals = list(evals)

    bst = Booster(params, [dtrain] + [d[0] for d in evals])

    if xgb_model is not None:
        bst = Booster(params, [dtrain] + [d[0] for d in evals],
                      model_file=xgb_model)

    start_iteration = 0

    is_new_callback = _is_new_callback(callbacks)
    if is_new_callback:
        assert all(isinstance(c, callback.TrainingCallback)
                   for c in callbacks), "You can't mix new and old callback styles."
        if verbose_eval:
            verbose_eval = 1 if verbose_eval is True else verbose_eval
            callbacks.append(callback.EvaluationMonitor(period=verbose_eval))
        if early_stopping_rounds:
            callbacks.append(callback.EarlyStopping(
                rounds=early_stopping_rounds, maximize=maximize))
        callbacks = callback.CallbackContainer(callbacks, metric=feval)
    else:
        callbacks = _configure_deprecated_callbacks(
            verbose_eval, early_stopping_rounds, maximize, start_iteration,
            num_boost_round, feval, evals_result, callbacks,
            show_stdv=False, cvfolds=None)

    bst = callbacks.before_training(bst)

    for i in range(start_iteration, num_boost_round):
        if callbacks.before_iteration(bst, i, dtrain, evals):
            break
        bst.update(dtrain, i, obj)
        if callbacks.after_iteration(bst, i, dtrain, evals):
            break

    bst = callbacks.after_training(bst)

    if evals_result is not None and is_new_callback:
        evals_result.update(callbacks.history)

    # These should be moved into callback functions `after_training`, but until old
    # callbacks are removed, the train function is the only place for setting the
    # attributes.
    num_parallel_tree, _ = _get_booster_layer_trees(bst)
    if bst.attr('best_score') is not None:
        bst.best_score = float(bst.attr('best_score'))
        bst.best_iteration = int(bst.attr('best_iteration'))
        # num_class is handled internally
        bst.set_attr(
            best_ntree_limit=str((bst.best_iteration + 1) * num_parallel_tree)
        )
        bst.best_ntree_limit = int(bst.attr("best_ntree_limit"))
    else:
        # Due to compatibility with version older than 1.4, these attributes are added
        # to Python object even if early stopping is not used.
        bst.best_iteration = bst.num_boosted_rounds() - 1
        bst.best_ntree_limit = (bst.best_iteration + 1) * num_parallel_tree

    # Copy to serialise and unserialise booster to reset state and free
    # training memory
    return bst.copy()

def _train_transboost_internal(params, dtrain,
                    sample_type,
                    num_boost_round=10, evals=(),
                    obj=None, feval=None,
                    xgb_model=None, callbacks=None,
                    evals_result=None, maximize=None,
                    verbose_eval=None, early_stopping_rounds=None, param_transboost=None):
    """internal training function"""
    callbacks = [] if callbacks is None else copy.copy(callbacks)
    evals = list(evals)

    bst = Booster(params, [dtrain] + [d[0] for d in evals])

    if xgb_model is not None:
        bst = Booster(params, [dtrain] + [d[0] for d in evals],
                      model_file=xgb_model)

    start_iteration = 0

    is_new_callback = _is_new_callback(callbacks)
    if is_new_callback:
        assert all(isinstance(c, callback.TrainingCallback)
                   for c in callbacks), "You can't mix new and old callback styles."
        if verbose_eval:
            verbose_eval = 1 if verbose_eval is True else verbose_eval
            callbacks.append(callback.EvaluationMonitor(period=verbose_eval))
        if early_stopping_rounds:
            callbacks.append(callback.EarlyStopping(
                rounds=early_stopping_rounds, maximize=maximize))
        callbacks = callback.CallbackContainer(callbacks, metric=feval)
    else:
        callbacks = _configure_deprecated_callbacks(
            verbose_eval, early_stopping_rounds, maximize, start_iteration,
            num_boost_round, feval, evals_result, callbacks,
            show_stdv=False, cvfolds=None)

    bst = callbacks.before_training(bst)
    # TransBoost code 
    source_domain_size = len(sample_type[sample_type==1])
    target_domain_size = len(sample_type[sample_type==0])
    gb_ratio_source = dtrain.get_label()[sample_type==1].sum() / source_domain_size
    
    # Optional:Add a decay for source domain samples. In the final iteration, the decay equals target_domain_size/source_domain_size
    decay_ratio = param_transboost['transfer_decay_ratio']
    if source_domain_size >target_domain_size * decay_ratio and decay_ratio>0:
        decay_rate = np.log(target_domain_size * decay_ratio / source_domain_size)/ param_transboost['n_estimators']
    else:
        decay_rate = 0.
    
    # Initialize trasfer margin with base score 0.5 (margin = .0)
    margin_transfer = np.zeros([sample_type.shape[0],2],dtype=np.float32) 
    # Make a backup for sample_weight
    sample_weight_origin = np.copy(dtrain.get_weight())
    

    for i in range(start_iteration, num_boost_round):

        if callbacks.before_iteration(bst, i, dtrain, evals):
            break

        # Because we do not need to implicitly output the ancillary model for source domain, 
        # we only restore the prediction of ancillary model in each iteration to save memory. 

        # Compute loss and gradient on ancillary model  
        y_pred_transfer = np.hstack((expit(margin_transfer[:,1][sample_type==1]),\
                        expit(margin_transfer[:,0][sample_type==0])))
        jacc_transfer = (dtrain.get_label() - y_pred_transfer) * dtrain.get_weight()

        if param_transboost['transfer_margin_estimation'] == 'firstorder':
            hess_transfer = dtrain.get_weight()
        else:
            hess_transfer = y_pred_transfer*(1-y_pred_transfer) * dtrain.get_weight()

        # Update a new tree for main model
        bst.update(dtrain, i, obj)

        # For each leaf, count the samples of source and target domian
        # Compute optimal leaf weight in ancillary model by minimizing Tylor's approximation of the loss.
        transfer_leaf = bst.predict(dtrain,output_margin=True,iteration_range=[bst.num_boosted_rounds()-1,bst.num_boosted_rounds()])
        transfer_grad = pd.DataFrame(np.array([transfer_leaf,sample_type,jacc_transfer,hess_transfer]).T).groupby([0,1]).agg({2:['sum'],3:['sum','count']}).reset_index()
        transfer_grad.columns = ['leaf','sample_type','jacc','hess','count']
        transfer_grad['grad'] = transfer_grad['jacc']/(transfer_grad['hess']+param_transboost['reg_lambda'])
        transfer_grad = transfer_grad.pivot(index='leaf',columns='sample_type',values=['grad','count','jacc','hess']).reset_index()
        transfer_grad.columns = ['leaf','grad_target','grad_source','count_target','count_source','jacc_target','jacc_source','hess_target','hess_source']
        transfer_grad['grad_prior'] = (transfer_grad['jacc_target'].fillna(0)+transfer_grad['jacc_source'].fillna(0))/(transfer_grad['hess_target'].fillna(0)+transfer_grad['hess_source'].fillna(0)+param_transboost['reg_lambda'])
        
        X_source = transfer_leaf.astype(np.float32)
        X_target = np.copy(X_source)
        X_target_mask = np.copy(X_target)
        X_source_mask = np.copy(X_source)
        
        X_source_weight = np.copy(X_source)        
        # Compute marinal distribution difference
        for leaf,grad_target,grad_source,count_target,count_source,jacc_target,jacc_source,hess_target,hess_source,grad_prior in transfer_grad.values:
            if np.isnan(grad_source) or count_target<param_transboost['transfer_min_leaf_size']:

                if param_transboost['transfer_prior_margin'] == 'zero':
                    grad_source =.0
                elif param_transboost['transfer_prior_margin'] == 'mirror' and not np.isnan(grad_target):
                    grad_source = grad_target
                elif param_transboost['transfer_prior_margin'] == 'prior':
                    grad_source = grad_prior
                else:
                    grad_source = leaf
            if np.isnan(grad_target) or count_target<param_transboost['transfer_min_leaf_size']:
                if param_transboost['transfer_prior_margin'] == 'zero':
                    grad_target =.0
                elif param_transboost['transfer_prior_margin'] == 'mirror' and not np.isnan(grad_source):
                    grad_target = grad_source
                elif param_transboost['transfer_prior_margin'] == 'prior':
                    grad_target = grad_prior
                else:
                    grad_target = leaf
            X_target[X_target_mask == leaf] = grad_target
            X_source[X_source_mask == leaf] = grad_source
            if count_target < param_transboost['transfer_min_leaf_size']:
                weight = 1.0
            else:             
                weight = (count_target/count_source)/(target_domain_size/source_domain_size)
            X_source_weight[X_source_mask == leaf] = weight
        # Update the prediction of ancillay model
        margin_transfer = margin_transfer + param_transboost['learning_rate'] * np.vstack((X_target,X_source)).T * param_transboost['transfer_velocity']

        # Compute sample weight for marginal distribution
        X_source_weight = pd.Series(X_source_weight).fillna(1.0).values
        marginal_weight = X_source_weight[sample_type==1]
    
        # Compute sample weight for conditional distribution。
        p_target = expit(margin_transfer[:,0][sample_type==1])
        p_source = expit(margin_transfer[:,1][sample_type==1])
        conditional_weight = (p_target * dtrain.get_label()[sample_type==1] + (1-p_target) * (1-dtrain.get_label()[sample_type==1])) \
                        / (p_source * dtrain.get_label()[sample_type==1] + (1-p_source) * (1-dtrain.get_label()[sample_type==1]))
       
        # Compute overall weight（Eq.4 in the paper)
        sample_weight = marginal_weight * conditional_weight

        # Optional:add a decay to the weight of source domain; 
        sample_weight = sample_weight * np.e ** (decay_rate * i)
        
        # Optional:reweight source domain's sample_weight to initial gb_ratio_source, default False
        if param_transboost['transfer_rebalance']:
            weight_1 = gb_ratio_source * sample_weight.sum() / sample_weight[dtrain.get_label()[sample_type==1]==1].sum()
            weight_0 = (1-gb_ratio_source) * sample_weight.sum() / sample_weight[dtrain.get_label()[sample_type==1]==0].sum()
            reweight_matrix = np.ones(source_domain_size)
            reweight_matrix[dtrain.get_label()[sample_type==1]==1] = weight_1
            reweight_matrix[dtrain.get_label()[sample_type==1]==0] = weight_0    
            sample_weight = sample_weight * reweight_matrix
    
        # Update sample weight for next iteration
        sample_weight = np.append(sample_weight,np.ones(target_domain_size))
        sample_weight = sample_weight * sample_weight_origin
        dtrain.set_weight(sample_weight)
        
        if callbacks.after_iteration(bst, i, dtrain, evals):
            break
    bst = callbacks.after_training(bst)

    if evals_result is not None and is_new_callback:
        evals_result.update(callbacks.history)

    # These should be moved into callback functions `after_training`, but until old
    # callbacks are removed, the train function is the only place for setting the
    # attributes.
    num_parallel_tree, _ = _get_booster_layer_trees(bst)
    if bst.attr('best_score') is not None:
        bst.best_score = float(bst.attr('best_score'))
        bst.best_iteration = int(bst.attr('best_iteration'))
        # num_class is handled internally
        bst.set_attr(
            best_ntree_limit=str((bst.best_iteration + 1) * num_parallel_tree)
        )
        bst.best_ntree_limit = int(bst.attr("best_ntree_limit"))
    else:
        # Due to compatibility with version older than 1.4, these attributes are added
        # to Python object even if early stopping is not used.
        bst.best_iteration = bst.num_boosted_rounds() - 1
        bst.best_ntree_limit = (bst.best_iteration + 1) * num_parallel_tree

    # Copy to serialise and unserialise booster to reset state and free
    # training memory
    return bst.copy()

def train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
          maximize=None, early_stopping_rounds=None, evals_result=None,
          verbose_eval=True, xgb_model=None, callbacks=None):
    # pylint: disable=too-many-statements,too-many-branches, attribute-defined-outside-init
    """Train a booster with given parameters.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained.
    num_boost_round: int
        Number of boosting iterations.
    evals: list of pairs (DMatrix, string)
        List of validation sets for which metrics will evaluated during training.
        Validation metrics will help us track the performance of the model.
    obj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
    maximize : bool
        Whether to maximize feval.
    early_stopping_rounds: int
        Activates early stopping. Validation metric needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.
        Requires at least one item in **evals**.
        The method returns the model from the last iteration (not the best one).  Use
        custom callback or model slicing if the best model is desired.
        If there's more than one item in **evals**, the last entry will be used for early
        stopping.
        If there's more than one metric in the **eval_metric** parameter given in
        **params**, the last metric will be used for early stopping.
        If early stopping occurs, the model will have three additional fields:
        ``bst.best_score``, ``bst.best_iteration`` and ``bst.best_ntree_limit``.  Use
        ``bst.best_ntree_limit`` to get the correct value if ``num_parallel_tree`` and/or
        ``num_class`` appears in the parameters.  ``best_ntree_limit`` is the result of
        ``num_parallel_tree * best_iteration``.
    evals_result: dict
        This dictionary stores the evaluation results of all the items in watchlist.

        Example: with a watchlist containing
        ``[(dtest,'eval'), (dtrain,'train')]`` and
        a parameter containing ``('eval_metric': 'logloss')``,
        the **evals_result** returns

        .. code-block:: python

            {'train': {'logloss': ['0.48253', '0.35953']},
             'eval': {'logloss': ['0.480385', '0.357756']}}

    verbose_eval : bool or int
        Requires at least one item in **evals**.
        If **verbose_eval** is True then the evaluation metric on the validation set is
        printed at each boosting stage.
        If **verbose_eval** is an integer then the evaluation metric on the validation set
        is printed at every given **verbose_eval** boosting stage. The last boosting stage
        / the boosting stage found by using **early_stopping_rounds** is also printed.
        Example: with ``verbose_eval=4`` and at least one item in **evals**, an evaluation metric
        is printed every 4 boosting stages, instead of every boosting stage.
    xgb_model : file name of stored xgb model or 'Booster' instance
        Xgb model to be loaded before training (allows training continuation).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        :ref:`Callback API <callback_api>`.
        Example:

        .. code-block:: python

            [xgb.callback.LearningRateScheduler(custom_rates)]

    Returns
    -------
    Booster : a trained booster model
    """
    bst = _train_internal(params, dtrain,
                          num_boost_round=num_boost_round,
                          evals=evals,
                          obj=obj, feval=feval,
                          xgb_model=xgb_model, callbacks=callbacks,
                          verbose_eval=verbose_eval,
                          evals_result=evals_result,
                          maximize=maximize,
                          early_stopping_rounds=early_stopping_rounds)
    return bst

def train_transboost(params, dtrain, sample_type, num_boost_round=10, evals=(), obj=None, feval=None,
          maximize=None, early_stopping_rounds=None, evals_result=None,
          verbose_eval=True, xgb_model=None, callbacks=None, param_transboost=None):
    # pylint: disable=too-many-statements,too-many-branches, attribute-defined-outside-init
    """Train a booster with given parameters.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained.
    num_boost_round: int
        Number of boosting iterations.
    evals: list of pairs (DMatrix, string)
        List of validation sets for which metrics will evaluated during training.
        Validation metrics will help us track the performance of the model.
    obj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
    maximize : bool
        Whether to maximize feval.
    early_stopping_rounds: int
        Activates early stopping. Validation metric needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.
        Requires at least one item in **evals**.
        The method returns the model from the last iteration (not the best one).  Use
        custom callback or model slicing if the best model is desired.
        If there's more than one item in **evals**, the last entry will be used for early
        stopping.
        If there's more than one metric in the **eval_metric** parameter given in
        **params**, the last metric will be used for early stopping.
        If early stopping occurs, the model will have three additional fields:
        ``bst.best_score``, ``bst.best_iteration`` and ``bst.best_ntree_limit``.  Use
        ``bst.best_ntree_limit`` to get the correct value if ``num_parallel_tree`` and/or
        ``num_class`` appears in the parameters.  ``best_ntree_limit`` is the result of
        ``num_parallel_tree * best_iteration``.
    evals_result: dict
        This dictionary stores the evaluation results of all the items in watchlist.

        Example: with a watchlist containing
        ``[(dtest,'eval'), (dtrain,'train')]`` and
        a parameter containing ``('eval_metric': 'logloss')``,
        the **evals_result** returns

        .. code-block:: python

            {'train': {'logloss': ['0.48253', '0.35953']},
             'eval': {'logloss': ['0.480385', '0.357756']}}

    verbose_eval : bool or int
        Requires at least one item in **evals**.
        If **verbose_eval** is True then the evaluation metric on the validation set is
        printed at each boosting stage.
        If **verbose_eval** is an integer then the evaluation metric on the validation set
        is printed at every given **verbose_eval** boosting stage. The last boosting stage
        / the boosting stage found by using **early_stopping_rounds** is also printed.
        Example: with ``verbose_eval=4`` and at least one item in **evals**, an evaluation metric
        is printed every 4 boosting stages, instead of every boosting stage.
    xgb_model : file name of stored xgb model or 'Booster' instance
        Xgb model to be loaded before training (allows training continuation).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        :ref:`Callback API <callback_api>`.
        Example:
    param_transboost : hyper-parameters for Transboost

        .. code-block:: python

            [xgb.callback.LearningRateScheduler(custom_rates)]

    Returns
    -------
    Booster : a trained booster model
    """
    bst = _train_transboost_internal(params, dtrain, sample_type,
                          num_boost_round=num_boost_round,
                          evals=evals,
                          obj=obj, feval=feval,
                          xgb_model=xgb_model, callbacks=callbacks,
                          verbose_eval=verbose_eval,
                          evals_result=evals_result,
                          maximize=maximize,
                          early_stopping_rounds=early_stopping_rounds,
                          param_transboost=param_transboost)
    return bst


class CVPack(object):
    """"Auxiliary datastruct to hold one fold of CV."""
    def __init__(self, dtrain, dtest, param):
        """"Initialize the CVPack"""
        self.dtrain = dtrain
        self.dtest = dtest
        self.watchlist = [(dtrain, 'train'), (dtest, 'test')]
        self.bst = Booster(param, [dtrain, dtest])

    def __getattr__(self, name):
        def _inner(*args, **kwargs):
            return getattr(self.bst, name)(*args, **kwargs)
        return _inner

    def update(self, iteration, fobj):
        """"Update the boosters for one iteration"""
        self.bst.update(self.dtrain, iteration, fobj)

    def eval(self, iteration, feval):
        """"Evaluate the CVPack for one iteration."""
        return self.bst.eval_set(self.watchlist, iteration, feval)


class _PackedBooster:
    def __init__(self, cvfolds) -> None:
        self.cvfolds = cvfolds

    def update(self, iteration, obj):
        '''Iterate through folds for update'''
        for fold in self.cvfolds:
            fold.update(iteration, obj)

    def eval(self, iteration, feval):
        '''Iterate through folds for eval'''
        result = [f.eval(iteration, feval) for f in self.cvfolds]
        return result

    def set_attr(self, **kwargs):
        '''Iterate through folds for setting attributes'''
        for f in self.cvfolds:
            f.bst.set_attr(**kwargs)

    def attr(self, key):
        '''Redirect to booster attr.'''
        return self.cvfolds[0].bst.attr(key)

    def set_param(self, params, value=None):
        """Iterate through folds for set_param"""
        for f in self.cvfolds:
            f.bst.set_param(params, value)

    def num_boosted_rounds(self):
        '''Number of boosted rounds.'''
        return self.cvfolds[0].num_boosted_rounds()

    @property
    def best_iteration(self):
        '''Get best_iteration'''
        return int(self.cvfolds[0].bst.attr("best_iteration"))

    @property
    def best_score(self):
        """Get best_score."""
        return float(self.cvfolds[0].bst.attr("best_score"))


def groups_to_rows(groups, boundaries):
    """
    Given group row boundaries, convert ground indexes to row indexes
    :param groups: list of groups for testing
    :param boundaries: rows index limits of each group
    :return: row in group
    """
    return np.concatenate([np.arange(boundaries[g], boundaries[g+1]) for g in groups])


def mkgroupfold(dall, nfold, param, evals=(), fpreproc=None, shuffle=True):
    """
    Make n folds for cross-validation maintaining groups
    :return: cross-validation folds
    """
    # we have groups for pairwise ranking... get a list of the group indexes
    group_boundaries = dall.get_uint_info('group_ptr')
    group_sizes = np.diff(group_boundaries)

    if shuffle is True:
        idx = np.random.permutation(len(group_sizes))
    else:
        idx = np.arange(len(group_sizes))
    # list by fold of test group indexes
    out_group_idset = np.array_split(idx, nfold)
    # list by fold of train group indexes
    in_group_idset = [np.concatenate([out_group_idset[i] for i in range(nfold) if k != i])
                      for k in range(nfold)]
    # from the group indexes, convert them to row indexes
    in_idset = [groups_to_rows(in_groups, group_boundaries) for in_groups in in_group_idset]
    out_idset = [groups_to_rows(out_groups, group_boundaries) for out_groups in out_group_idset]

    # build the folds by taking the appropriate slices
    ret = []
    for k in range(nfold):
        # perform the slicing using the indexes determined by the above methods
        dtrain = dall.slice(in_idset[k], allow_groups=True)
        dtrain.set_group(group_sizes[in_group_idset[k]])
        dtest = dall.slice(out_idset[k], allow_groups=True)
        dtest.set_group(group_sizes[out_group_idset[k]])
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            dtrain, dtest, tparam = fpreproc(dtrain, dtest, param.copy())
        else:
            tparam = param
        plst = list(tparam.items()) + [('eval_metric', itm) for itm in evals]
        ret.append(CVPack(dtrain, dtest, plst))
    return ret


def mknfold(dall, nfold, param, seed, evals=(), fpreproc=None, stratified=False,
            folds=None, shuffle=True):
    """
    Make an n-fold list of CVPack from random indices.
    """
    evals = list(evals)
    np.random.seed(seed)

    if stratified is False and folds is None:
        # Do standard k-fold cross validation. Automatically determine the folds.
        if len(dall.get_uint_info('group_ptr')) > 1:
            return mkgroupfold(dall, nfold, param, evals=evals, fpreproc=fpreproc, shuffle=shuffle)

        if shuffle is True:
            idx = np.random.permutation(dall.num_row())
        else:
            idx = np.arange(dall.num_row())
        out_idset = np.array_split(idx, nfold)
        in_idset = [np.concatenate([out_idset[i] for i in range(nfold) if k != i])
                    for k in range(nfold)]
    elif folds is not None:
        # Use user specified custom split using indices
        try:
            in_idset = [x[0] for x in folds]
            out_idset = [x[1] for x in folds]
        except TypeError:
            # Custom stratification using Sklearn KFoldSplit object
            splits = list(folds.split(X=dall.get_label(), y=dall.get_label()))
            in_idset = [x[0] for x in splits]
            out_idset = [x[1] for x in splits]
        nfold = len(out_idset)
    else:
        # Do standard stratefied shuffle k-fold split
        sfk = XGBStratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
        splits = list(sfk.split(X=dall.get_label(), y=dall.get_label()))
        in_idset = [x[0] for x in splits]
        out_idset = [x[1] for x in splits]
        nfold = len(out_idset)

    ret = []
    for k in range(nfold):
        # perform the slicing using the indexes determined by the above methods
        dtrain = dall.slice(in_idset[k])
        dtest = dall.slice(out_idset[k])
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            dtrain, dtest, tparam = fpreproc(dtrain, dtest, param.copy())
        else:
            tparam = param
        plst = list(tparam.items()) + [('eval_metric', itm) for itm in evals]
        ret.append(CVPack(dtrain, dtest, plst))
    return ret


def cv(params, dtrain, num_boost_round=10, nfold=3, stratified=False, folds=None,
       metrics=(), obj=None, feval=None, maximize=None, early_stopping_rounds=None,
       fpreproc=None, as_pandas=True, verbose_eval=None, show_stdv=True,
       seed=0, callbacks=None, shuffle=True):
    # pylint: disable = invalid-name
    """Cross-validation with given parameters.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained.
    num_boost_round : int
        Number of boosting iterations.
    nfold : int
        Number of folds in CV.
    stratified : bool
        Perform stratified sampling.
    folds : a KFold or StratifiedKFold instance or list of fold indices
        Sklearn KFolds or StratifiedKFolds object.
        Alternatively may explicitly pass sample indices for each fold.
        For ``n`` folds, **folds** should be a length ``n`` list of tuples.
        Each tuple is ``(in,out)`` where ``in`` is a list of indices to be used
        as the training samples for the ``n`` th fold and ``out`` is a list of
        indices to be used as the testing samples for the ``n`` th fold.
    metrics : string or list of strings
        Evaluation metrics to be watched in CV.
    obj : function
        Custom objective function.
    feval : function
        Custom evaluation function.
    maximize : bool
        Whether to maximize feval.
    early_stopping_rounds: int
        Activates early stopping. Cross-Validation metric (average of validation
        metric computed over CV folds) needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.
        The last entry in the evaluation history will represent the best iteration.
        If there's more than one metric in the **eval_metric** parameter given in
        **params**, the last metric will be used for early stopping.
    fpreproc : function
        Preprocessing function that takes (dtrain, dtest, param) and returns
        transformed versions of those.
    as_pandas : bool, default True
        Return pd.DataFrame when pandas is installed.
        If False or pandas is not installed, return np.ndarray
    verbose_eval : bool, int, or None, default None
        Whether to display the progress. If None, progress will be displayed
        when np.ndarray is returned. If True, progress will be displayed at
        boosting stage. If an integer is given, progress will be displayed
        at every given `verbose_eval` boosting stage.
    show_stdv : bool, default True
        Whether to display the standard deviation in progress.
        Results are not affected, and always contains std.
    seed : int
        Seed used to generate the folds (passed to numpy.random.seed).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        :ref:`Callback API <callback_api>`.
        Example:

        .. code-block:: python

            [xgb.callback.LearningRateScheduler(custom_rates)]
    shuffle : bool
        Shuffle data before creating folds.

    Returns
    -------
    evaluation history : list(string)
    """
    if stratified is True and not SKLEARN_INSTALLED:
        raise XGBoostError('sklearn needs to be installed in order to use stratified cv')

    if isinstance(metrics, str):
        metrics = [metrics]

    if isinstance(params, list):
        _metrics = [x[1] for x in params if x[0] == 'eval_metric']
        params = dict(params)
        if 'eval_metric' in params:
            params['eval_metric'] = _metrics
    else:
        params = dict((k, v) for k, v in params.items())

    if (not metrics) and 'eval_metric' in params:
        if isinstance(params['eval_metric'], list):
            metrics = params['eval_metric']
        else:
            metrics = [params['eval_metric']]

    params.pop("eval_metric", None)

    results = {}
    cvfolds = mknfold(dtrain, nfold, params, seed, metrics, fpreproc,
                      stratified, folds, shuffle)

    # setup callbacks
    callbacks = [] if callbacks is None else callbacks
    is_new_callback = _is_new_callback(callbacks)
    if is_new_callback:
        assert all(isinstance(c, callback.TrainingCallback)
                   for c in callbacks), "You can't mix new and old callback styles."
        if isinstance(verbose_eval, bool) and verbose_eval:
            verbose_eval = 1 if verbose_eval is True else verbose_eval
            callbacks.append(callback.EvaluationMonitor(period=verbose_eval,
                                                        show_stdv=show_stdv))
        if early_stopping_rounds:
            callbacks.append(callback.EarlyStopping(
                rounds=early_stopping_rounds, maximize=maximize))
        callbacks = callback.CallbackContainer(callbacks, metric=feval, is_cv=True)
    else:
        callbacks = _configure_deprecated_callbacks(
            verbose_eval, early_stopping_rounds, maximize, 0,
            num_boost_round, feval, None, callbacks,
            show_stdv=show_stdv, cvfolds=cvfolds)
    booster = _PackedBooster(cvfolds)
    callbacks.before_training(booster)

    for i in range(num_boost_round):
        if callbacks.before_iteration(booster, i, dtrain, None):
            break
        booster.update(i, obj)

        should_break = callbacks.after_iteration(booster, i, dtrain, None)
        res = callbacks.aggregated_cv
        for key, mean, std in res:
            if key + '-mean' not in results:
                results[key + '-mean'] = []
            if key + '-std' not in results:
                results[key + '-std'] = []
            results[key + '-mean'].append(mean)
            results[key + '-std'].append(std)

        if should_break:
            for k in results:
                results[k] = results[k][:(booster.best_iteration + 1)]
            break
    if as_pandas:
        try:
            import pandas as pd
            results = pd.DataFrame.from_dict(results)
        except ImportError:
            pass

    callbacks.after_training(booster)

    return results
