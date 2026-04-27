import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, recall_score, precision_score, auc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
#from sklearn.utils import ClassifierTags
from xgboost import XGBClassifier
from flaml import AutoML
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
pd.options.future.infer_string = False

class DefaultModel():
    def __init__(
            self, 
            model_type='logistic',
            **kwargs
            ):
        
        self.model_type = model_type
        self.params = kwargs or {}
        
    def _model_builder(self):
        if self.model_type == 'logistic':
            return LogisticRegression(
                **self.params,
            )
            
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                **self.params,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit_weights(self, X, approved_mask):
        """
        Fit a logistic regression model to estimate propensity scores and calculate inverse probability weights.
        """

        self.propensity_model = LogisticRegression(random_state=42)
        self.propensity_model.fit(X, approved_mask)
        self.ipw = True
    
    def fit(self, X, y):
        """
        Fit the logistic regression model to the data, optionally using inverse probability weights.
        """

        self.classes_ = np.unique(y)
        if getattr(self, 'ipw', False):
            propensity_scores = np.clip(self.propensity_model.predict_proba(X)[:, 1], a_min=0.1, a_max=0.9)
            weights = 1 / propensity_scores
            self.model_ = self._model_builder()
            self.model_.fit(X, y, sample_weight=weights)
        else:
            self.model_ = self._model_builder()
            self.model_.fit(X, y)

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict_proba(X)
    
    def predict(self, X, threshold=0.5):
        check_is_fitted(self, 'model_')
        return (self.model_.predict_proba(X)[:, 1] >= threshold).astype(int)


    #################
    # Not used in the final version, but keeping it here for reference
    #################
    def evaluate(self, y_true, y_pred_proba):
        """
        Evaluate the model using AUC-ROC and AUC-PR metrics and ROC and PR curves.
        """

        fpr, tpr, roc_threshold = roc_curve(y_true, y_pred_proba)
        precision, recall, pr_threshold = precision_recall_curve(y_true, y_pred_proba)
        
        auc_pr = auc(recall, precision)
        auc_roc = roc_auc_score(y_true, y_pred_proba)

        return auc_roc, auc_pr, (fpr, tpr, roc_threshold), (precision, recall, pr_threshold)
    
class DefaultModel_old(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            model_type='logistic',

            # logistic regression parameters
            C=1.0,
            class_weight='balanced',
            solver='lbfgs',
            max_iter=1000,

            # xgboost parameters
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=1.0,
            colsample_bylevel=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            max_leaves=0,
            ):
        

        self.model_type = model_type
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.max_leaves = max_leaves
        
    def _model_builder(self):
        if self.model_type == 'logistic':
            return LogisticRegression(
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=42,
            )
            
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                scale_pos_weight=self.scale_pos_weight,
                random_state=42,
                enable_categorical=True,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit_weights(self, X, approved_mask):
        """
        Fit a logistic regression model to estimate propensity scores and calculate inverse probability weights.
        """

        self.propensity_model = LogisticRegression(random_state=42)
        self.propensity_model.fit(X, approved_mask)
        self.ipw = True
    
    def fit(self, X, y):
        """
        Fit the logistic regression model to the data, optionally using inverse probability weights.
        """

        self.classes_ = np.unique(y)
        if getattr(self, 'ipw', False):
            propensity_scores = np.clip(self.propensity_model.predict_proba(X)[:, 1], a_min=0.1, a_max=0.9)
            weights = 1 / propensity_scores
            self.model_ = self._model_builder()
            self.model_.fit(X, y, sample_weight=weights)
        else:
            self.model_ = self._model_builder()
            self.model_.fit(X, y)

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict_proba(X)
    
    def predict(self, X, threshold=0.5):
        check_is_fitted(self, 'model_')
        return (self.model_.predict_proba(X)[:, 1] >= threshold).astype(int)

    def evaluate(self, y_true, y_pred_proba):
        """
        Evaluate the model using AUC-ROC and AUC-PR metrics and ROC and PR curves.
        """

        fpr, tpr, roc_threshold = roc_curve(y_true, y_pred_proba)
        precision, recall, pr_threshold = precision_recall_curve(y_true, y_pred_proba)
        
        auc_pr = auc(recall, precision)
        auc_roc = roc_auc_score(y_true, y_pred_proba)

        return auc_roc, auc_pr, (fpr, tpr, roc_threshold), (precision, recall, pr_threshold)
    
class HyperparameterTuner:
    def __init__(
            self, 
            time_budget=60, 
            metric='roc_auc', 
            estimators = ['xgboost'], 
            early_stop=True, 
            verbose=1, 
            split_type='time', 
            eval_method=None, 
            X_val=None, 
            y_val=None,
            scale_pos_weight=1.0,
            class_weight='balanced',
            ):

        self.settings = {
            "time_budget": time_budget,           
            "metric": metric,
            "task": 'classification',    
            "estimator_list": estimators, 
            "early_stop": early_stop,          
            "verbose": verbose,
            "split_type": split_type,
            "eval_method": 'cv' if X_val is None or y_val is None else 'holdout',
            "X_val": X_val,
            "y_val": y_val,
            "custom_hp": {
                'xgboost': {'scale_pos_weight': {'domain': scale_pos_weight}},
                'lrl1': {'class_weight': {'domain': class_weight}, 'max_iter': {'domain': 1000}},
                'lrl2': {'class_weight': {'domain': class_weight}, 'max_iter': {'domain': 1000}}
                }
        }

        self.automl = AutoML()

    def fit(self, X, y):
        self.automl.fit(X_train=X, y_train=y, **self.settings)

    def best_config(self):
        return self.automl.model.estimator.get_params()
    
    def best_estimator(self):
        return self.automl.model

def model_evals(y_true, y_pred_proba):
    """
    Evaluate and plot the ROC curve, Precision-Recall curve, and CDF of predicted probabilities for positive and negative classes.
    """
    positive_scores = y_pred_proba[y_true == 1]
    negative_scores = y_pred_proba[y_true == 0]
    
    ks_stat, p_value = ks_2samp(positive_scores, negative_scores)
    fpr, tpr, roc_threshold = roc_curve(y_true, y_pred_proba)
    precision, recall, pr_threshold = precision_recall_curve(y_true, y_pred_proba)
    
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    return ks_stat, p_value, roc_auc, pr_auc

def model_plots(y_true, y_pred_proba):
    """Evaluate and plot the ROC curve, Precision-Recall curve, and CDF of predicted probabilities for positive and negative classes, including KS statistic annotation."""

    positive_scores = y_pred_proba[y_true == 1]
    negative_scores = y_pred_proba[y_true == 0]
    
    ks_stat, p_value = ks_2samp(positive_scores, negative_scores)
    fpr, tpr, roc_threshold = roc_curve(y_true, y_pred_proba)
    precision, recall, pr_threshold = precision_recall_curve(y_true, y_pred_proba)
    
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
    ax[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax[0].set_title('ROC Curve')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].legend(loc='lower right')

    prevalance = np.mean(y_true)
    ax[1].plot([0, 1], [prevalance, prevalance], 'k--', label=f'Prevalence = {prevalance:.2f}')
    ax[1].plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})', color='green')
    ax[1].set_title('Precision-Recall Curve')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].legend(loc='upper right')

    cdf_positive = (np.searchsorted(np.sort(y_pred_proba[y_true == 1]), np.sort(y_pred_proba), side='right')) / len(y_pred_proba[y_true == 1])
    cdf_negative = (np.searchsorted(np.sort(y_pred_proba[y_true == 0]), np.sort(y_pred_proba), side='right')) / len(y_pred_proba[y_true == 0])\
    
    ax[2].plot(np.sort(y_pred_proba), cdf_positive, label='Positive Class', color='red')
    ax[2].plot(np.sort(y_pred_proba), cdf_negative, label='Negative Class', color='blue')
    # annotate KS statistic at upper right corner
    ax[2].annotate(f'KS Statistic: {ks_stat:.2f}\np-value: {p_value:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    ax[2].set_title('CDF of Predicted Probabilities')
    ax[2].set_xlabel('Predicted Probability')
    ax[2].set_ylabel('Cumulative Probability')
    ax[2].legend(loc='lower right')
    plt.tight_layout()
    plt.show()
