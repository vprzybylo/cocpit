'''
calculation and plotting functions for reporting performance metrics
'''

from cocpit.data_loaders import get_val_loader_predictions

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter

def add_model_fold_to_clf_report(clf_report, fold, model_name):
    '''
    add model name and fold iteration to clf_report
    Params
    ------
    - clf_report: classification report from sklearn
    - fold (int): kfold iteration
    - model_name (str): name of model being trained
    
    Return
    ------
    clf_report (df): tranposed and appened df for model and kfolds
    '''
    
    # transpose classes as columns and convert to df
    clf_report = pd.DataFrame(clf_report).iloc[:-1, :].T  
    
    # add fold iteration and model name
    clf_report['fold'] = fold
    clf_report['model'] = model_name
    
    return clf_report
    

def metrics_report(all_labels, all_preds, class_names):
    '''
    Build a text report showing the main classification metrics.
    Params
    -----
    - all_labels (list): 
    - all_preds (list): 
    - class_names (list): list of strings of classes
    
    Returns
    -------
    - clf_report (df): classifcation report from sklearn
    '''
    # sklearn classifcation report outputs metrics in dictionary
    clf_report = classification_report(all_labels, all_preds,
                                       digits=3, target_names=class_names, output_dict=True)
    
    return clf_report


def plot_confusion_matrix(all_preds, all_labels, class_names, norm=False, save_fig=False):
    '''
    Plot and save a confusion matrix from a saved validation dataloader
    Params
    ------
    - all_preds (list): list of predictions from the model for all batches
    - all_labels (list): actual labels (correctly hand labeled)
    - class_names (list): list of strings of classes
    - norm (bool): whether or not to normalize the conf matrix (for unbalanced classes)
    - save_fig (bool): save the conf matrix to file
    '''        
    fig, ax = plt.subplots(figsize=(13,9))
    cm = confusion_matrix(all_preds, all_labels)
    
    if norm:
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        heat = sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=class_names,
                           yticklabels=class_names, cmap="Blues", annot_kws={"size": 14})
        plt.title('Normalized')
        savename='/data/data/plots/norm_conf_matrix.pdf'
    else:
        heat = sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                           yticklabels=class_names, cmap="Blues", annot_kws={"size": 16})
        plt.title('Unweighted')
        savename='/data/data/plots/conf_matrix.pdf'
        
    plt.ylabel('Actual Labels', fontsize=20)
    plt.xlabel('Predicted Labels', fontsize=20);
    heat.set_xticklabels(heat.get_xticklabels(), rotation=90, fontsize=18)
    heat.set_yticklabels(heat.get_xticklabels(), rotation=0, fontsize=18)
    #plt.savefig('savename', dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_model_metric_folds(metric_filename):
    '''
    Plot each model w.r.t. precision, recall, and f1-score
    Params
    ------
    metric_filename (str): holds the csv file of metric scores per fold and model
    '''
    df_concat = pd.read_csv(metric_filename)
    dd=pd.melt(df_concat, id_vars=['model'],
               value_vars=['precision', 'recall', 'f1-score'], var_name='metric')
    sns.boxplot(x='model', y='value', data=dd, hue='metric')
    plt.show()
    
    
def plot_classification_report_classes(clf_report):
    '''
    plot precision, recall, and f1-score for each class from 1 model
    average across folds
    also includes accuracy, macro avg, and weighted avg total
    
    Params
    ------
    - clf_report: classification report from sklearn 
        or from metrics_report() above
    '''
    fig, ax = plt.subplots(figsize=(9,7))
    
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap='coolwarm',
            linecolor='k', linewidths=1, annot_kws={"fontsize": 14}, vmin=0.86, vmax=1.00)
    plt.show()
