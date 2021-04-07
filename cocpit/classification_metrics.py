'''
calculation and plotting functions for reporting performance metrics
'''

from cocpit.data_loaders import get_val_loader_predictions
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_confusion_matrix(all_preds, all_labels, class_names,
                          norm, save_name, save_fig=False):
    '''
    Plot and save a confusion matrix from a saved validation dataloader
    Params
    ------
    - all_preds (list): list of predictions from the model for all batches
    - all_labels (list): actual labels (correctly hand labeled)
    - class_names (list): list of strings of classes
    - norm (bool): whether or not to normalize the conf matrix (for unbalanced classes)
    - save_name (str): plot filename to save as 
    - save_fig (bool): save the conf matrix to file
    '''        
    fig, ax = plt.subplots(figsize=(13,9))
    cm = confusion_matrix(all_preds, all_labels)
    
    if norm:
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        heat = sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=class_names,
                           yticklabels=class_names,
                           cmap="Blues", annot_kws={"size": 14})
        plt.title('Normalized')
    else:
        heat = sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                           yticklabels=class_names,
                           cmap="Blues", annot_kws={"size": 16})
        plt.title('Unweighted')
        
    plt.ylabel('Actual Labels', fontsize=20)
    plt.xlabel('Predicted Labels', fontsize=20);
    heat.set_xticklabels(heat.get_xticklabels(), rotation=90, fontsize=18)
    heat.set_yticklabels(heat.get_xticklabels(), rotation=0, fontsize=18)
    if save_fig:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    
    
def plot_model_metric_folds(metric_filename, convert_names, save_name,
                            plot_classes=True, save_fig=False):
    '''
    Plot each model w.r.t. precision, recall, and f1-score
    Params
    ------
    metric_filename (str): holds the csv file of metric scores per fold and model
    convert_names (dict): keys: model names used during training, 
                    values: model names used for publication (capitalized and hyphenated)
    plot_classes (bool): keeps metrics in terms of each class from the
                    clf report in the df for plotting
                    if False, only use macro avg across classes
    - save_name (str): plot filename to save as 
    - save_fig (bool): save the figure to file
    '''
    
    fig, ax = plt.subplots(figsize=(13,9))
    df = pd.read_csv(metric_filename)
    df.columns.values[0] = 'class'
    df.replace(convert_names, inplace=True)
    
    if plot_classes:
        df = df[(df['class'] != 'accuracy') & (df['class'] != 'macro avg')
                & (df['class'] != 'weighted avg')]
    else:
        df = df[(df['class'] == 'macro avg')]
        
    dd=pd.melt(df, id_vars=['model'],
               value_vars=['precision', 'recall', 'f1-score'], var_name='Metric')
    
    g = sns.boxplot(x='model', y='value', data=dd, hue='Metric')
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set_xlabel("Model")
    g.set_ylabel("Value")
    plt.legend(loc='lower right')
    plt.setp(ax.get_legend().get_texts(), fontsize='14') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='16') # for legend title

    g.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
    if plot_classes:
        g.set_ylim(0.70, 1.001)
    else:
        g.set_ylim(0.885, 1.001)
    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_classification_report_classes(clf_report, save_name, save_fig=False):
    '''
    plot precision, recall, and f1-score for each class from 1 model
    average across folds
    also includes accuracy, macro avg, and weighted avg total
    
    Params
    ------
    - clf_report: classification report from sklearn 
        or from metrics_report() above
    - save_name (str): plot filename to save as 
    - save_fig (bool): save the figure to file
    '''
    fig, ax = plt.subplots(figsize=(9,7))
    
    
    
    clf_report = pd.DataFrame(clf_report).iloc[:-1, :].T  
    print(clf_report[clf_report['model'] == 'vgg16'])
    
    g = sns.heatmap(clf_report, annot=True, cmap='coolwarm',
            linecolor='k', linewidths=1, annot_kws={"fontsize": 14}, vmin=0.90, vmax=1.00)
    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()