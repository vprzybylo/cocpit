'''
plotting functions
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_batch_size_accuracy(val_csv, save_name, save_fig=False):

    sns.set_style("whitegrid")
    df_val = pd.read_csv(val_csv, names=["Model", "Epoch", "Batch Size", "Accuracy", "Loss"])
    df_val['Accuracy']=df_val['Accuracy']*100
    g=sns.barplot(x="Batch Size", y="Accuracy", data=df_val, ci=None, palette="dark")
    g.set_xlabel("Batch Size", fontsize=14, fontfamily='serif');
    g.set_ylabel("Accuracy [%]", fontsize=14, fontfamily='serif');
    g.set_ylim(80,100)
    g.set_xticks([0,1,2,3,4])
    g.set_yticks([80,85,90,95,100])
    g.set_xticklabels([64, 128, 256, 512, 1024], rotation=90, fontsize=14, fontfamily='serif');
    g.set_yticklabels(g.get_yticks(), size = 14, fontfamily='serif')
    g.grid(False)
    #plot text above bar
    groupedvalues = df_val.groupby('Batch Size')['Accuracy'].max()
    for p in g.patches:
        g.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=12,fontfamily='serif', color='k', xytext=(0, 8),
                     textcoords='offset points')

    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        
        
def plot_train_val_acc_loss(model_names, num_models, num_epochs, train_accs, colors, new_names,
                            val_accs, train_losses, val_losses, save_name, save_fig):
    sns.set_style("whitegrid")
    plt_params = {'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'legend.title_fontsize': 12}
    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update(plt_params)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13,8), sharex=True, sharey=True)


    #fig = plt.figure(figsize=(20,20))
    ax1 = plt.subplot(2, 2, 1)

    for i in range(num_models):
        ax1.scatter(np.arange(1,(num_epochs+1)), [i*100 for i in train_accs[i,:]],
                    c=colors[new_names[i]], marker='o', s=35, label=new_names[i])
    plt.ylabel("Accuracy [%]")
    plt.ylim(40,100)
    plt.xlim(0,num_epochs+1)
    ax1.legend(title='Model type:', loc='lower right', prop={'size': 12}, ncol=2)
    #ax1.axes.xaxis.set_ticks([])
    ax1.yaxis.set_ticks_position('both')
    ax1.minorticks_on()
    plt.xticks(np.arange(0,num_epochs+1, 5))
    ax1.tick_params(axis='y', which='minor', direction='out')
    #ax1.xaxis.set_tick_params(which='minor', bottom=False)
    ax1.title.set_text('Training Data')

    #fig = plt.figure(figsize=(20,5))
    ax2 = plt.subplot(2, 2, 2)
    for i in range(num_models):
        ax2.scatter(np.arange(1,(num_epochs+1)),
                    [i*100 for i in val_accs[i,:]], c=colors[new_names[i]],
                    marker='*', s=55, label=model_names[i])
    plt.ylim(40,100)
    plt.xlim(0,num_epochs+1)
    plt.xticks(np.arange(0,num_epochs+1, 5))
    #ax2.legend(title='Model type:', loc='best', prop={'size': 10})
#     ax2.axes.yaxis.set_ticks([])
#     ax2.axes.xaxis.set_ticks([])
#     ax2.yaxis.set_ticks_position('both')
    #ax2.minorticks_on()
    ax2.tick_params(axis='y', which='minor', direction='out')
    ax2.xaxis.set_tick_params(which='minor', bottom=False)
    ax2.title.set_text('Validation Data')

    ax3 = plt.subplot(2, 2, 3)
    for i in range(num_models):
        ax3.scatter(np.arange(1,(num_epochs+1)), [i for i in train_losses[i,:]],
                    c=colors[new_names[i]], marker='o', s=35, label=model_names[i])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #ax3.legend(title='Model type:', loc='best', prop={'size': 10})
    plt.ylim(0,2.4)
    plt.xlim(0,num_epochs+1)
    plt.xticks(np.arange(0,num_epochs+1, 5))
    plt.tight_layout()
    ax3.yaxis.set_ticks_position('both')
    ax3.minorticks_on()
    ax3.tick_params(axis='y', which='minor', direction='out')
    ax3.xaxis.set_tick_params(which='minor', bottom=False)

    ax4 = plt.subplot(2, 2, 4)
    for i in range(num_models):
        ax4.scatter(np.arange(1,(num_epochs+1)), [i for i in val_losses[i,:]],
                    c=colors[new_names[i]], marker='*', s=55, label=model_names[i])
    plt.xlabel("Epoch")
    #ax4.legend(title='Model type:', loc='best', prop={'size': 10})
    plt.ylim(0,2.4)
    plt.xlim(0,num_epochs+1)
    plt.xticks(np.arange(0,num_epochs+1, 5))
    #ax4.axes.yaxis.set_ticks([])
    plt.tight_layout()
    #ax4.yaxis.set_ticks_position('both')
    ax4.minorticks_on()
    ax4.tick_params(axis='y', which='minor', direction='out')
    ax4.xaxis.set_tick_params(which='minor', bottom=False)
    
    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        
def plot_time_samples(time_csv, save_name, convert_names, save_fig=False):
    sns.set_style("whitegrid")
    df = pd.read_csv(time_csv, names=["Model", "Samples", "Time"])
    df['Model'].astype(str)
    df['Samples'].astype(int)
    df['Time'].astype(float)
    df.replace(convert_names, inplace=True)
    df = df.set_index('Model')
    df = df.loc[['ResNet-18', 'AlexNet', 'ResNet-34', 'Efficient-b0', 'VGG-16',
           'DenseNet-169', 'VGG-19', 'DenseNet-201', 'ResNet-152']]
    df.reset_index(inplace=True)
    
    ax=sns.catplot(data=df, kind="bar", x="Model", y="Time", hue="Samples", legend=True, ci=False);
    ax.set_xticklabels(rotation=90, fontsize=14);
    ax.set(xlabel='Model', ylabel='Time [seconds]');
    
    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")

def plot_model_timing(time_csv, convert_names, colors, save_name, save_fig=False):
    sns.set_style("whitegrid")
    df = pd.read_csv(time_csv)
    df.replace(convert_names, inplace=True)
    df['Model'].astype(str)
    df['Time'].astype(float)
    df = df.sort_values(by=['Time'])
    sorted_colors = dict((k,colors[k]) for k in df['Model'])

    fig, ax = plt.subplots(1,1,figsize = (10,5))

    time = df['Time']/60
    #ax = time.plot(kind='bar')
    g=sns.barplot(x="Model", y=time, data=df, ci=None, palette=sorted_colors.values())
    g.set_xlabel("Model");
    g.set_ylabel("Training Time [minutes]");
    g.set_xticklabels(df['Model'], rotation=90, fontsize=14);
    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches = "tight")