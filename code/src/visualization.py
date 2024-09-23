import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats 

def plot_stats(x, y, xlab, ylab, savefig=None, less=False):
    """Helper function to make the plots aesthetic and run stats"""

    def overlap_less(second, ovlp):
        """Filter two lists to just the positions where overlap is less than 100"""
        ovlp_new = []
        second_new = []
        for o, s in zip(ovlp, second):
            if o < 1:
                ovlp_new.append(o)
                second_new.append(s) 
        return second_new, ovlp_new
    
    # Convert the input to arrays so we can get a line of best fit 
    if less:
        x, y = overlap_less(x, y)
    x = np.asarray(x)
    y = np.asarray(y)
    a, b = np.polyfit(x, y, 1)
    
    # Plot the points and the line of best fit 
    plt.clf()
    plt.scatter(x, y, color="indigo", alpha=0.2, s=70)
    plt.plot(x, a*x+b, linestyle="-", color="indigo", linewidth=1)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(f"{ylab} vs. {xlab}")
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig, dpi=500)
    
    # Run the statistics 
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    kendall_tau = stats.kendalltau(x, y)
    for name, stat in [("Pearson's R", pearson), ("Spearman", spearman), ("Kendall's Tau", kendall_tau)]:
        print(f"{name}:\t{stat.statistic :.3f},\t p = {stat.pvalue}")

def run_correlation(PATH_TO_DATA, funct1, funct2):
    """Run a single correlation between two functions over the data"""
    xs = []
    ys = []
    langs = []
    for language_family in [l for l in os.listdir(PATH_TO_DATA) if "." not in l]:
        for language in set([f.split(".")[0] for f in os.listdir(f"{PATH_TO_DATA}/{language_family}")]):
            xs.append(funct1(f"{PATH_TO_DATA}/{language_family}/{language}"))
            ys.append(funct2(f"{PATH_TO_DATA}/{language_family}/{language}"))
            langs.append(f"{language_family}/{language}")
    return xs, ys, langs

def run_stats(df, col1, col2):
    """Here's a helper function to run some stats for us"""
    x = df[col1].to_numpy()
    y = df[col2].to_numpy()
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    kendall_tau = stats.kendalltau(x, y)
    for name, stat in [("Pearson's R", pearson), ("Spearman", spearman), ("Kendall's Tau", kendall_tau)]:
        print(f"{name}:\t{stat.statistic :.3f},\t p = {stat.pvalue}")
        
def run_plotting(df, col1, col2, xlabel=None, ylabel=None, title=None):
    """Here's a helper function to handle SNS for us"""
    
    # Scatter plot all the languages together 
    sns.scatterplot(
        data=df, 
        x=col1, 
        y=col2, 
        hue="Family",
        palette=["teal", "indigo"],
        alpha=0.6,
        s=80
    )
    plt.xlabel(col1 if xlabel is None else xlabel)
    plt.ylabel(col2 if ylabel is None else ylabel)
    plt.title(f"{col1} vs. {col2}" if title is None else title)
    plt.show()
    
    
    # Plot the languages separately 
    facet = sns.lmplot(
        data=df, 
        x=col1, 
        y=col2, 
        hue="Family",
        col="Family",
        palette=["teal", "indigo"],
        facet_kws = {"sharex": False},
        ci=0,
    )

    if xlabel is not None:
        facet.axes[0,0].set_xlabel(xlabel)
        facet.axes[0,1].set_xlabel(xlabel)
    if ylabel is not None:
        facet.axes[0,0].set_ylabel(ylabel)
    if title is not None:
        facet.axes[0,1].set_title(f'Uralic {title}')
        facet.axes[0,0].set_title(f'Niger-Congo {title}')

    plt.show()
    
    
    # Run stats
    print("-------------------\nAll Families:\n-------------------")
    run_stats(df, col1, col2)
    print("\n-------------------\nNiger-Congo:\n-------------------")
    run_stats(df.query("Family == 'niger_congo'"), col1, col2)
    print("\n-------------------\nUralic:\n-------------------")
    run_stats(df.query("Family == 'uralic'"), col1, col2)
