import matplotlib.pyplot as plt
from scipy import stats 
import numpy as np

def plot_stats(x, y, xlab, ylab, savefig=None, less=False):
    """Helper function to make the plots aesthetic and run stats"""
    
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
