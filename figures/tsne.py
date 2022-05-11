import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from stock_data import get_list


def tsne():
    """

    It has the same functionality as k_means()

    """

    tsne = TSNE(n_components=2,      # Dimension of the embedded space = 2.
                verbose=1,           # It produces lots of logging output.
                perplexity=75,       # Numbers of iterations to converge.
                n_iter=1000)         # Number of iterations run: default 1000.

    tsne_results = tsne.fit_transform(get_list())
    #print(tsne_results)

    df_subset = pd.DataFrame.from_dict({})
    df_subset['X'] = tsne_results[:, 0]
    df_subset['Y'] = tsne_results[:, 1]
    #print(df_subset)
    grad = df_subset.eval("X / Y").rename("grad")

    sns.scatterplot(
        #palette=sns.color_palette("hls", 10),
        data=df_subset,
        x='X',
        y="Y",
        hue = grad
    )

    plt.savefig("/Users/mcarmentz/Desktop/stock_screener/figures/tsne.png")    # Figures are saved in figures directory.

def test_run():
    """Function called by Test Run."""
    tsne()

if __name__ == "__main__":
    test_run()



