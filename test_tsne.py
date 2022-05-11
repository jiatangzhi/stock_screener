import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from stock_data import get_list


def tsne():

    """

    Return clusters of the different industries.

    """

    tsne = TSNE(n_components=2,      # Dimension of the embedded space.
                verbose=1,           # It produces lots of logging output.
                perplexity=40,       # Numbers of iterations to converge.
                n_iter=1000)         # Number of iterations run: default 1000.

    tsne_results = tsne.fit_transform(get_list())
    #print(tsne_results)

    df_subset = pd.DataFrame.from_dict({})
    df_subset["AAP", "AMZN", "APTV]", "AZO", "BBWI", "BBY","BKNG", "BWA", "CCL", "CMG", "CZR", "DG", "DHI", "DLTR", "DPZ",  \
              "DRI", "EBAY", "ETSY", "EXPE", "F", "GM", "GPC", "GRMN", "HAS", "HD", "HLT", "KMX", "LEN", "LKQ", "LOW", "LVS",      \
              "MAR", "MCD", "MGM", "MHK", "NCLH", "NKE", "NVR", "NWL", "ORLY", "PENN", "PHM", "POOL", "PVH", "RCL", "RL", "ROST",  \
              "SBUX", "TGT", "TJX", "TPR", "TSCO", "TSLA", "UA", "UAA", "ULTA", "VFC", "WHR", "WYNN", "YUM"] = consumer_discretionary

    df_subset["ADM", "CAG", "CHD", "CL", "CLX", "COST", "CPB", "EL", "GIS", "HRL", "HSY", "K", "KHC", "KMB", "KO", "KR",    \
              "LW", "MDLZ", "MKC", "MNST", "MO", "PEP", "PG", "PM", "SJM","STZ", "SYY", "TAP", "TSN", "WBA", "WMT"] = consumer_staples

    df_subset["APA", "BKR", "COP", "CTRA", "CVX", "DVN", "EOG", "FANG", "HAL", "HES", "KMI", "MPC", "MRO", "OKE", "OXY",    \
              "PSX", "PXD", "SLB", "VLO", "WMB", "XOM"] = energies

    df_subset["AFL", "AIG", "AIZ", "AJG", "ALL", "AMP", "AON", "AXP", "BAC", "BEN", "BK", "BLK", "BRO", "C", "CB", "CBOE",  \
              "CFG", "CINF", "CMA", "CME", "COF", "DFS", "FDS", "FITB", "FRC", "GL", "GS", "HBAN", "HIG", "ICE", "IVZ",     \
              "JPM", "KEY", "L", "LNC", "MCO", "MET", "MKTX", "MMC", "MS", "MSCI", "MTB", "NDAQ", "NTRS", "PFG", "PGR",     \
              "PNC", "PRU", "RE", "RF", "RJF", "SBNY", "SCHW", "SIVB", "SPGI", "STT", "SYF", "TFC", "TROW", "TRV", "USB",   \
              "WFC", "WRB", "WTW", "ZION"] = financials

    df_subset["A", "ABBV", "ABC", "ABMD", "ABT", "ALGN", "AMGN", "ANTM", "BAX", "BDX", "BIIB", "BIO", "BMY", "BSX", "CAH",  \
              "CERN", "CI", "CNC", "COO", "CRL", "CTLT", "CVS", "DGX", "DHR", "DVA", "DXCM", "EW", "GILD", "HCA", "HOLX",   \
              "HSIC", "HUM", "IDXX", "ILMN", "INCY", "IQV", "ISRG", "JNJ", "LH", "LLY", "MCK", "MDT", "MOH", "MRK", "MRNA", \
              "MTD", "OGN", "PFE", "PKI", "REGN", "RMD", "STE", "SYK", "TECH", "TFX", "TMO", "UHS", "UNH", "VRTX", "VTRS",  \
              "WAT", "WST", "XRAY", "ZBH", "ZTS"] = health_care

    df_subset["AAL", "ALK", "ALLE", "AME", "AOS", "BA", "CARR", "CAT", "CHRW", "CMI", "CPRT", "CSX", "CTAS", "DAL", "DE",   \
              "DOV", "EFX", "EMR", "ETN", "EXPD", "FAST", "FBHS", "FDX", "FTV", "GD", "GE", "GNRC", "GWW", "HII", "HON",    \
              "HWM", "IEX", "IR", "ITW", "J", "JBHT", "JCI", "LDOS", "LHX", "LMT", "LUV", "MAS", "MMM", "NDSN", "NLSN",
              "NOC", "NSC","ODFL", "OTIS", "PCAR", "PH", "PNR", "PWR", "RHI", "ROK", "ROL", "ROP", "RSG", "RTX", "SNA",     \
              "SPGI", "SWK", "TDG", "TT", "TXT", "UAL", "UNP", "UPS", "URI", "VRSK", "WAB", "WM", "XYL"] = industrials

    df_subset["AEE", "AEP", "AES", "ATO", "AWK", " CEG", "CMS", "CNP", "D", "DTE", "DUK", "ED", "EIX", "ED", "EIX", "ES",   \
              "ETR", "EVRG", "EXG", "FE", "LNT","NEE", "NI", "NRG", "PEG", "PNW", "PPL", "SO", "SRE", "WEC", "XEL"] = utilities

    df_subset["AMT", "ARE", "AVB", "BXP", "CBRE", "CCI", "CPT", "DLR", "DRE", "EQIX", "EQR", "ESS", "EXR", "FRT", "HST",    \
              "IRM", "KIM", "MAA", "O", "PEAK", "PLD", "PSA", "REG", "SBAC", "SPG", "UDR", "VNO", "VTR", "WELL", "WY"] = real_estate
    #df_subset = pd.DataFrame.from_dict({})
    df_subset['X'] = tsne_results[:, 0]
    df_subset['Y'] = tsne_results[:, 1]
    sns.scatterplot(x="X", y="Y", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 3),
                    data=df).set(title="Stock Market T-SNE projection")
    #print(df_subset)
    #grad = df_subset.eval("X / Y").rename("grad")

    #sns.scatterplot(
        #palette=sns.color_palette("hls", 10),
    #    data=df_subset,
    #    x='X',
    #    y="Y",
    #    hue = grad
    #)

    plt.savefig("/Users/mcarmentz/Desktop/stock_screener/figures/industries.png")


#def get_sector():

 #   df["X"] = z[:, 0]
   # df["Y"] = z[:, 1]

def test_run():
    """Function called by Test Run."""
    tsne()

if __name__ == "__main__":
    test_run()




