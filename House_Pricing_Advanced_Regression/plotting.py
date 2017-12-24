import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from main import load_data, clean_data

# Plotting Style
plt.style.use('fivethirtyeight')

def plot_correlation_matrix(df):
    """
    Plot correlation matrix of feature variables.

    Args:
        df: training dataframe

    Returns:
        Plotted correlation matrix of df features.
    """
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(12,8))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                yticklabels=True, cbar_kws={'shrink':.5})
    plt.title('Correlation Matrix')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.show()

def scatter_matrix(df):
    """
    Plot scatter matrix of the top six most correlated features with sale price.

    Args:
        df: training dataframe

    Returns:
        Plotted scatter matrix of top six most correlated features with sale price.
    """
    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Top Six Most Correlated Features w/ Sale Price', fontsize=20)
    ax1 = sns.regplot(x='OverallQual', y='SalePrice', data=df, fit_reg=False,
                        ax=axs[0, 0], color='steelblue', scatter_kws={"s": 7})
    ax1.set_xlabel('Quality of House', size=10)
    ax1.set_ylabel('Sale Price', size=10)
    ax2 = sns.regplot(x='GrLivArea', y='SalePrice', data=df, fit_reg=False,
                        ax=axs[0, 1], color='steelblue', scatter_kws={"s": 5})
    ax2.set_xlabel('Above Ground Square Feet', size=10)
    ax2.set_ylabel('Sale Price', size=10)
    ax3 = sns.regplot(x='GarageCars', y='SalePrice', data=df, fit_reg=False,
                        ax=axs[0, 2], color='steelblue', scatter_kws={"s": 7})
    ax3.set_xlabel('Garage Car Capacity', size=10)
    ax3.set_ylabel('Sale Price', size=10)
    ax4 = sns.regplot(x='GarageArea', y='SalePrice', data=df, fit_reg=False,
                        ax=axs[1, 0], color='steelblue', scatter_kws={"s": 5})
    ax4.set_xlabel('Garage Square Feet', size=10)
    ax4.set_ylabel('Sale Price', size=10)
    ax5 = sns.regplot(x='TotalBsmtSF', y='SalePrice', data=df, fit_reg=False,
                        ax=axs[1, 1], color='steelblue', scatter_kws={"s": 5})
    ax5.set_xlabel('Total Basement Square Feet', size=10)
    ax5.set_ylabel('Sale Price', size=10)
    ax6 = sns.regplot(x='1stFlrSF', y='SalePrice', data=df, fit_reg=False,
                        ax=axs[1, 2], color='steelblue', scatter_kws={"s": 5})
    ax6.set_xlabel('First Floor Square Feet', size=10)
    ax6.set_ylabel('Sale Price', size=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def saleprice_dist(df):
    """
    Plot distribution of 'SalePrice' before and after log-transformation.

    Args:
        df: training dataframe

    Returns:
        Plotted distributions of 'SalePrice' before and after log-transformation.
    """
    fig, axs = plt.subplots(ncols=2, figsize=(14,5))
    fig.suptitle('Sale Price Distribution', fontsize=20)
    ax1 = sns.distplot(df['SalePrice'], kde=True, bins=75, ax=axs[0],
                        color='steelblue')
    ax1.set_xlabel('Normal Sale Price')
    ax1.set_ylabel('Frequency', size=12)
    ax2 = sns.distplot(df['SalePrice_Log'], kde=True, bins=75, ax=axs[1],
                        color='steelblue')
    ax2.set_xlabel('Log Transformed Sale Price')
    ax2.set_ylabel('Frequency', size=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__=='__main__':
    #Load Data
    train, test = load_data()

    # Plot Correlation Matrix of all features
    train = clean_data(train, dummy=False)
    # plot_correlation_matrix(train)
    scatter_matrix(train)

    # Log-Transform SalePrice
    train['SalePrice_Log'] = np.log1p(train["SalePrice"])
    saleprice_dist(train)
