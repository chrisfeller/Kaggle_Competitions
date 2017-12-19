import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting Style
plt.style.use('fivethirtyeight')

def eda_plots(df):
    """
    Plots all EDA visuals from uncleaned Titanic training data.

    Args:
        df: DataFrame to plot.

    Returns:
        Seven EDA plots.
    """
    barplot(df)
    swarmplot(df)
    side_by_side(df)
    pointplot(df)
    violinplot_Age(df)
    violinplot_Fare(df)
    plot_correlation_matrix(df)

def barplot(df):
    """
    Barplot of Survival by Sex and Class of Titanic passengers.

    Args:
        df: DataFrame to plot.

    Returns:
        Barplot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df, palette='coolwarm')
    leg_handles = ax.get_legend_handles_labels()[0]
    ax.legend(leg_handles, ['First', 'Second', 'Third'], title='Class')
    plt.title('Survival by Sex and Class')
    plt.show()

def swarmplot(df):
    """
    Swarmplot of Survival by Sex and Age of Titanic passengers.

    Args:
        df: DataFrame to plot.

    Returns:
        Swarmplot
    """
    fig, ax = plt.subplots(figsize=(10,8))
    sns.swarmplot(x='Sex', y='Age', hue='Survived', data=df)
    plt.title('Survival by Sex and Age')
    plt.show()

def side_by_side(df):
    """
    Two Violin plots of Survival based on # of Siblings/Spouses Aboard the Titanic
    and # of Parents/Children Aboard

    Args:
        df: DataFrame to plot.
    Returns:
        Two violin plots.
    """
    fig, axs = plt.subplots(ncols=2, figsize=(10,8))
    fig.suptitle('Survival by Party Size', fontsize=20)
    ax1 = sns.violinplot(x='Survived', y='SibSp', data=df, ax=axs[0], palette='coolwarm')
    ax1.set_ylabel('# of Siblings/Spouses Aboard')
    ax2 = sns.violinplot(x='Survived', y='Parch', data=df, ax=axs[1], palette='coolwarm')
    ax2.set_ylabel('# of Parents/Children Aboard')
    plt.show()

def pointplot(df):
    """
    Pointplot of Survival by Embarkment location.

    Args:
        df: DataFrame to plot.
    Returns:
        Pointplot
    """
    fig, ax = plt.subplots(figsize=(10,8))
    ax = sns.pointplot(x='Sex', y='Survived', hue='Embarked', data=df,
                        palette='coolwarm', ci=90)
    leg_handles = ax.get_legend_handles_labels()[0]
    ax.legend(leg_handles, ['Cherbourg', 'Queenstown', 'Southampton'],
                title='Embarkment', loc='upper left')
    plt.title('Survival by Sex and Embarkment')
    plt.show()

def violinplot_Age(df):
    """
    Violin plot of Survival by Age.

    Args:
        df: DataFrame to plot.

    Returns:
        Violin Plot
    """
    fig, ax = plt.subplots(figsize=(10,8))
    sns.violinplot(x='Survived', y='Age', data=df, palette='coolwarm')
    plt.title('Survival by Age')
    plt.show()

def violinplot_Fare(df):
    """
    Violin plot of Survival by Fare.

    Args:
        df: DataFrame to plot.

    Returns:
        Violin Plot
    """
    fig, ax = plt.subplots(figsize=(10,8))
    sns.violinplot(x='Survived', y='Fare', data=df, orient='v', palette='coolwarm')
    plt.title('Survival by Fare')
    plt.show()

def plot_correlation_matrix(df):
    """
    Plot correlation matrix of feature variables.

    Args:
        df: DataFrame to plot.

    Returns:
        Plotted correlation matrix of df features.
    """
    corr = df[[col for col in df.columns if col != 'PassengerId']].corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(10,8))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                cbar_kws={'shrink':.5}, annot=True, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

def distribution_plot(df):
    df = df.loc[:,['Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',
                'Embarked_S','Pclass_2', 'Pclass_3']]
    fig, axes = plt.subplots(figsize = (11,9))
    sns.violinplot(data=df, ax = axes, scale = 'width', palette="coolwarm", cut = 0)
    axes.set_title('Violin Plot')

    axes.yaxis.grid(True)
    for tick in axes.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")
    axes.set_xlabel('Features')
    axes.set_ylabel('Range')
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    pass
