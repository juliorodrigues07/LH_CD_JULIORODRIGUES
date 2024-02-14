import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os import getcwd


def plot_correlation_matrix(df: pd.DataFrame, graph_width: int) -> None:

    # Remove rows which have any missing value in its features
    df = df.dropna(axis='index')
    df = df[[col for col in df if df[col].nunique() > 1]]

    if df.shape[1] < 2:
        return

    corr = df.corr()
    plt.figure(num=None, figsize=(graph_width, graph_width), dpi=80, facecolor='w', edgecolor='k')
    corr_mat = plt.matshow(corr, fignum=1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corr_mat)

    # -1: weak relationship | 0: neutral relationship | 1: strong relationship
    plt.clim(-1, 1)
    plt.title(f'Correlation Matrix', fontsize=16)

    # plt.savefig(f'{getcwd()}/../plots/correlation_matrix.pdf', format='pdf')
    plt.show()


def plot_feature_importance(cols, attributes, f_importances) -> None:

    feature_dataframe = pd.DataFrame({'Feature': attributes, 'Feature Relevance': f_importances})
    # feature_dataframe['Mean'] = feature_dataframe.mean(axis=1)

    feature_data = pd.DataFrame({'Feature': cols, 'Feature Relevance': feature_dataframe['Feature Relevance']})
    # feature_data = pd.DataFrame({'Feature': cols, 'Feature Relevance': feature_dataframe['Mean'].values})
    feature_data = feature_data.sort_values(by='Feature Relevance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.title('Average Feature Importance (LightGBM Regressor)', fontsize=16)

    s = sns.barplot(y='Feature', x='Feature Relevance', data=feature_data, orient='h', palette='coolwarm')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    s.set_xticklabels(s.get_xticklabels(), rotation=90)

    # plt.savefig(f'{getcwd()}/../plots/features/feat_importance(XGBoost).svg', format='svg')
    plt.show()
