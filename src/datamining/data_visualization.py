from os import getcwd

from datamining.preprocessing import discretize_values

from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np


def dataset_stats(df: pd.DataFrame) -> None:

    features = df.columns.values

    print(df.info())
    print(f'\nDuplicated rows: {df.duplicated().sum()}\n')

    for feat in features:
        if df[feat].dtype == int or df[feat].dtype == float:
            print(df[feat].describe())
            print()

    print('UNIQUE VALUES')
    print(df.nunique())
    print('\nMISSING VALUES')
    print(df.isnull().sum())


def plot_eda_graphs(df: pd.DataFrame) -> None:

    graph1 = df.groupby(('Borough'))[['Price']].mean().reset_index()
    fig1 = px.bar(graph1, x='Borough', y='Price', color='Borough',
                  title="Average Price per Borough", width=800, height=400)
    fig1.write_image(f'{getcwd()}/../plots/avg_price_borough.svg')

    graph2 = df.groupby(('Borough'))[['Days Available']].mean().reset_index()
    fig2 = px.bar(graph2, x='Borough', y='Days Available', color='Borough',
                  title='Average Availability per Borough', width=800, height=400)
    fig2.write_image(f'{getcwd()}/../plots/avg_available_borough.svg')

    graph3 = df.groupby(('Room Type'))[['Price']].mean().reset_index()
    fig3 = px.bar(graph3, x='Room Type', y='Price', color='Room Type',
                  title="Average Price per Room Type", width=600, height=400)
    fig3.write_image(f'{getcwd()}/../plots/avg_price_room.svg')

    graph4 = df.groupby(('Room Type'))[['Days Available']].mean().reset_index()
    fig4 = px.bar(graph4, x='Room Type', y='Days Available', color='Room Type',
                  title="Average Availability per Room Type", width=600, height=400)
    fig4.write_image(f'{getcwd()}/../plots/avg_available_room.svg')

    graph5 = df.groupby('Room Type')[['Price']].sum().reset_index()
    fig5 = px.pie(graph5, values='Price', names='Room Type', title='Total Receipt by Room Type',
                  width=600, height=400)
    fig5.write_image(f'{getcwd()}/../plots/total_receipt.svg')

    graph6 = df.groupby(('Borough'))[['Monthly Reviews']].mean().reset_index()
    fig6 = px.bar(graph6, x='Borough', y='Monthly Reviews', color='Borough',
                  title="Average Montlhy Reviews per Borough", width=800, height=400)
    fig6.write_image(f'{getcwd()}/../plots/avg_review_borough.svg')

    graph7 = df.groupby(['Borough', 'Room Type'])[['Minimum Nights']].mean().reset_index()
    fig7 = px.bar(graph7, x='Borough', y='Minimum Nights', color='Room Type',
                  title="Average Minimum Nights Required", width=800, height=400)
    fig7.write_image(f'{getcwd()}/../plots/avg_minnights.svg')

    graph8 = df.groupby(['Minimum Nights'])[['Price']].mean().reset_index()
    graph8['Price'] = np.log2(graph8['Price'])
    fig8 = px.bar(graph8, x='Minimum Nights', y='Price', title="Average Price by Minimum Nights Required",
                  labels={'Price': 'Price (log 2)'}, width=800, height=400)
    fig8.write_image(f'{getcwd()}/../plots/avg_price_minnights.svg')

    graph9 = df.groupby(['Days Available'])[['Price']].mean().reset_index()
    fig9 = px.bar(graph9, x='Days Available', y='Price', title="Average Price by Availability",
                  width=1200, height=400)
    fig9.write_image(f'{getcwd()}/../plots/avg_price_available.svg')


def plot_price_boxplot(df: pd.DataFrame) -> None:

    df['Price'] = np.log2(df['Price'])
    fig, axes = plt.subplots(figsize=(8, 5))
    sns.boxplot(ax=axes, x='Borough', y='Price', data=df)

    axes.set_title('Price Distribution per Borough')
    axes.set_xlabel('Borough')
    axes.set_ylabel('Price (log 2)')

    plt.savefig(f'{getcwd()}/../plots/price_boxplot.svg', format='svg')
    plt.show()


def plot_price_kde(df: pd.DataFrame) -> None:

    sns.kdeplot(data=df, x="Price")
    plt.title('Price Kernel Density Estimate (KDE)')

    plt.savefig(f'{getcwd()}/../plots/price_distribution.svg', format='svg')
    plt.show()


def plot_wordclouds(df: pd.DataFrame) -> None:

    # Get rows below the global price mean
    low_df = df.query('Price <= @df["Price"].mean()')

    text = ' '.join(str(n).lower() for n in low_df['Name'])
    wordcloud = WordCloud(max_words=100, background_color='black').generate(text)

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Cheap WordCloud')
    plt.axis('off')

    plt.savefig(f'{getcwd()}/../plots/cheap_wordcloud.svg', format='svg')
    plt.show()

    high_df = df.query('Price > @df["Price"].mean()')

    text = ' '.join(str(n).lower() for n in high_df['Name'])
    wordcloud = WordCloud(max_words=100, background_color='black').generate(text)

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Expensive WordCloud')
    plt.axis('off')

    plt.savefig(f'{getcwd()}/../plots/expensive_wordcloud.svg', format='svg')
    plt.show()


def plot_pca(df: pd.DataFrame) -> None:

    # PCA doesn't support missing and categorical values
    pca_df = df.dropna(axis='index')
    for col in pca_df.columns.values:
        if pca_df[col].dtype != float and pca_df[col].dtype != int:
            discretize_values(df=pca_df, column=col)

    var_ratio = list()
    nums = np.arange(len(pca_df.columns.values) + 1)

    std_scaler = StandardScaler()
    pca_df = std_scaler.fit_transform(pca_df)

    for num in nums:
        pca = PCA(n_components=num)
        pca.fit(pca_df)
        var_ratio.append(np.sum(pca.explained_variance_ratio_))

    plt.figure(figsize=(8, 6))
    plt.grid()
    plt.plot(nums, var_ratio, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Dataset PCA')

    plt.savefig(f'{getcwd()}/../plots/pca.svg', format='svg')
    plt.show()


def analyze_investment(df: pd.DataFrame) -> None:

    # Availability median is far lower, but with these conditions, there isn't such property in the dataset
    price_mean = df['Price'].mean()
    availability_mean = df['Days Available'].mean()
    reviews_mean = df['Reviews'].mean()

    # Obtaining {Price, Days Available, Reviews} average by district
    ideal_locations = df.groupby('District')[['Price', 'Days Available', 'Reviews']].mean().reset_index()

    # Queries for districts in which {Price, Days Available, Reviews} are higher than averages
    ideal_locations = ideal_locations.query(
        'Price > @price_mean & `Days Available` < @availability_mean & Reviews > @reviews_mean')
    print(ideal_locations.sort_values(by='Price', ascending=False).to_string(index=False))

    location = ideal_locations.query('District == "Cobble Hill"')
    price_q = stats.percentileofscore(df["Price"], location['Price'].max(), kind='weak')
    available_q = stats.percentileofscore(df["Days Available"], location['Days Available'].max(), kind='weak')
    review_q = stats.percentileofscore(df["Reviews"], location['Reviews'].max(), kind='weak')

    print('\nThe selected district is Cobble Hill in Brooklyn!\n')
    print(f'The price is in the highests {round(100 - price_q, 2)}%')
    print(f'The availability is in the highests {round(available_q, 2)}%')
    print(f'The number of reviews is in the highests {round(100 - review_q, 2)}%')


def plot_correlation_matrix(df: pd.DataFrame, graph_width: int, file_name: str = '') -> None:

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

    plt.savefig(f'{getcwd()}/../plots/{file_name}_correlation_matrix.svg', format='svg')
    plt.show()


def plot_metrics(df: pd.DataFrame) -> None:

    get_scores = df.query('Metric == "R²" | Metric == "MAE"')
    get_scores = get_scores.drop(['Metric'], axis='columns')

    tested_models = list(df.columns.values)
    tested_models.remove('Metric')

    plot_scores = pd.DataFrame()
    plot_scores['Algorithm'] = tested_models
    plot_scores['R²'] = get_scores.iloc[0].values
    plot_scores['MAE'] = get_scores.iloc[1].values

    fig1 = px.bar(plot_scores, x='Algorithm', y='R²', color='Algorithm',
                  title="Average R² with 5-fold CV (Higher is better)", width=800, height=400)
    fig1.update_yaxes(range=[0, 1])
    fig1.write_image(f'{getcwd()}/../plots/r2_per_algorithm.svg')

    fig2 = px.bar(plot_scores, x='Algorithm', y='MAE', color='Algorithm',
                  title="Average MAE with 5-fold CV (Lower is better)", width=800, height=400)
    fig2.write_image(f'{getcwd()}/../plots/mae_per_algorithm.svg')


def plot_learning_curve(attributes, classes, estimators, n_trainings):

    plt.grid()
    plt.title('Learning Curves (5-fold CV)')
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Mean')
    plt.ylim(0, 1)

    i = 0
    plot_colors = ['r', 'g', 'b']
    for est in estimators:
        training_sizes, _, cv_scores = learning_curve(estimators[est],
                                                      attributes, classes,
                                                      train_sizes=np.linspace(0.1, 1, n_trainings),
                                                      cv=5, scoring='r2', n_jobs=-1)
        cv_scores_mean = np.mean(cv_scores, axis=1)
        plt.plot(training_sizes, cv_scores_mean, color=plot_colors[i], label=est)
        i += 1

    plt.legend(loc='best')
    plt.savefig(f'{getcwd()}/../plots/learning_curve.svg', format='svg')
    plt.show()


def plot_selection_graph(rfecv: any) -> None:

    n_scores = len(rfecv.cv_results_['mean_test_score'])
    plt.figure()
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Average R²')
    plt.ylim(0, 1)
    plt.errorbar(
        range(1, n_scores + 1),
        rfecv.cv_results_['mean_test_score'],
        yerr=rfecv.cv_results_['std_test_score'],
    )
    plt.title('Recursive Feature Selection with CV')

    plt.savefig(f'{getcwd()}/../plots/feature_selection.svg', format='svg')
    plt.show()


def plot_feature_importance(attributes: any, f_importances: any, alg_name: str) -> None:

    feature_dataframe = pd.DataFrame({'Feature': attributes, 'Feature Relevance': f_importances})

    feature_data = pd.DataFrame({'Feature': attributes, 'Feature Relevance': feature_dataframe['Feature Relevance']})
    feature_data = feature_data.sort_values(by='Feature Relevance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.title(f'Average Feature Importance ({alg_name})', fontsize=16)

    s = sns.barplot(y='Feature', x='Feature Relevance', data=feature_data, orient='h', palette='coolwarm')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    s.set_xticklabels(s.get_xticklabels(), rotation=90)

    plt.savefig(f'{getcwd()}/../plots/feature_importance({alg_name}).svg', format='svg')
    plt.show()
