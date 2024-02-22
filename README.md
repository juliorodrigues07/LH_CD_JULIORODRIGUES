[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=ffffff)](https://jupyter.org/)
[![Python 3.10.12](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3106/)
[![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://www.linux.org/)
# Rent Pricing

- EDA on rent pricing at NY (New York) boroughs with interactive dashboards, along with development of a ML regression model.

- If you want to see the deployed application, click down below and feel free to test the models with your own instances, interact with dynamic dashboards about the dataset or visualize static ones:

     - **Deploy**: [![Deploy](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)](https://rent-pricing.streamlit.app/)
 
# 1. Requirements

- [Python3](https://python.org) and [pip](https://pip.pypa.io/en/stable/installation/) package manager:

      sudo apt install python3 python3-pip build-essential python3-dev
 
- [virtualenv](https://virtualenv.pypa.io/en/latest/) tool:

      pip install virtualenv

- Libraries: [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/stable/index.html), [mlxtend](https://rasbt.github.io/mlxtend/), [xgboost](https://xgboost.readthedocs.io/en/stable/), [lightgbm](https://lightgbm.readthedocs.io/en/stable/), [Streamlit](https://streamlit.io/), [Dash](https://dash.plotly.com/), [Plotly express](https://plotly.com/python/plotly-express/), [Kaleido](https://github.com/plotly/Kaleido), [Matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/), [numpy](https://numpy.org/), [WordCloud](https://amueller.github.io/word_cloud/), [Cerberus](https://docs.python-cerberus.org/), [joblib](https://joblib.readthedocs.io/en/latest/index.html), [gdown](https://pypi.org/project/gdown/);

- Environments: [Jupyter](https://jupyter.org/).

# 2. Screens

In this section, you can see the interactive and static dashboards screens made with Streamlit, as well as the predictor GUI.

## 2.1. Price Predictor
![Predictor](/assets/predictor.png)

## 2.2. Interactive Dashboard
![Interactive](/assets/interactive.png)

## 2.3. Static Dashboard
![Static](/assets/static.png)
  
# 3. Execution

1. Clone the repository

       git clone -b extras git@github.com:juliorodrigues07/LH_CD_JULIORODRIGUES.git

2. Unzip the repository

       unzip LH_CD_JULIORODRIGUES-extras.zip && cd LH_CD_JULIORODRIGUES-extras

2. Create a virtual environment

       python3 -m venv .venv

3. Activate the virtual environment

       source .venv/bin/activate

4. Install the dependencies

       pip install -r requirements.txt

## 3.1. Predictor and Dashboards

- You first need to be in the _dashboards_ directory to run the commands.

     - With Streamlit:
     
           streamlit run 1_🏠_Home.py
     
     - With Dash Plotly (only dashboard):
     
           python3 dash_test.py

## 3.2. Data Mining

- To visualize the notebooks online and run them ([Google Colaboratory](https://colab.research.google.com/)), click on the following links:
    -  [EDA](https://colab.research.google.com/github/juliorodrigues07/LH_CD_JULIORODRIGUES/blob/extras/notebooks/1_eda.ipynb);
    -  [Preprocessing](https://colab.research.google.com/github/juliorodrigues07/LH_CD_JULIORODRIGUES/blob/extras/notebooks/2_preprocessing.ipynb);
    -  [Machine Learning](https://colab.research.google.com/github/juliorodrigues07/LH_CD_JULIORODRIGUES/blob/extras/notebooks/3_ml_methods.ipynb).

- To run the notebooks locally, run the commands in the _notebooks_ directory following the template: `jupyter notebook <file_name>.ipynb`.
  
    - EDA (Exploratory Data Analysis):

          jupyter notebook 1_eda.ipynb

    - Preprocessing:

          jupyter notebook 2_preprocessing.ipynb

    - Machine Leaning:

          jupyter notebook 3_ml_methods.ipynb

- To run python scripts locally, you first need to be in the _src_ directory and then run the command:

      python3 main.py

# 4. Project Structure

    .
    ├── README.md                       # Project's documentation
    ├── requirements.txt                # File containing all the required dependencies to run the project
    ├── plots                           # Directory containing all the graph plots generated in EDA
    ├── assets                          # Directory containing images used in README.md and in the deployed app
    ├── notebooks                       # Directory containing project's jupyter notebooks
    |   ├── 1_eda.ipynb
    |   ├── 2_preprocessing.ipynb
    |   └── 3_ml_methods.ipynb
    ├── dashboards                      # Directory containing the web application
    |   ├── 1_🏠_Home.py                <- Main page with the price predictor
    |   ├── pages                       # Child pages directory
    |   |   ├── 2_📈_Interactive.py     <- Script responsible for generating the interactive dashboards
    |   |   └── 3_📊_Static.py          <- Script responsible for generating the static dashboards
    |   └── dash_test.py                <- Interactive and static dashboards made with Dash library
    ├── src                             # Directory containing all the python scripts for data mining
    |   ├── main.py                     <- Main script for evaluating ML models
    |   └── datamining                  # Directory containing scripts responsible for all KDD process
    |       ├── data_visualization.py
    |       ├── preprocessing.py
    |       ├── ml_methods.py
    |       └── __init__.py
    ├── datasets                        # Directory containing all used or generated datasets in the project
    |   ├── pricing.csv                 <- Original dataset
    |   ├── reduced.parquet             <- Result after applying memory optimizing techniques on the original dataset
    |   ├── filled.parquet              <- Result after inputting missing values in the reduced.parquet dataset
    |   ├── preprocessed.parquet        <- Result after applying preprocessing techniques on the filled.parquet dataset
    |   └── feature_selected.parquet    <- Final result after applying feature selection on the preprocessed.parquet dataset
    └── models                          # Directory containing all generated models in the project
        ├── lgbm_model.pkl              <- LightGBM algorithm fitted model
        ├── xgb_model.pkl               <- XGBoost algorithm fitted model
        └── histgb_model.pkl            <- HistGradientBoosting algorithm fitted model



# 5. Outro

- To uninstall all dependencies, run the following command:

      pip uninstall -r requirements.txt -y

- To deactivate the virtual environment, run the following command:

      deactivate

- To delete the virtual environment, run the following command:

      rm -rf .venv
