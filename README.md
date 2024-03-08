[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=ffffff)](https://jupyter.org/)
[![Python 3.10.12](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3106/)
[![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://www.linux.org/)
# Rent Pricing

- EDA on rent pricing at NY (New York) boroughs with interactive dashboards, along with development of a ML regression model.

- If you want to see the deployed application, click down below and feel free to test the models with your own instances, interact with dynamic dashboards about the dataset or visualize static ones:

     - **Deploy**: [![Deploy](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)](https://ny-rent-pricing.streamlit.app/)
 
# 1. Requirements

- [Python3](https://python.org) and [pip](https://pip.pypa.io/en/stable/installation/) package manager:

      sudo apt install python3 python3-pip build-essential python3-dev
 
- [virtualenv](https://virtualenv.pypa.io/en/latest/) tool:

      pip install virtualenv

- Libraries: [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/stable/index.html), [xgboost](https://xgboost.readthedocs.io/en/stable/), [lightgbm](https://lightgbm.readthedocs.io/en/stable/), [Streamlit](https://streamlit.io/), [Plotly express](https://plotly.com/python/plotly-express/), [numpy](https://numpy.org/), [Cerberus](https://docs.python-cerberus.org/), [joblib](https://joblib.readthedocs.io/en/latest/index.html), [gdown](https://pypi.org/project/gdown/);

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

       git clone -b deploy https://github.com/juliorodrigues07/ny_rent_pricing.git

2. Enter the repository's directory

       cd ny_rent_pricing

2. Create a virtual environment

       python3 -m venv .venv

3. Activate the virtual environment

       source .venv/bin/activate

4. Install the dependencies

       pip install -r requirements.txt

5. You first need to be in the _dashboards_ directory and  then to run the command:
     
       streamlit run 1_🏠_Home.py

# 4. Project Structure

    .
    ├── README.md                       # Project's documentation
    ├── requirements.txt                # File containing all the required dependencies to run the project
    ├── assets                          # Directory containing images used in README.md and in the deployed app
    └── dashboards                      # Directory containing the web application
        ├── 1_🏠_Home.py                <- Main page with the price predictor
        └── pages                       # Child pages directory
            ├── 2_📈_Interactive.py     <- Script responsible for generating the interactive dashboards
            └── 3_📊_Static.py          <- Script responsible for generating the static dashboards

# 5. Outro

- To uninstall all dependencies, run the following command:

      pip uninstall -r requirements.txt -y

- To deactivate the virtual environment, run the following command:

      deactivate

- To delete the virtual environment, run the following command:

      rm -rf .venv
