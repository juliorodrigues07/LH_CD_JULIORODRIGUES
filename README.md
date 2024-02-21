[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=ffffff)](https://jupyter.org/)
[![Python 3.10.12](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3106/)
# Rent Pricing
EDA on rent pricing at american districts with interactive dashboards, along with develelopment of a ML regression model.

- **To see the complete project, click [HERE](https://github.com/juliorodrigues07/LH_CD_JULIORODRIGUES/tree/extras)**;
- **To see the deployed application, click on the following button: [![Deploy](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)](https://rent-pricing.streamlit.app/)**
 
# Requirements

- [Python3](https://python.org) and [pip](https://pip.pypa.io/en/stable/installation/) package manager:

      sudo apt install python3 python3-pip build-essential python3-dev
 
- [virtualenv](https://virtualenv.pypa.io/en/latest/) tool:

      pip install virtualenv

# Execution

1. Clone the repository

       git clone git@github.com:juliorodrigues07/LH_CD_JULIORODRIGUES.git

2. Unzip the repository

       unzip LH_CD_JULIORODRIGUES-master.zip && cd LH_CD_JULIORODRIGUES-master

2. Create a virtual environment

       python3 -m venv .venv

3. Activate the virtual environment

       source .venv/bin/activate

4. Install the dependencies

       pip install -r requirements.txt

## 1. Regressor and Dashboards

You first need to be in the _dashboards_ directory to run the commands.

- With Streamlit:

      streamlit run Home.py

- With Dash Plotly (only dashboard):

      python3 dash_test.py

### Interactive Dashboard (Streamlit)
![Interactive](/docs/interactive.png)

### Static Dashboard (Streamlit)
![Static](/docs/static.png)

## 2. Data Mining

- To visualize the notebook, its cells and run them, click [here](https://colab.research.google.com/github/juliorodrigues07/rent_pricing/blob/master/notebook/LH_CD_JULIORODRIGUES.ipynb) (visualization in GitHub is [broken](https://github.com/juliorodrigues07/LH_CD_JULIORODRIGUES/tree/master/notebook));

- To run the notebook locally:
  
     1. Install [VS Code](https://code.visualstudio.com/download);
     2. Open the notebook (_.ipynb_) file in it;
     3. Create the virtual environment and install the dependencies as shown before;
     4. Make sure the the right Python interpreter is configured;
     5. In VS Code press Ctrl + Shift + P and select the recommended or venv option;

- To run python scripts locally, you first need to be in the _src_ directory to run the commands.

      python3 main.py
