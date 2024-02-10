[![Python 3.10.12](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3106/)
# Rent Pricing
EDA on rent pricing at american districts with interactive dashboards.
 
# Requirements

- [Python3](https://python.org) and [pip](https://pip.pypa.io/en/stable/installation/) package manager:

      sudo apt install python3 python3-pip build-essential python3-dev
 
- [virtualenv](https://virtualenv.pypa.io/en/latest/) tool:

      pip install virtualenv

# Execution

1. Clone the repository

       git clone https://github.com/juliorodrigues07/rent_pricing.git

2. Unzip the repository

       unzip rent_pricing-master.zip && cd rent_pricing-master/src

2. Create a virtual environment

       python3 -m venv .venv

3. Activate the virtual environment

       source .venv/bin/activate

4. Install the dependencies

       pip install -r requirements.txt

5. Run the application (Inside _src_ directory)

- With Streamlit:

      streamlit run Home.py

- With Dash Plotly:

      python3 dash_test.py

5. Deactivate the virtual environment

       deactivate
