# Telco Customer Churn Prediction

> A simple, easy-to-understand machine learning project that predicts if a customer will leave their telecom provider. Built with scikit-learn, Pydantic, and Loguru.

---

## What is this project?

Customer churn means a customer cancels their service. It is very expensive for a company to lose customers. If we can predict who is going to leave, the company can try to keep them!

This project builds a **Machine Learning Model** to guess if a customer will churn (`Yes`) or stay (`No`). It uses a famous dataset from Kaggle called the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

I built this project to be clean, organized, and easy to read. It uses good coding practices like configuration files, logging, and unit tests.

---

## Why is this project cool?

- **Easy to change settings:** You can change the machine learning model or the testing settings just by editing `config.yaml`. You don't need to touch the Python code!
- **Catches errors early:** It uses Pydantic to check the `config.yaml` file before the program starts running.
- **Custom Data Prep:** I wrote a custom step (`FeatureEngineering`) to create new, helpful data columns before training the model.
- **Nice colorful logs:** It uses Loguru to print clear, colored messages so you always know what the program is doing.
- **Unit Tested:** I wrote tests using `pytest` to make sure all parts of the code work correctly.

---

## How the folders are organized

```text
telco-churn-prediction/
│
├── config/
│   └── config.yaml          # All the settings for the project are here
│
├── data/
│   ├── raw/                 # Put the downloaded Kaggle dataset here
│   └── processed/           # The cleaned data is saved here automatically
│
├── logs/                    # Text files showing what the program did
│
├── models/                  # The trained model is saved here
│
├── notebooks/
│   ├── eda.ipynb            # Data exploration and charts
│   └── feature_engineering.ipynb
│
├── reports/
│   └── metrics.json         # How well the model scored (F1, Accuracy, etc.)
│
├── src/
│   ├── __init__.py
│   ├── logger.py            # Sets up the colorful print messages
│   ├── schemas.py           # Checks if config.yaml is written correctly
│   ├── utils.py             # Small helper functions
│   ├── data_loader.py       # Reads and cleans the raw data
│   ├── preprocess.py        # Prepares the data for the model
│   ├── train.py             # Trains the machine learning model
│   ├── evaluation.py        # Calculates the scores
│   └── visualizer.py        # Draws charts showing model performance
│
├── tests/                   # Folder containing all the unit tests
├── main.py                  # The main file you run to start the program
└── environment.yml          # The list of packages you need to install
```

---

## How it works (Step-by-Step)

1. **Data Loading:** The program reads the CSV file and removes bad data (like missing values and duplicates).
2. **Feature Engineering:** It creates new helpful columns:
   - `HasFamily`: 1 if they have a partner/dependents, 0 if not.
   - `TotalService`: How many extra services they bought.
   - `TotalCharges_log`: Fixes the math for the total charges column.
3. **Preprocessing:** It scales numbers (so they aren't too big) and changes text into numbers so the model can read it.
4. **Training:** It trains the model using the settings from `config.yaml`.
5. **Evaluation:** It tests the model and saves the scores (like F1 score) to `reports/metrics.json`.

---

## How to use it

### Step 1: Install the requirements

You can use Conda to install everything you need:

```bash
git clone https://github.com/<selim-sunman>/telco-churn-prediction.git
cd telco-churn-prediction

conda env create -f environment.yml
conda activate telco-churn-prediction
```

*(If you prefer pip, you can also use a virtual environment and a requirements file).*

### Step 2: Download the data

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.
Put the `dataset.csv` file into the `data/raw/` folder.

### Step 3: Run the project

Just run the main file:

```bash
python main.py
```

The program will run and tell you what it is doing in the console. When it's done, you can check `reports/metrics.json` to see how well the model did!

---

## Changing the Model

Want to try a different model? You don't need to write Python code. Just open `config/config.yaml` and change the `model` section. 

For example, to use a Random Forest:

```yaml
model:
  module:     "sklearn.ensemble"
  model_name: "RandomForestClassifier"
  params:
    n_estimators: 200
    max_depth:    10
    random_state: 42
```

---

## Running the Tests

To make sure everything is working, you can run the test suite:

```bash
pytest tests/ -v
```

---

## Tech Stack

Here are the main tools I used for this project:

- **Python 3.10**
- **scikit-learn** (Machine Learning)
- **pandas & NumPy** (Data manipulation)
- **Pydantic** (Config checking)
- **Loguru** (Logging)
- **pytest** (Testing)
