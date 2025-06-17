# 📊 Machine Learning Tabular Model (Keras + TensorFlow)

A simple but effective machine learning pipeline built with **TensorFlow/Keras** for training on structured/tabular data using a CSV file as input.

---

### 🚀 Features

- ✅ Automatic encoding and normalization for numeric and categorical features  
- 🧠 Dense Neural Network (DNN) architecture  
- 📦 TensorFlow `tf.data.Dataset` pipeline  
- 🔎 Model visualization with `keras.utils.plot_model`  
- 📑 CSV-based training data  
- 🧪 Separate training (`main.py`) and testing (`test.py`) scripts  

---

### 📁 Project Structure

| File/Folder     | Description                                      |
|------------------|--------------------------------------------------|
| `main.py`        | Trains the model and processes the dataset       |
| `test.py`        | Loads the model and makes predictions            |
| `data.csv`       | Tabular dataset including input features + target |
| `Data/`          | (Optional) Additional files or output directory  |

---

🔍 Technical Notes
Numeric features are normalized using Normalization layers.

Categorical features are encoded using StringLookup.

The target column must be named Target in the CSV.

Dataset is processed using the efficient tf.data.Dataset pipeline.

The model structure can be visualized using plot_model.


### 🔧 How to Run

#### 1. Install dependencies:
```bash
pip install tensorflow pandas numpy
python main.py
python test.py



