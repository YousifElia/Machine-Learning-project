# Big Data Spam Detector (PySpark + PyTorch)

A scalable machine learning pipeline that uses **Apache Spark** for distributed data processing and **PyTorch** for deep learning-based classification.

## 📌 Project Overview
This project demonstrates how to integrate Big Data technologies with modern Deep Learning frameworks. It simulates a high-volume email filtering system where:
1.  **PySpark** handles the ETL (Extract, Transform, Load) and feature engineering.
2.  **PyTorch** performs the binary classification (Spam vs. Ham) using a Neural Network.

## 🛠️ Technologies Used
* **Apache Spark (PySpark):** Used for scalable feature extraction (Tokenization, Feature Engineering).
* **PyTorch:** Used to build and train the Neural Network (Binary Cross Entropy Loss, Adam Optimizer).
* **Python:** Core programming language.

## ⚙️ How it Works
### 1. Data Processing (Spark)
Raw text data is processed into numerical features using Spark's distributed DataFrame API.
* **Input:** Raw strings (e.g., "WIN FREE MONEY")
* **Transformation:** * `char_count`: Length of message
    * `word_count`: Number of tokens
    * `keyword_flag`: Boolean check for spam trigger words
* **Output:** Vectorized features ready for training.

### 2. Model Architecture (PyTorch)
A Feed-Forward Neural Network takes the processed Spark vectors as input.
* **Input Layer:** 3 Nodes (corresponding to the Spark features)
* **Hidden Layer:** 5 Nodes (ReLU activation)
* **Output Layer:** 1 Node (Sigmoid activation for probability)

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install pyspark torch numpy
