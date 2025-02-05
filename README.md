# Neural-Network-Model-for-Wine-Type-Prediction
 Project Overview

This project builds a Neural Network Model using TensorFlow and Keras to classify different types of wines based on their chemical properties. The dataset is sourced from the UCI Machine Learning Repository.

📂 Dataset
The dataset consists of 13 numerical features describing the chemical composition of wines.
The target variable represents three classes (wine types).
The dataset is loaded directly from UCI Wine Dataset.

🛠 Tech Stack
Python
TensorFlow/Keras
Pandas
Matplotlib
Scikit-learn

🚀 Project Workflow

Data Preprocessing:
Load the dataset.
Encode the target variable using one-hot encoding.
Split the dataset into training and testing sets.

Model Building:
Create a Sequential Neural Network with fully connected layers.
Use ReLU activation for hidden layers and Softmax for the output layer.

Compilation & Training:
Compile the model using the Adam optimizer and categorical cross-entropy loss.
Train the model for 30 epochs with batch size 16.

Evaluation & Visualization:
Plot the accuracy over epochs.
Generate a classification report to analyze model performance.

How to Run the Project

Clone the repository:
git clone https://github.com/yourusername/wine-classification.git
cd wine-classification

Install dependencies:
pip install -r requirements.txt

Run the script:
python wine_classification.py

📌 Results
The model achieves a high accuracy in classifying different types of wines.
The training process is visualized using a plot of accuracy over epochs.
A classification report is generated to analyze precision, recall, and F1-score
