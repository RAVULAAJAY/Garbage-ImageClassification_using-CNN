# 🗑️ Garbage Image Classification (CNN Model)

This project focuses on classifying different types of garbage images
using a **Convolutional Neural Network (CNN)** built with
**TensorFlow/Keras**.\
The model is trained on a labeled dataset containing multiple waste
categories such as plastic, paper, metal, cardboard, organic, and more.\
It helps in developing smart waste-management automation systems.

## 📁 Project Overview

-   Preprocessed dataset (resizing, normalization)\
-   Applied **image augmentation** for improved generalization\
-   Designed and trained a custom **CNN architecture**\
-   Evaluated performance using accuracy, loss, and confusion matrix\
-   Visualized results using plots\
-   Ready to extend using **Transfer Learning** or deploy using
    **Streamlit/Gradio**

## 📂 Project Structure

    📦 Garbage-Classification/
    │── garbage-image-classification.ipynb
    │── dataset/
    │   ├── train/
    │   ├── test/
    │── README.md

## 🧠 Model Summary

The CNN model consists of:

-   Multiple Conv2D + MaxPooling layers\
-   Dense layers for classification\
-   Dropout for regularization\
-   Softmax output layer

It extracts image features and predicts the correct garbage category.

## 🖼️ Dataset Sample Images

👉 *(Add your dataset sample images below)*

**\[Add Screenshot Here\]**

## 📊 Training & Validation Graphs

👉 *(Add accuracy and loss graphs here)*

**\[https://github.com/RAVULAAJAY/Garbage-ImageClassification_using-CNN/blob/aa1b18d1537cfd695308b83417dfc68cc5cdf996/Images/MODEL%20%20Acuracy%20Loss.png\]**

## 📉 Confusion Matrix

👉 *(Add confusion matrix screenshot here)*

**\[Add Screenshot Here\]**

## ✔️ Final Results

You can view the final accuracy, loss, and evaluation metrics in the
notebook.\
(Add your exact accuracy manually if needed.)

## ▶️ How to Run

Install dependencies:

    pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

Run the notebook:

    jupyter notebook garbage-image-classification.ipynb

## 🚀 Future Enhancements

-   Apply **Transfer Learning** (MobileNet, EfficientNet, ResNet)\
-   Deploy model using **Streamlit/Gradio web app**\
-   Increase dataset size\
-   Add real-time garbage detection API

## 👤 Author

**Ravula Ajay**\
Garbage Image Classification using Deep Learning (CNN)
