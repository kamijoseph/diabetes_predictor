# 🧠 Streamlit Diabetes Prediction App

🔗 **Live Demo**: [Click here to try the deployed app](https://diabetes-predictor.streamlit.app/)

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-SVM%20Classifier-brightgreen.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

This project is an interactive **diabetes risk classification tool** built with **Streamlit** and powered by a trained **Support Vector Machine (SVM)** model. It enables users to simulate clinical scenarios by entering key medical indicators and obtain real-time predictions on diabetes presence.

The classifier is based on the **Pima Indians Diabetes Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes) and [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which includes diagnostic data such as glucose level, insulin, BMI, and more.

> ⚠️ **Disclaimer**: This application is for **educational and research purposes only**. It is **not intended as a diagnostic tool**. Always consult a licensed healthcare professional for medical diagnosis or treatment.

---

## 📁 Project Structure
```bash
diabetes_predictor/
│
├── app/
│   └── main.py
│
├── data/
│   └── diabetes.csv
│
├── notebook/
│   ├── diabetics_prediction.ipynb
│   ├── model.sav
│   └── scaler.sav
│
├── requirements.txt
└── README.md
````

## 🚀 Features

- 🎚️ **Interactive Sliders and Manual Inputs** for eight diabetes related medical features

- 🔍 **Hybrid Input Mode:** Use either sidebar sliders or text inputs

- 📈 **Real-Time** Prediction using SVM model

- 📊 **Probability Scores** per class (Diabetic / Not Diabetic)

- ⚠️ **Medical Disclaimer Message** for responsible and proper application

## ⚙️ Installation
### 🔐 Prerequisites
- Python ≥ 3.10

- Conda (recommended)

- Git

### 📦 Setup Guide
1. Clone this repository
```bash
git clone https://github.com/kamijoseph/diabetes_predictor.git
cd diabetes_predictor
```
2. Create a new Conda environment
```bash
conda create -n diabetes-predictor python=3.12
```
3. Activate the environment
```bash
conda activate diabetes-predictor
```
4. Install dependencies
```bash
conda install --file requirements.txt
```
5. Run streamlit application
```bash
cd app
streamlit run main.py
```

---

## 🙋‍♂️ Questions or Feedback?

Feel free to open an issue or reach out if you have suggestions, questions, or ideas to improve this project.
Built by @kamijoseph using Streamlit

---

### Built by [@kamijoseph](https://github.com/kamijoseph) using [Streamlit](https://streamlit.io/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://breast-cancer-prediction-f3wtpgbjzvpvohqysgbefx.streamlit.app//)
