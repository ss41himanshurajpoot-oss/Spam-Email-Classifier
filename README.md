# 📧 Spam Email Classifier using Naive Bayes

## 📌 Overview
This project is a machine learning-based spam email classifier that uses the Naive Bayes algorithm to classify messages as **Spam** or **Ham (Not Spam)**.

The system analyzes the content of text messages and predicts whether they are unwanted spam or legitimate communication.

---

## 🎯 Objectives
- To build a spam detection system using machine learning
- To understand and apply the Naive Bayes theorem
- To perform text preprocessing and feature extraction
- To evaluate model performance using accuracy metrics

---

## 🧠 Key Concepts Used
- Machine Learning
- Supervised Learning
- Naive Bayes Classification
- Natural Language Processing (NLP)
- Probability Theory
- TF-IDF (Term Frequency - Inverse Document Frequency)

---

## ⚙️ Technologies & Tools
- Python 3.x
- Pandas (data handling)
- NumPy (numerical computation)
- Scikit-learn (ML model)
- Pickle (model saving)

---

## 📂 Project Structure
spam_classifier/
│── spam_classifier.py # Main code
│── spam.csv # Dataset
│── model.pkl # Saved trained model
│── README.md # Documentation


---

## 📊 Dataset Information
- Dataset: SMS Spam Collection Dataset
- Format: CSV file
- Columns:
  - `v1`: Label (spam or ham)
  - `v2`: Message text

Example:
ham, Hey how are you?
spam, Congratulations! You won a prize!


---

## 🔄 Workflow

1. Data Loading  
2. Data Cleaning (removing punctuation, lowercasing)  
3. Feature Extraction (TF-IDF)  
4. Model Training (Naive Bayes)  
5. Prediction  
6. Evaluation  

---

## 🧪 Model Details

The classifier uses:

- **CountVectorizer** → Converts text to numeric vectors  
- **TF-IDF Transformer** → Weighs important words  
- **Multinomial Naive Bayes** → Performs classification  

---

## ▶️ How to Run the Project

### Step 1: Install Dependencies

### Step 2: Add Dataset
Place `spam.csv` in the same folder.

### Step 3: Run the Program
---

## 📈 Evaluation Metrics

- Accuracy Score  
- Precision  
- Recall  
- F1 Score  

These metrics are printed after training.

---

## 🚀 Features
- Interactive user input
- Real-time prediction
- Probability/confidence output
- Model saving and reuse

---

## ⚠️ Limitations
- Depends on dataset quality
- May misclassify rare messages
- No GUI interface

---

## 🔮 Future Improvements
- Add graphical interface (Streamlit)
- Use deep learning (LSTM/Transformers)
- Integrate with email services
- Improve preprocessing (stopwords removal, stemming)

---

## 👨‍💻 Author
**Himanshu Rajpoot**

---

## 📜 License
This project is for educational purposes only.
