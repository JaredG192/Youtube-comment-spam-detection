# 💬 YouTube Comment Spam Classification

This project applies machine learning techniques to classify YouTube comments as either **spam** or **not spam**. Using a labeled dataset of user comments, the project explores patterns in text data and evaluates different classification models to detect spam efficiently.

---

## 📁 Project Structure

- `YoutubeSpamDataset.csv` – Cleaned dataset containing comment text, video titles, and class labels (spam or not spam)  
- `YouTube_Spam_Classifier.py` – Python script for data analysis, preprocessing, model training, and visualization  
- `YouTube_Spam_Classifier.ipynb` – Jupyter Notebook version for interactive exploration and evaluation  

---

## 🧪 Objectives

- Explore and visualize spam vs non-spam comment distribution  
- Analyze spam ratios across different videos  
- Engineer text features using TF-IDF  
- Build and evaluate multiple machine learning models for text classification  
- Visualize results using confusion matrices and word clouds  

---

## 🛠️ Methods and Tools

**Languages & Libraries:**
- Python  
- Pandas, NumPy  
- Scikit-learn  
- imbalanced-learn (SMOTE)  
- Matplotlib, Seaborn  
- WordCloud  

**Techniques:**
- Data Cleaning & Feature Mapping  
- TF-IDF Vectorization  
- Class Balancing using SMOTE  
- Classification Algorithms:
  - Linear Support Vector Classifier (Linear SVC)  
  - Support Vector Machine (SVM with RBF kernel)  
  - Decision Tree  
- Evaluation Metrics:
  - Accuracy Score  
  - Confusion Matrix  
  - Classification Report  

---

## 📊 Key Features and Visualizations

- 📈 **Class Distribution** – Pie chart and bar chart showing ratio of spam vs non-spam comments  
- 🎞️ **Spam Ratio per Video** – Bar chart showing proportion of spam comments per video  
- 🧠 **TF-IDF Features** – 5000 most frequent tokens used for modeling  
- 📊 **Confusion Matrices** – Heatmaps comparing predicted vs true labels  
- ☁️ **Word Clouds** – Separate word clouds for spam and non-spam comment content  

---

## 📈 Results Summary

All three models performed well, but the **Linear SVC** and **SVM** models achieved the highest F1-scores. SMOTE was used to balance the dataset and improve classification accuracy.

---

## 📂 Dataset

The dataset includes:
- `COMMENT_ID`, `AUTHOR`, `CONTENT`, `VIDEO_NAME`, and `CLASS` columns  
- 1 = spam, 0 = not spam (mapped to labels during processing)

📥Full dataset is on Kaggle using this link: [Youtube Spam Comments Dataset](https://www.kaggle.com/code/ahmedhassansaqr/youtube-comments-spam-detection-f1-score-96)

---

## 👨‍💻 Authors

**Jared Gonzalez**, **Tejaswi Chigurupati**, **A Sai Prasanth Reddy**   


California State University, San Bernardino 
Bachelor in Science, Computer Science,
Minor in Data Science  

---

## 📃 License

This project is for academic purposes and is shared under the MIT License.
