# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud

import warnings
warnings.filterwarnings("ignore")

# Loading the data
try:
    df = pd.read_csv("YoutubeSpamDataset.csv")
    print("CSV loaded successfully!")
    print(df.head())  # Show the first few rows of the dataset
    print("Columns in the DataFrame:", df.columns.tolist())

except FileNotFoundError:
    print("CSV file not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

# Remove leading and trailing spaces
df.columns = df.columns.str.strip()

# Map the class values to 'spam' and 'not spam'
df['CLASS'] = df['CLASS'].map({1: 'spam', 0: 'not spam'})  

# Create a pie chart
class_counts = df['CLASS'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=['blue', 'red'], startangle=90, explode=(0.05, 0))
plt.title('Spam vs Non-Spam Comments')
plt.show()

# Create distribution of spam and not spam comments
sns.countplot(x='CLASS', data=df)
plt.title('Distribution of Spam and Not Spam Comments')
plt.xlabel('Comment Type')
plt.ylabel('Count')
plt.show()

# Create a barchart to find ratio of spam and nonspam comments in each video
video_spam_counts = df.groupby('VIDEO_NAME')['CLASS'].value_counts().unstack(fill_value=0)
video_spam_counts['Spam Ratio'] = video_spam_counts['spam'] / (video_spam_counts['not spam'] + video_spam_counts['spam'])
plt.figure(figsize=(12, 6))
video_spam_counts['Spam Ratio'].plot(kind='bar', color='blue')
plt.title('Spam to Non-Spam Ratio for Each Video')
plt.xlabel('Video Name')
plt.ylabel('Spam Ratio')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for ratio
plt.grid(axis='y')
plt.tight_layout() 
plt.show()

# Split into features (X) and target (y)
X = df['CONTENT']
y = df['CLASS']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Fit and transform the training data, transform the testing data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Apply SMOTE to balance the classes
smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

# Initialize models
models = {
    "Linear SVC": LinearSVC(),
    "SVM": SVC(kernel='rbf', random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Train models, make predictions, and evaluate performance
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_res, y_train_res)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate the model
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
    print(f"{model_name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"{model_name} Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\n" + "="*50 + "\n")  # Separator between model outputs

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Spam', 'Spam'], 
                yticklabels=['Not Spam', 'Spam'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Create word cloud for spam comments
spam_comments = df[df['CLASS'] == 'spam']['CONTENT']
spam_wordcloud = WordCloud(width=800, height=400, max_words=100).generate(' '.join(spam_comments))
plt.figure(figsize=(10, 5))
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Spam Comments')
plt.show()

# Create word cloud for non-spam comments
nonspam_comments = df[df['CLASS'] == 'not spam']['CONTENT']
nonspam_wordcloud = WordCloud(width=800, height=400, max_words=100).generate(' '.join(nonspam_comments))
plt.figure(figsize=(10, 5))
plt.imshow(nonspam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Not Spam Comments')
plt.show()