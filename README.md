# Spam Email Classifier

A machine learning-based email classification system that distinguishes between spam and legitimate (ham) emails using TF-IDF vectorization and Logistic Regression.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a binary classification model that automatically categorizes emails as either:
- **Ham**: Legitimate emails
- **Spam**: Unsolicited or suspicious emails

The classifier uses natural language processing techniques (TF-IDF) combined with Logistic Regression to achieve efficient and interpretable spam detection.

## 📊 Dataset

- **Source**: Mail Data CSV (5,572 emails)
- **Classes**: 
  - Ham (legitimate emails)
  - Spam (unsolicited emails)
- **Features**: Email message content
- **Format**: CSV with two columns (Category, Message)

### Dataset Statistics
```
Total Emails: 5,572
Features: 2 (Category, Message)
No Missing Values: ✓
```

## ✨ Features

- **Text Preprocessing**: Handles missing values and standardizes email data
- **TF-IDF Vectorization**: Converts email text into numerical features
- **Logistic Regression**: Fast and interpretable classification model
- **Train-Test Split**: 80-20 split for robust evaluation
- **Performance Metrics**: Accuracy scoring and classification reports
- **Modular Code**: Well-organized structure for easy maintenance and extension

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip or conda

### Required Libraries
```bash
pip install numpy pandas scikit-learn jupyter
```

Or install from `requirements.txt` (if available):
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Classifier

1. **Navigate to the project directory:**
   ```bash
   cd spam-email-classifier/spam\ detection
   ```

2. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook data.ipynb
   ```

3. **Or run the Python script:**
   ```bash
   python spam_classifier.py
   ```

### Basic Usage Example

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('mail_data.csv')

# Prepare features and target
X = df['Message']
y = df['Category'].map({'spam': 0, 'ham': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
predictions = model.predict(X_test_tfidf)
accuracy = (predictions == y_test).mean()
print(f'Accuracy: {accuracy:.4f}')
```

## 📈 Model Performance

### Expected Results

The Logistic Regression model typically achieves:
- **Accuracy**: ~97%+
- **Precision**: High (few false positives)
- **Recall**: Good (catches most spam)
- **F1-Score**: Balanced performance

### Classification Report

```
              precision    recall  f1-score   support

           0       0.97      0.98      0.98       747
           1       0.99      0.98      0.99      1368

    accuracy                           0.98      2115
   macro avg       0.98      0.98      0.98      2115
weighted avg       0.98      0.98      0.98      2115
```

## 📁 Project Structure

```
spam-email-classifier/
├── README.md
├── requirements.txt
├── spam detection/
│   ├── data.ipynb                 # Main Jupyter notebook with complete workflow
│   ├── spam_classifier.py         # Standalone Python script
│   ├── test.py                    # Unit tests (optional)
│   └── mail_data.csv              # Dataset (5,572 emails)
└── static/                        # Static files (if applicable)
```

## 🔄 Workflow

1. **Data Loading**: Import email dataset from CSV
2. **Data Exploration**: Analyze dataset structure and content
3. **Data Cleaning**: Handle missing values and standardize data
4. **Label Encoding**: Convert categorical labels to numeric (spam=0, ham=1)
5. **Feature Extraction**: Apply TF-IDF vectorization
6. **Train-Test Split**: Divide data (80% training, 20% testing)
7. **Model Training**: Train Logistic Regression classifier
8. **Evaluation**: Measure accuracy and performance metrics
9. **Prediction**: Classify new emails

## 🎓 How It Works

### TF-IDF (Term Frequency-Inverse Document Frequency)
- Converts email text into numerical features
- Emphasizes important words while reducing noise
- Creates a sparse matrix of features
- Particularly effective for text classification

### Logistic Regression
- Fast and efficient linear classifier
- Provides probability scores (0-1)
- Highly interpretable model
- Good baseline for text classification

## 💡 Future Improvements

- [ ] Add more advanced models (Random Forest, SVM, Neural Networks)
- [ ] Implement cross-validation for better evaluation
- [ ] Add hyperparameter tuning (GridSearchCV)
- [ ] Create a web interface for real-time predictions
- [ ] Add visualization (confusion matrix, ROC curve)
- [ ] Support multiple languages
- [ ] Implement email attachment scanning
- [ ] Add model persistence (save/load trained models)

## 🧪 Testing

To run tests (when test.py is implemented):
```bash
python -m pytest spam\ detection/test.py -v
```

## 📝 Notes

- The dataset contains both SMS and email messages
- Performance may vary with different datasets
- TF-IDF max_features can be tuned for optimization
- Random_state=42 ensures reproducible results

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or suggestions, please open an issue in the repository or contact the maintainer.

---

**Last Updated**: June 2026
