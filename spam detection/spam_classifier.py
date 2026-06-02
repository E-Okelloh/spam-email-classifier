"""
Spam Email Classifier
A machine learning model to classify emails as spam or ham (legitimate).

Author: E-Okelloh
Created: 2024
License: MIT
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


class SpamClassifier:
    """
    A spam email classifier using TF-IDF and Logistic Regression.
    
    Attributes:
        model: Trained Logistic Regression model
        tfidf: Fitted TF-IDF vectorizer
        accuracy: Model accuracy score
    """
    
    def __init__(self, random_state=42, test_size=0.2, max_features=5000):
        """
        Initialize the SpamClassifier.
        
        Args:
            random_state (int): Random seed for reproducibility
            test_size (float): Proportion of data to use for testing
            max_features (int): Maximum number of features for TF-IDF
        """
        self.random_state = random_state
        self.test_size = test_size
        self.max_features = max_features
        self.model = None
        self.tfidf = None
        self.accuracy = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        
    def load_data(self, filepath):
        """
        Load email dataset from CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV structure is invalid
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Data loaded successfully: {len(df)} emails")
            return df
        except FileNotFoundError:
            print(f"✗ Error: File not found at {filepath}")
            raise
        except pd.errors.ParserError:
            print(f"✗ Error: Invalid CSV format at {filepath}")
            raise
    
    def preprocess_data(self, df):
        """
        Clean and preprocess the data.
        
        Args:
            df (pd.DataFrame): Raw dataframe with 'Category' and 'Message' columns
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        print("\n📋 Data Preprocessing:")
        
        # Handle missing values
        data = df.where((pd.notnull(df)), '')
        print(f"  - Handled missing values")
        
        # Display basic statistics
        print(f"  - Dataset shape: {data.shape}")
        print(f"  - Columns: {list(data.columns)}")
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            data = data.drop_duplicates()
            print(f"  - Removed {duplicates} duplicate rows")
        
        # Display class distribution
        print(f"\n  Class Distribution:")
        print(f"    Ham: {(data['Category'] == 'ham').sum()}")
        print(f"    Spam: {(data['Category'] == 'spam').sum()}")
        
        return data
    
    def encode_labels(self, data):
        """
        Convert categorical labels to numeric values.
        
        Args:
            data (pd.DataFrame): Data with 'Category' column
            
        Returns:
            pd.DataFrame: Data with encoded labels (spam=0, ham=1)
        """
        data_encoded = data.copy()
        data_encoded['Category'] = data_encoded['Category'].map({'spam': 0, 'ham': 1})
        print("\n🔢 Label Encoding: spam=0, ham=1")
        return data_encoded
    
    def extract_features(self, X_train, X_test):
        """
        Extract TF-IDF features from text data.
        
        Args:
            X_train: Training text data
            X_test: Testing text data
            
        Returns:
            tuple: (X_train_tfidf, X_test_tfidf)
        """
        print("\n📝 TF-IDF Vectorization:")
        self.tfidf = TfidfVectorizer(max_features=self.max_features)
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_test_tfidf = self.tfidf.transform(X_test)
        
        print(f"  - Vocabulary size: {len(self.tfidf.vocabulary_)}")
        print(f"  - Training features shape: {X_train_tfidf.shape}")
        print(f"  - Testing features shape: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf
    
    def train(self, df):
        """
        Train the spam classifier.
        
        Args:
            df (pd.DataFrame): Dataset with 'Category' and 'Message' columns
            
        Returns:
            float: Model accuracy on test set
        """
        print("\n🚀 Training Spam Classifier")
        print("=" * 50)
        
        # Preprocess data
        data = self.preprocess_data(df)
        
        # Encode labels
        data = self.encode_labels(data)
        
        # Prepare features and target
        X = data['Message']
        y = data['Category']
        
        # Split data
        print("\n🔀 Train-Test Split (80-20):")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        print(f"  - Training set: {len(X_train)} emails")
        print(f"  - Testing set: {len(X_test)} emails")
        
        # Extract features
        X_train_tfidf, X_test_tfidf = self.extract_features(X_train, X_test)
        
        # Train model
        print("\n🤖 Model Training:")
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train_tfidf, y_train)
        print("  - Logistic Regression model trained")
        
        # Evaluate model
        print("\n📊 Model Evaluation:")
        self.predictions = self.model.predict(X_test_tfidf)
        self.accuracy = accuracy_score(y_test, self.predictions)
        print(f"  - Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        
        # Store test data for later evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        return self.accuracy
    
    def evaluate(self):
        """
        Display detailed evaluation metrics.
        """
        if self.predictions is None or self.y_test is None:
            print("✗ Model not trained yet. Please run train() first.")
            return
        
        print("\n📈 Detailed Classification Report:")
        print("=" * 50)
        print(classification_report(
            self.y_test, 
            self.predictions,
            target_names=['Spam', 'Ham']
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.predictions)
        print("\n🔲 Confusion Matrix:")
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")
    
    def predict(self, email_text):
        """
        Classify a single email.
        
        Args:
            email_text (str): Email content to classify
            
        Returns:
            dict: Classification result with prediction and confidence
        """
        if self.model is None or self.tfidf is None:
            print("✗ Model not trained yet. Please run train() first.")
            return None
        
        # Vectorize input
        email_tfidf = self.tfidf.transform([email_text])
        
        # Make prediction
        prediction = self.model.predict(email_tfidf)[0]
        confidence = self.model.predict_proba(email_tfidf)[0]
        
        result = {
            'classification': 'Ham' if prediction == 1 else 'Spam',
            'prediction': prediction,
            'spam_probability': confidence[0],
            'ham_probability': confidence[1]
        }
        
        return result
    
    def predict_batch(self, email_list):
        """
        Classify multiple emails.
        
        Args:
            email_list (list): List of email texts
            
        Returns:
            list: List of classification results
        """
        if self.model is None or self.tfidf is None:
            print("✗ Model not trained yet. Please run train() first.")
            return None
        
        results = []
        for email in email_list:
            result = self.predict(email)
            results.append(result)
        
        return results


def main():
    """Main execution function."""
    print("\n" + "="*50)
    print("SPAM EMAIL CLASSIFIER")
    print("="*50)
    
    # Initialize classifier
    classifier = SpamClassifier()
    
    # Define data path
    data_path = Path(__file__).parent / 'mail_data.csv'
    
    # Load and train
    try:
        df = classifier.load_data(str(data_path))
        accuracy = classifier.train(df)
        classifier.evaluate()
        
        # Example predictions
        print("\n💬 Example Predictions:")
        print("=" * 50)
        
        test_emails = [
            "Click here to win FREE money! Limited time offer!!!",
            "Hi, how are you doing? Let's catch up soon."
        ]
        
        for email in test_emails:
            result = classifier.predict(email)
            print(f"\nEmail: {email[:50]}...")
            print(f"Classification: {result['classification']}")
            print(f"Confidence: {max(result['spam_probability'], result['ham_probability']):.2%}")
        
        print("\n✓ Classifier training completed successfully!")
        
    except FileNotFoundError:
        print(f"\n✗ Error: {data_path} not found")
        print("Please ensure mail_data.csv is in the spam detection directory")
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
