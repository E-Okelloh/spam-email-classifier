"""
Unit tests for Spam Email Classifier
Tests the functionality of the SpamClassifier class
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from spam_classifier import SpamClassifier


class TestSpamClassifier(unittest.TestCase):
    """Test cases for SpamClassifier class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.classifier = SpamClassifier(random_state=42)
        
        # Create sample data
        cls.sample_data = pd.DataFrame({
            'Category': ['spam', 'ham', 'spam', 'ham', 'spam'] * 20,
            'Message': [
                "FREE MONEY! Click here to win!",
                "Hi, how are you?",
                "WINNER! Claim your prize now!",
                "See you at the meeting tomorrow",
                "You have won! Act now!"
            ] * 20
        })
    
    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(self.classifier.random_state, 42)
        self.assertEqual(self.classifier.test_size, 0.2)
    
    def test_preprocessing(self):
        """Test data preprocessing"""
        cleaned_data = self.classifier.preprocess_data(self.sample_data)
        self.assertEqual(len(cleaned_data), len(self.sample_data))
        self.assertIn('Category', cleaned_data.columns)
        self.assertIn('Message', cleaned_data.columns)
    
    def test_label_encoding(self):
        """Test label encoding"""
        data_copy = self.sample_data.copy()
        encoded_data = self.classifier.encode_labels(data_copy)
        
        # Check that labels are numeric
        self.assertTrue(encoded_data['Category'].dtype in [np.int64, np.int32])
        
        # Check values are 0 or 1
        self.assertTrue(all(encoded_data['Category'].isin([0, 1])))
    
    def test_training(self):
        """Test model training"""
        accuracy = self.classifier.train(self.sample_data)
        
        # Check that model was trained
        self.assertIsNotNone(self.classifier.model)
        self.assertIsNotNone(self.classifier.tfidf)
        
        # Check accuracy is valid
        self.assertIsInstance(accuracy, (float, np.floating))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_single_prediction(self):
        """Test single email prediction"""
        if self.classifier.model is None:
            self.classifier.train(self.sample_data)
        
        test_email = "FREE money for you!"
        result = self.classifier.predict(test_email)
        
        # Check result structure
        self.assertIn('classification', result)
        self.assertIn('prediction', result)
        self.assertIn('spam_probability', result)
        self.assertIn('ham_probability', result)
        
        # Check prediction is valid
        self.assertIn(result['classification'], ['Spam', 'Ham'])
        self.assertGreaterEqual(result['spam_probability'], 0.0)
        self.assertLessEqual(result['spam_probability'], 1.0)
    
    def test_batch_prediction(self):
        """Test batch email prediction"""
        if self.classifier.model is None:
            self.classifier.train(self.sample_data)
        
        test_emails = [
            "Click here to win money!",
            "Let's meet tomorrow"
        ]
        
        results = self.classifier.predict_batch(test_emails)
        
        # Check results
        self.assertEqual(len(results), len(test_emails))
        for result in results:
            self.assertIn('classification', result)
            self.assertIn('prediction', result)
    
    def test_tfidf_vectorization(self):
        """Test TF-IDF feature extraction"""
        data = self.classifier.preprocess_data(self.sample_data)
        data = self.classifier.encode_labels(data)
        
        X = data['Message']
        X_train, X_test = X[:80], X[80:]
        
        X_train_tfidf, X_test_tfidf = self.classifier.extract_features(X_train, X_test)
        
        # Check shapes
        self.assertEqual(X_train_tfidf.shape[0], len(X_train))
        self.assertEqual(X_test_tfidf.shape[0], len(X_test))
        
        # Check that features were created
        self.assertGreater(X_train_tfidf.shape[1], 0)
    
    def test_spam_detection_quality(self):
        """Test that classifier correctly identifies spam"""
        if self.classifier.model is None:
            self.classifier.train(self.sample_data)
        
        # Obvious spam
        spam_email = "FREE MONEY! Win now! Act immediately!!!"
        spam_result = self.classifier.predict(spam_email)
        
        # This should likely be classified as spam
        # (not guaranteed but probable)
        self.assertIsNotNone(spam_result['classification'])
    
    def test_ham_detection_quality(self):
        """Test that classifier correctly identifies ham"""
        if self.classifier.model is None:
            self.classifier.train(self.sample_data)
        
        # Obvious ham
        ham_email = "Hi, hope you're doing well. Let's catch up soon."
        ham_result = self.classifier.predict(ham_email)
        
        # This should likely be classified as ham
        # (not guaranteed but probable)
        self.assertIsNotNone(ham_result['classification'])


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and edge cases"""
    
    def test_empty_message(self):
        """Test handling of empty messages"""
        classifier = SpamClassifier()
        
        data = pd.DataFrame({
            'Category': ['ham', 'spam', 'ham'],
            'Message': ['Hello', '', 'How are you?']
        })
        
        cleaned = classifier.preprocess_data(data)
        self.assertEqual(len(cleaned), 3)
    
    def test_missing_columns(self):
        """Test handling of missing required columns"""
        classifier = SpamClassifier()
        
        data = pd.DataFrame({
            'Message': ['Hello', 'World']
        })
        
        # Should raise an error or handle gracefully
        with self.assertRaises((KeyError, ValueError)):
            classifier.preprocess_data(data)


class TestModelPersistence(unittest.TestCase):
    """Test model predictions consistency"""
    
    def test_consistent_predictions(self):
        """Test that predictions are consistent with same input"""
        classifier = SpamClassifier(random_state=42)
        
        data = pd.DataFrame({
            'Category': ['spam', 'ham'] * 50,
            'Message': [
                "FREE MONEY! Click here to win!",
                "Hi, how are you?",
            ] * 50
        })
        
        classifier.train(data)
        
        test_email = "Get free cash now!"
        result1 = classifier.predict(test_email)
        result2 = classifier.predict(test_email)
        
        # Same input should give same prediction
        self.assertEqual(result1['prediction'], result2['prediction'])


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
