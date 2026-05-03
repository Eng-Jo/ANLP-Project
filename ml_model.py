"""
Machine Learning Model for CV Scoring
Uses Random Forest Classifier and Feature Engineering
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any

class CVScorerModel:
    """Machine Learning model for CV scoring"""
    
    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = [
            'has_email',
            'has_phone', 
            'has_linkedin',
            'has_github',
            'num_sections',
            'num_technical_skills',
            'num_action_verbs',
            'word_count',
            'has_bullets',
            'education_quality',
            'experience_quality',
            'project_count',
            'quantifiable_achievements',
            'keyword_density',
            'formatting_score'
        ]
    
    def extract_features(self, cv_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from CV analysis data"""
        features = []
        
        # Contact features (4 features)
        features.append(1 if cv_data.get('email') else 0)
        features.append(1 if cv_data.get('phone') else 0)
        features.append(1 if cv_data.get('linkedin') else 0)
        features.append(1 if cv_data.get('github') else 0)
        
        # Section features (1 feature)
        features.append(len(cv_data.get('found_sections', [])))
        
        # Skills and verbs (2 features)
        features.append(len(cv_data.get('technical_skills', [])))
        features.append(len(cv_data.get('action_verbs', [])))
        
        # Length feature (1 feature)
        word_count = cv_data.get('word_count', 0)
        # Normalize word count (optimal range: 300-800)
        normalized_word_count = min(word_count / 800, 1.5) if word_count > 0 else 0
        features.append(normalized_word_count)
        
        # Formatting (1 feature)
        formatting_issues = cv_data.get('formatting_issues', [])
        has_bullets = 1 if len(formatting_issues) == 0 else 0
        features.append(has_bullets)
        
        # Education quality (1 feature)
        education_score = self._calculate_education_quality(cv_data)
        features.append(education_score)
        
        # Experience quality (1 feature)
        experience_score = self._calculate_experience_quality(cv_data)
        features.append(experience_score)
        
        # Project count (1 feature)
        project_count = 1 if 'projects' in cv_data.get('found_sections', []) else 0
        features.append(project_count)
        
        # Quantifiable achievements (1 feature)
        achievements = self._count_achievements(cv_data)
        features.append(min(achievements / 5, 1))  # Normalize to 0-1
        
        # Keyword density (1 feature)
        keyword_density = self._calculate_keyword_density(cv_data)
        features.append(keyword_density)
        
        # Overall formatting score (1 feature)
        formatting_score = self._calculate_formatting_score(cv_data)
        features.append(formatting_score)
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_education_quality(self, cv_data: Dict[str, Any]) -> float:
        """Calculate education section quality (0-1)"""
        if 'education' not in cv_data.get('found_sections', []):
            return 0.0
        return 1.0
    
    def _calculate_experience_quality(self, cv_data: Dict[str, Any]) -> float:
        """Calculate experience section quality (0-1)"""
        if 'experience' not in cv_data.get('found_sections', []):
            return 0.0
        
        # Consider action verbs as proxy for experience quality
        num_verbs = len(cv_data.get('action_verbs', []))
        return min(num_verbs / 10, 1.0)
    
    def _count_achievements(self, cv_data: Dict[str, Any]) -> int:
        """Count quantifiable achievements"""
        action_verbs = cv_data.get('action_verbs', [])
        return len(action_verbs)
    
    def _calculate_keyword_density(self, cv_data: Dict[str, Any]) -> float:
        """Calculate relevant keyword density"""
        total_keywords = (
            len(cv_data.get('technical_skills', [])) + 
            len(cv_data.get('action_verbs', []))
        )
        word_count = cv_data.get('word_count', 1)
        
        # Keywords per 100 words
        density = (total_keywords / word_count) * 100 if word_count > 0 else 0
        
        # Normalize to 0-1 (optimal: 3-8 keywords per 100 words)
        return min(density / 8, 1.0)
    
    def _calculate_formatting_score(self, cv_data: Dict[str, Any]) -> float:
        """Calculate formatting quality score"""
        issues = len(cv_data.get('formatting_issues', []))
        # Fewer issues = better score
        return max(1 - (issues * 0.2), 0)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Train the model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training R² Score: {train_score:.4f}")
        print(f"Testing R² Score: {test_score:.4f}")
        
        return float(train_score), float(test_score)
    
    def predict(self, cv_data: Dict[str, Any]) -> float:
        """Predict CV score using the trained model"""
        if self.model is None or self.scaler is None:
            raise Exception("Model not trained or loaded! Run train_model.py first.")
        
        # Extract features
        features = self.extract_features(cv_data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        score = self.model.predict(features_scaled)[0]
        
        # Ensure score is between 0 and 100
        score = np.clip(score, 0, 100)
        
        return float(score)
    
    def get_feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        """Get feature importance from the model"""
        if self.model is None:
            return None
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_features
    
    def save_model(self, filepath: str = 'models/cv_scorer_model.pkl') -> None:
        """Save the trained model"""
        if self.model is None or self.scaler is None:
            raise Exception("Cannot save: Model not trained!")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/cv_scorer_model.pkl') -> None:
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")