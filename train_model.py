"""
Generate synthetic training data and train the CV scoring model
"""

import pandas as pd
import numpy as np
from ml_model import CVScorerModel
import os
from typing import List

def generate_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic training data for CV scoring"""
    
    np.random.seed(42)
    data: List[List[float]] = []
    
    for i in range(n_samples):
        # Generate random CV features
        has_email = float(np.random.choice([0, 1], p=[0.1, 0.9]))
        has_phone = float(np.random.choice([0, 1], p=[0.2, 0.8]))
        has_linkedin = float(np.random.choice([0, 1], p=[0.3, 0.7]))
        has_github = float(np.random.choice([0, 1], p=[0.4, 0.6]))
        
        num_sections = float(np.random.randint(2, 6))
        num_technical_skills = float(np.random.randint(0, 20))
        num_action_verbs = float(np.random.randint(0, 15))
        
        word_count = float(np.random.randint(200, 1500) / 800)  # Normalized
        has_bullets = float(np.random.choice([0, 1], p=[0.2, 0.8]))
        
        education_quality = float(np.random.choice([0, 0.5, 1], p=[0.1, 0.3, 0.6]))
        experience_quality = float(min(num_action_verbs / 10, 1.0))
        project_count = float(np.random.choice([0, 1], p=[0.3, 0.7]))
        
        quantifiable_achievements = float(min(num_action_verbs / 5, 1.0))
        keyword_density = float(min((num_technical_skills + num_action_verbs) / 8, 1.0))
        formatting_score = float(np.random.uniform(0.5, 1.0))
        
        # Calculate target score based on features (rule-based for training)
        score = 0.0
        
        # Contact info (15 points)
        score += has_email * 7.5
        score += (1.0 if (has_phone or has_linkedin or has_github) else 0.0) * 7.5
        
        # Sections (30 points)
        score += (num_sections / 5) * 30
        
        # Skills (20 points)
        score += min(num_technical_skills / 10, 1) * 20
        
        # Action verbs (15 points)
        score += min(num_action_verbs / 10, 1) * 15
        
        # Length (10 points)
        if 0.375 <= word_count <= 1:  # 300-800 words
            score += 10
        elif word_count < 0.375:
            score += word_count * 13.33
        else:
            score += max(10 - (word_count - 1) * 5, 0)
        
        # Formatting (10 points)
        score += formatting_score * 10
        
        # Add some noise
        score += float(np.random.normal(0, 3))
        score = float(np.clip(score, 0, 100))
        
        # Create feature vector
        features: List[float] = [
            has_email,
            has_phone,
            has_linkedin,
            has_github,
            num_sections,
            num_technical_skills,
            num_action_verbs,
            word_count,
            has_bullets,
            education_quality,
            experience_quality,
            project_count,
            quantifiable_achievements,
            keyword_density,
            formatting_score
        ]
        
        data.append(features + [score])
    
    # Create DataFrame
    columns = [
        'has_email', 'has_phone', 'has_linkedin', 'has_github',
        'num_sections', 'num_technical_skills', 'num_action_verbs',
        'word_count', 'has_bullets', 'education_quality',
        'experience_quality', 'project_count', 'quantifiable_achievements',
        'keyword_density', 'formatting_score', 'score'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    return df

def main() -> None:
    """Main training function"""
    print("=" * 60)
    print("CV Scorer Model Training")
    print("=" * 60)
    
    # Generate training data
    print("\n1. Generating training data...")
    df = generate_training_data(n_samples=1000)
    
    # Save training data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/cv_training_data.csv', index=False)
    print(f"   Training data saved: {len(df)} samples")
    
    # Prepare features and target
    # FIX: Use .to_numpy() instead of .values
    X = df.drop('score', axis=1).to_numpy()
    y = df['score'].to_numpy()
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # Train model
    print("\n2. Training model...")
    model = CVScorerModel()
    train_score, test_score = model.train(X, y)
    
    # Get feature importance
    print("\n3. Feature Importance:")
    importance = model.get_feature_importance()
    if importance:
        for feature, imp in importance[:10]:
            print(f"   {feature:30} {imp:.4f}")
    
    # Save model
    print("\n4. Saving model...")
    model.save_model()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Test prediction
    print("\n5. Testing prediction...")
    test_cv_data = {
        'email': 'test@gmail.com',
        'phone': '1234567890',
        'linkedin': 'linkedin.com/in/test',
        'github': 'github.com/test',
        'found_sections': ['contact', 'education', 'experience', 'skills', 'projects'],
        'technical_skills': ['python', 'java', 'sql', 'git', 'docker'],
        'action_verbs': ['developed', 'managed', 'created', 'improved'],
        'word_count': 600,
        'formatting_issues': []
    }
    
    predicted_score = model.predict(test_cv_data)
    print(f"   Test prediction: {predicted_score:.2f}/100")

if __name__ == "__main__":
    main()