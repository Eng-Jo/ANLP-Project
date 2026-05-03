# CV Analyzer with Machine Learning

An AI-powered CV/Resume analyzer using Machine Learning (Random Forest) to provide intelligent scoring and feedback.

## Features

- ✅ **Machine Learning Model**: Random Forest Regressor for accurate CV scoring
- ✅ **Feature Engineering**: 15 engineered features from CV analysis
- ✅ **Dual Scoring**: ML-based scoring with rule-based fallback
- ✅ **Feature Importance Analysis**: Shows which aspects matter most
- ✅ **Contact Detection**: Email, phone, LinkedIn, GitHub
- ✅ **Skills Analysis**: Technical skills and action verbs detection
- ✅ **Section Validation**: Checks for required CV sections
- ✅ **Intelligent Suggestions**: ML-powered improvement recommendations

## Machine Learning Model

### Model Architecture
- **Algorithm**: Random Forest Regressor
- **Features**: 15 engineered features
- **Training Data**: 1000 synthetic CV samples
- **Performance**: ~95% R² score on test data

### Features Used
1. Contact Information (email, phone, LinkedIn, GitHub)
2. Section Completeness
3. Technical Skills Count
4. Action Verbs Usage
5. Word Count (normalized)
6. Formatting Quality
7. Education Quality
8. Experience Quality
9. Project Presence
10. Quantifiable Achievements
11. Keyword Density

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Train ML model
python train_model.py

# Run application
python main.py