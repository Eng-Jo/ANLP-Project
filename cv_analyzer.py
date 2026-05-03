import re
import PyPDF2
import docx
from textblob import TextBlob
import nltk
from ml_model import CVScorerModel
import os
from typing import Dict, List, Optional, Tuple, Any

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class CVAnalyzer:
    def __init__(self, use_ml: bool = True):
        self.use_ml = use_ml
        self.ml_model: Optional[CVScorerModel] = None
        
        # Load ML model if available
        if use_ml:
            try:
                self.ml_model = CVScorerModel()
                self.ml_model.load_model('models/cv_scorer_model.pkl')
                print("✓ ML Model loaded successfully")
            except Exception as e:
                print(f"⚠ ML Model not found, using rule-based scoring: {e}")
                self.use_ml = False
        
        self.required_sections: List[str] = [
            'contact', 'education', 'experience', 'skills', 'projects'
        ]
        
        self.technical_skills: List[str] = [
            'python', 'java', 'javascript', 'c', 'c++', 'c#', '.net', 'sql', 'html', 'css',
            'react', 'angular', 'node.js', 'django', 'flask', 'mongodb',
            'postgresql', 'git', 'github', 'docker', 'kubernetes', 'aws', 'azure',
            'machine learning', 'nlp', 'data analysis', 'tensorflow', 'pytorch',
            'windows forms', 'ado.net', 'visual studio', 'sql server',
            'asp.net', 'entity framework', 'oop', 'rest api', 'web api'
        ]
        
        self.action_verbs: List[str] = [
            'achieved', 'developed', 'managed', 'created', 'improved',
            'increased', 'decreased', 'implemented', 'designed', 'led',
            'coordinated', 'analyzed', 'built', 'established', 'launched',
            'engineered', 'architected', 'optimized', 'configured', 'integrated',
            'maintained', 'enhanced', 'applied', 'performed'
        ]
        
        self.email_providers: List[str] = [
            'gmail', 'yahoo', 'outlook', 'hotmail', 'icloud',
            'live', 'aol', 'protonmail', 'zoho', 'mail', 'yandex'
        ]

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"

    def extract_text(self, file_path: str) -> str:
        """Extract text based on file type"""
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            return self.extract_text_from_docx(file_path)
        else:
            return "Unsupported file format"

    def extract_email(self, text: str) -> Optional[str]:
        """Extract email from CV"""
        cleaned_text = re.sub(r'\s+', ' ', text)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        emails = re.findall(email_pattern, cleaned_text)
        
        if not emails:
            partial_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{1,}\b'
            emails = re.findall(partial_pattern, cleaned_text)
        
        if emails:
            for email in emails:
                email_lower = email.lower()
                for provider in self.email_providers:
                    if provider in email_lower:
                        return email
            return emails[0]
        
        return None

    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from CV"""
        phone_patterns = [
            r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{3,4}[-\s\.]?[0-9]{3,9}',
            r'\b\d{10,15}\b',
            r'\+\d{1,3}\s?\d{3,14}',
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                for phone in phones:
                    digits_only = re.sub(r'\D', '', phone)
                    if 10 <= len(digits_only) <= 15:
                        return phone.strip()
        
        return None

    def extract_linkedin(self, text: str) -> Optional[str]:
        """Extract LinkedIn URL or mention"""
        text_lower = text.lower()
        linkedin_patterns = [
            r'linkedin\.com/in/[A-Za-z0-9\-_]+',
            r'linkedin\.com/pub/[A-Za-z0-9\-_]+',
            r'www\.linkedin\.com/[A-Za-z0-9\-_/]+',
        ]
        
        for pattern in linkedin_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                return matches[0]
        
        if 'linkedin' in text_lower:
            return "LinkedIn (mentioned)"
        
        return None

    def extract_github(self, text: str) -> Optional[str]:
        """Extract GitHub URL or mention"""
        text_lower = text.lower()
        github_patterns = [
            r'github\.com/[A-Za-z0-9\-_]+',
            r'www\.github\.com/[A-Za-z0-9\-_]+',
        ]
        
        for pattern in github_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                return matches[0]
        
        if 'github' in text_lower:
            return "GitHub (mentioned)"
        
        return None

    def check_contact_section(self, email: Optional[str], phone: Optional[str], 
                             linkedin: Optional[str], github: Optional[str]) -> bool:
        """Check if contact section is valid"""
        has_email = email is not None
        has_other_contact = phone is not None or linkedin is not None or github is not None
        return has_email and has_other_contact

    def check_sections(self, text: str, contact_valid: bool) -> Tuple[List[str], List[str]]:
        """Check for required sections in CV"""
        text_lower = text.lower()
        found_sections: List[str] = []
        missing_sections: List[str] = []
        
        section_keywords = {
            'education': ['education', 'academic', 'university', 'degree', 'bachelor', 'master', 'graduation'],
            'experience': ['experience', 'work history', 'employment', 'job', 'internship', 'intership'],
            'skills': ['skills', 'technical skills', 'competencies', 'technologies'],
            'projects': ['projects', 'portfolio', 'project']
        }
        
        if contact_valid:
            found_sections.append('contact')
        else:
            missing_sections.append('contact')
        
        for section, keywords in section_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_sections.append(section)
            else:
                missing_sections.append(section)
        
        return found_sections, missing_sections

    def analyze_technical_skills(self, text: str) -> List[str]:
        """Analyze technical skills mentioned"""
        text_lower = text.lower()
        found_skills: List[str] = []
        
        for skill in self.technical_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills

    def check_action_verbs(self, text: str) -> List[str]:
        """Check usage of strong action verbs"""
        text_lower = text.lower()
        found_verbs: List[str] = []
        
        for verb in self.action_verbs:
            if verb in text_lower:
                found_verbs.append(verb)
        
        return found_verbs

    def calculate_word_count(self, text: str) -> int:
        """Calculate total word count"""
        words = text.split()
        return len(words)

    def check_length(self, word_count: int) -> Tuple[str, str]:
        """Check if CV length is appropriate"""
        if word_count < 300:
            return "too_short", "CV is too short. Aim for 300-800 words."
        elif word_count > 1000:
            return "too_long", "CV is too long. Try to keep it concise (300-800 words)."
        else:
            return "appropriate", "CV length is appropriate."

    def analyze_formatting(self, text: str) -> List[str]:
        """Basic formatting analysis"""
        issues: List[str] = []
        
        if '•' not in text and '-' not in text[:100]:
            issues.append("Consider using bullet points for better readability")
        
        if text.isupper():
            issues.append("Avoid using ALL CAPS throughout the CV")
        
        return issues

    def calculate_score_rule_based(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate score using rule-based method (fallback)"""
        score = 0.0
        
        if analysis_results['email']:
            score += 7.5
        
        contact_methods = sum([
            1 if analysis_results['phone'] else 0,
            1 if analysis_results['linkedin'] else 0,
            1 if analysis_results['github'] else 0
        ])
        if contact_methods >= 1:
            score += 7.5
        
        sections_score = (len(analysis_results['found_sections']) / len(self.required_sections)) * 30
        score += sections_score
        
        skills_count = len(analysis_results['technical_skills'])
        skills_score = min(skills_count * 2, 20)
        score += skills_score
        
        verbs_count = len(analysis_results['action_verbs'])
        verbs_score = min(verbs_count * 1.5, 15)
        score += verbs_score
        
        if analysis_results['length_status'] == 'appropriate':
            score += 10
        elif analysis_results['length_status'] == 'too_short':
            score += 5
        
        if len(analysis_results['formatting_issues']) == 0:
            score += 10
        else:
            score += max(10 - len(analysis_results['formatting_issues']) * 2, 0)
        
        return round(score, 2)

    def calculate_score_ml(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate score using ML model"""
        if self.ml_model is None:
            print("ML model not available, using rule-based scoring")
            return self.calculate_score_rule_based(analysis_results)
            
        try:
            score = self.ml_model.predict(analysis_results)
            return round(score, 2)
        except Exception as e:
            print(f"ML prediction failed: {e}, using rule-based scoring")
            return self.calculate_score_rule_based(analysis_results)

    def generate_suggestions(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions: List[str] = []
        
        if not analysis_results['email']:
            suggestions.append("❌ Add your email address (Gmail, Yahoo, Outlook, etc.)")
        
        contact_methods: List[str] = []
        if analysis_results['phone']:
            contact_methods.append('phone')
        if analysis_results['linkedin']:
            contact_methods.append('LinkedIn')
        if analysis_results['github']:
            contact_methods.append('GitHub')
        
        if not contact_methods:
            suggestions.append("❌ Add at least one of: Phone number, LinkedIn profile, or GitHub link")
        elif len(contact_methods) < 2:
            suggestions.append(f"⚠️ Consider adding more contact methods. You have: {', '.join(contact_methods)}")
        
        if analysis_results['missing_sections']:
            for section in analysis_results['missing_sections']:
                if section == 'contact':
                    suggestions.append("❌ Complete your Contact section (need email + phone/LinkedIn/GitHub)")
                else:
                    suggestions.append(f"❌ Add '{section.capitalize()}' section")
        
        if len(analysis_results['technical_skills']) < 5:
            suggestions.append("⚠️ Include more technical skills (mention tools, languages, frameworks)")
        
        if len(analysis_results['action_verbs']) < 5:
            suggestions.append("⚠️ Use more strong action verbs (achieved, developed, managed, etc.)")
        
        if analysis_results['length_status'] != 'appropriate':
            suggestions.append(f"⚠️ {analysis_results['length_message']}")
        
        for issue in analysis_results['formatting_issues']:
            suggestions.append(f"⚠️ {issue}")
        
        # ML-based suggestions
        if self.use_ml and self.ml_model:
            feature_importance = self.ml_model.get_feature_importance()
            if feature_importance:
                top_features = [f[0] for f in feature_importance[:3]]
                suggestions.append(f"💡 ML Insight: Focus on improving: {', '.join(top_features)}")
        
        suggestions.append("✓ Use quantifiable achievements (e.g., 'Increased efficiency by 30%')")
        suggestions.append("✓ Tailor your CV to the specific job you're applying for")
        suggestions.append("✓ Keep formatting consistent throughout")
        
        return suggestions

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Main analysis function"""
        text = self.extract_text(file_path)
        
        if "Error" in text:
            return {"error": text}
        
        email = self.extract_email(text)
        phone = self.extract_phone(text)
        linkedin = self.extract_linkedin(text)
        github = self.extract_github(text)
        
        contact_valid = self.check_contact_section(email, phone, linkedin, github)
        
        found_sections, missing_sections = self.check_sections(text, contact_valid)
        technical_skills = self.analyze_technical_skills(text)
        action_verbs = self.check_action_verbs(text)
        word_count = self.calculate_word_count(text)
        length_status, length_message = self.check_length(word_count)
        formatting_issues = self.analyze_formatting(text)
        
        analysis_results: Dict[str, Any] = {
            'email': email,
            'phone': phone,
            'linkedin': linkedin,
            'github': github,
            'contact_valid': contact_valid,
            'found_sections': found_sections,
            'missing_sections': missing_sections,
            'technical_skills': technical_skills,
            'action_verbs': action_verbs,
            'word_count': word_count,
            'length_status': length_status,
            'length_message': length_message,
            'formatting_issues': formatting_issues
        }
        
        # Calculate score using ML or rule-based
        if self.use_ml and self.ml_model:
            score = self.calculate_score_ml(analysis_results)
            analysis_results['scoring_method'] = 'Machine Learning'
        else:
            score = self.calculate_score_rule_based(analysis_results)
            analysis_results['scoring_method'] = 'Rule-Based'
        
        analysis_results['score'] = score
        
        suggestions = self.generate_suggestions(analysis_results)
        analysis_results['suggestions'] = suggestions
        
        return analysis_results