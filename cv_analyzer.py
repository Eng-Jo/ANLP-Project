import re
import PyPDF2
import docx
from textblob import TextBlob
import nltk
from collections import Counter

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class CVAnalyzer:
    def __init__(self):
        self.required_sections = [
            'contact', 'education', 'experience', 'skills', 'projects'
        ]
        
        self.technical_skills = [
            'python', 'java', 'javascript', 'c++', 'c#', '.net', 'sql', 'html', 'css',
            'react', 'angular', 'node.js', 'django', 'flask', 'mongodb',
            'postgresql', 'git', 'github', 'docker', 'kubernetes', 'aws', 'azure',
            'machine learning', 'data analysis', 'tensorflow', 'pytorch',
            'windows forms', 'ado.net', 'visual studio', 'sql server',
            'asp.net', 'entity framework', 'oop', 'rest api', 'web api'
        ]
        
        self.action_verbs = [
            'achieved', 'developed', 'managed', 'created', 'improved',
            'increased', 'decreased', 'implemented', 'designed', 'led',
            'coordinated', 'analyzed', 'built', 'established', 'launched',
            'engineered', 'architected', 'optimized', 'configured', 'integrated',
            'maintained', 'enhanced', 'applied', 'performed'
        ]
        
        # Common email providers
        self.email_providers = [
            'gmail', 'yahoo', 'outlook', 'hotmail', 'icloud',
            'live', 'aol', 'protonmail', 'zoho', 'mail', 'yandex'
        ]

    def extract_text_from_pdf(self, file_path):
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

    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"

    def extract_text(self, file_path):
        """Extract text based on file type"""
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            return self.extract_text_from_docx(file_path)
        else:
            return "Unsupported file format"

    def extract_email(self, text):
        """Extract email from CV - supports common providers and broken/spaced emails"""
        # Clean text - remove extra spaces that might break email
        cleaned_text = re.sub(r'\s+', ' ', text)
        
        # Standard email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        emails = re.findall(email_pattern, cleaned_text)
        
        # Try to find emails even if they have spaces or are cut off
        if not emails:
            # Pattern for emails that might be truncated (e.g., "user@yahoo.co" missing 'm')
            partial_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{1,}\b'
            emails = re.findall(partial_pattern, cleaned_text)
        
        # Validate the email by checking for common providers
        if emails:
            for email in emails:
                email_lower = email.lower()
                for provider in self.email_providers:
                    if provider in email_lower:
                        return email
            # Return first email even if not from common providers
            return emails[0]
        
        return None

    def extract_phone(self, text):
        """Extract phone number from CV"""
        # Multiple phone patterns to catch different formats
        phone_patterns = [
            r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{3,4}[-\s\.]?[0-9]{3,9}',
            r'\b\d{10,15}\b',  # Simple 10-15 digit numbers
            r'\+\d{1,3}\s?\d{3,14}',  # International format
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                # Filter out numbers that are too short or too long
                for phone in phones:
                    digits_only = re.sub(r'\D', '', phone)
                    if 10 <= len(digits_only) <= 15:
                        return phone.strip()
        
        return None

    def extract_linkedin(self, text):
        """Extract LinkedIn URL or mention"""
        text_lower = text.lower()
        
        # LinkedIn URL patterns
        linkedin_patterns = [
            r'linkedin\.com/in/[A-Za-z0-9\-_]+',
            r'linkedin\.com/pub/[A-Za-z0-9\-_]+',
            r'www\.linkedin\.com/[A-Za-z0-9\-_/]+',
        ]
        
        for pattern in linkedin_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                return matches[0]
        
        # Check if 'linkedin' word is mentioned
        if 'linkedin' in text_lower:
            return "LinkedIn (mentioned)"
        
        return None

    def extract_github(self, text):
        """Extract GitHub URL or mention"""
        text_lower = text.lower()
        
        # GitHub URL patterns
        github_patterns = [
            r'github\.com/[A-Za-z0-9\-_]+',
            r'www\.github\.com/[A-Za-z0-9\-_]+',
        ]
        
        for pattern in github_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                return matches[0]
        
        # Check if 'github' word is mentioned
        if 'github' in text_lower:
            return "GitHub (mentioned)"
        
        return None

    def check_contact_section(self, email, phone, linkedin, github):
        """
        Check if contact section is valid.
        Valid if: email exists AND (phone OR linkedin OR github) exists
        """
        has_email = email is not None
        has_other_contact = phone is not None or linkedin is not None or github is not None
        
        return has_email and has_other_contact

    def check_sections(self, text, contact_valid):
        """Check for required sections in CV"""
        text_lower = text.lower()
        found_sections = []
        missing_sections = []
        
        section_keywords = {
            'education': ['education', 'academic', 'university', 'degree', 'bachelor', 'master', 'graduation'],
            'experience': ['experience', 'work history', 'employment', 'job', 'internship', 'intership'],
            'skills': ['skills', 'technical skills', 'competencies', 'technologies'],
            'projects': ['projects', 'portfolio', 'project']
        }
        
        # Add contact section based on validation
        if contact_valid:
            found_sections.append('contact')
        else:
            missing_sections.append('contact')
        
        # Check other sections
        for section, keywords in section_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_sections.append(section)
            else:
                missing_sections.append(section)
        
        return found_sections, missing_sections

    def analyze_technical_skills(self, text):
        """Analyze technical skills mentioned"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.technical_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills

    def check_action_verbs(self, text):
        """Check usage of strong action verbs"""
        text_lower = text.lower()
        found_verbs = []
        
        for verb in self.action_verbs:
            if verb in text_lower:
                found_verbs.append(verb)
        
        return found_verbs

    def calculate_word_count(self, text):
        """Calculate total word count"""
        words = text.split()
        return len(words)

    def check_length(self, word_count):
        """Check if CV length is appropriate"""
        if word_count < 300:
            return "too_short", "CV is too short. Aim for 300-800 words."
        elif word_count > 1000:
            return "too_long", "CV is too long. Try to keep it concise (300-800 words)."
        else:
            return "appropriate", "CV length is appropriate."

    def analyze_formatting(self, text):
        """Basic formatting analysis"""
        issues = []
        
        # Check for bullet points
        if '•' not in text and '-' not in text[:100]:
            issues.append("Consider using bullet points for better readability")
        
        # Check for all caps
        if text.isupper():
            issues.append("Avoid using ALL CAPS throughout the CV")
        
        return issues

    def calculate_score(self, analysis_results):
        """Calculate overall CV score (0-100)"""
        score = 0
        
        # Contact Information (15 points)
        # Email is required
        if analysis_results['email']:
            score += 7.5
        
        # At least one of: phone, LinkedIn, GitHub
        contact_methods = sum([
            1 if analysis_results['phone'] else 0,
            1 if analysis_results['linkedin'] else 0,
            1 if analysis_results['github'] else 0
        ])
        if contact_methods >= 1:
            score += 7.5
        
        # Sections (30 points)
        sections_score = (len(analysis_results['found_sections']) / len(self.required_sections)) * 30
        score += sections_score
        
        # Technical skills (20 points)
        skills_count = len(analysis_results['technical_skills'])
        skills_score = min(skills_count * 2, 20)
        score += skills_score
        
        # Action verbs (15 points)
        verbs_count = len(analysis_results['action_verbs'])
        verbs_score = min(verbs_count * 1.5, 15)
        score += verbs_score
        
        # Length (10 points)
        if analysis_results['length_status'] == 'appropriate':
            score += 10
        elif analysis_results['length_status'] == 'too_short':
            score += 5
        
        # Formatting (10 points)
        if len(analysis_results['formatting_issues']) == 0:
            score += 10
        else:
            score += max(10 - len(analysis_results['formatting_issues']) * 2, 0)
        
        return round(score, 2)

    def generate_suggestions(self, analysis_results):
        """Generate improvement suggestions"""
        suggestions = []
        
        # Contact information
        if not analysis_results['email']:
            suggestions.append("❌ Add your email address (Gmail, Yahoo, Outlook, etc.)")
        
        contact_methods = []
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
        
        # Missing sections
        if analysis_results['missing_sections']:
            for section in analysis_results['missing_sections']:
                if section == 'contact':
                    suggestions.append("❌ Complete your Contact section (need email + phone/LinkedIn/GitHub)")
                else:
                    suggestions.append(f"❌ Add '{section.capitalize()}' section")
        
        # Technical skills
        if len(analysis_results['technical_skills']) < 5:
            suggestions.append("⚠️ Include more technical skills (mention tools, languages, frameworks)")
        
        # Action verbs
        if len(analysis_results['action_verbs']) < 5:
            suggestions.append("⚠️ Use more strong action verbs (achieved, developed, managed, etc.)")
        
        # Length
        if analysis_results['length_status'] != 'appropriate':
            suggestions.append(f"⚠️ {analysis_results['length_message']}")
        
        # Formatting
        for issue in analysis_results['formatting_issues']:
            suggestions.append(f"⚠️ {issue}")
        
        # General tips
        suggestions.append("✓ Use quantifiable achievements (e.g., 'Increased efficiency by 30%')")
        suggestions.append("✓ Tailor your CV to the specific job you're applying for")
        suggestions.append("✓ Keep formatting consistent throughout")
        suggestions.append("✓ Use a professional font (Arial, Calibri, Times New Roman)")
        
        return suggestions

    def analyze(self, file_path):
        """Main analysis function"""
        # Extract text
        text = self.extract_text(file_path)
        
        if "Error" in text:
            return {"error": text}
        
        # Extract contact information
        email = self.extract_email(text)
        phone = self.extract_phone(text)
        linkedin = self.extract_linkedin(text)
        github = self.extract_github(text)
        
        # Validate contact section
        contact_valid = self.check_contact_section(email, phone, linkedin, github)
        
        # Perform other analysis
        found_sections, missing_sections = self.check_sections(text, contact_valid)
        technical_skills = self.analyze_technical_skills(text)
        action_verbs = self.check_action_verbs(text)
        word_count = self.calculate_word_count(text)
        length_status, length_message = self.check_length(word_count)
        formatting_issues = self.analyze_formatting(text)
        
        # Compile results
        analysis_results = {
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
        
        # Calculate score
        score = self.calculate_score(analysis_results)
        analysis_results['score'] = score
        
        # Generate suggestions
        suggestions = self.generate_suggestions(analysis_results)
        analysis_results['suggestions'] = suggestions
        
        return analysis_results