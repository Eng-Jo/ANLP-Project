import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from cv_analyzer import CVAnalyzer
import os

class CVAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CV Analyzer - ML-Powered Resume Analysis")  # Updated title
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        self.analyzer = CVAnalyzer(use_ml=True)  # Enable ML
        self.file_path = None
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', pady=15)
        title_frame.pack(fill='x')
        
        title_label = tk.Label(
            title_frame,
            text="📄 CV Analyzer",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Analyze your resume and get instant feedback",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack()
        
        # File selection frame
        file_frame = tk.Frame(self.root, bg='#f0f0f0', pady=20)
        file_frame.pack(fill='x', padx=20)
        
        self.file_label = tk.Label(
            file_frame,
            text="No file selected",
            font=('Arial', 11),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        self.file_label.pack(side='left', padx=10)
        
        browse_btn = tk.Button(
            file_frame,
            text="📁 Browse CV",
            command=self.browse_file,
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10,
            relief='flat',
            cursor='hand2'
        )
        browse_btn.pack(side='left', padx=5)
        
        analyze_btn = tk.Button(
            file_frame,
            text="🔍 Analyze CV",
            command=self.analyze_cv,
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=10,
            relief='flat',
            cursor='hand2'
        )
        analyze_btn.pack(side='left', padx=5)
        
        # Score frame
        self.score_frame = tk.Frame(self.root, bg='#ecf0f1', pady=20)
        self.score_frame.pack(fill='x', padx=20, pady=10)
        
        self.score_label = tk.Label(
            self.score_frame,
            text="Score: --/100",
            font=('Arial', 32, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        self.score_label.pack()
        
        self.score_status = tk.Label(
            self.score_frame,
            text="Upload a CV to get started",
            font=('Arial', 12),
            bg='#ecf0f1',
            fg='#7f8c8d'
        )
        self.score_status.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.score_frame,
            length=400,
            mode='determinate'
        )
        self.progress.pack(pady=10)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Results tab
        results_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(results_frame, text='📊 Analysis Results')
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=('Arial', 10),
            bg='white',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        self.results_text.pack(fill='both', expand=True)
        
        # Suggestions tab
        suggestions_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(suggestions_frame, text='💡 Suggestions')
        
        self.suggestions_text = scrolledtext.ScrolledText(
            suggestions_frame,
            wrap=tk.WORD,
            font=('Arial', 10),
            bg='white',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        self.suggestions_text.pack(fill='both', expand=True)
        
        # Footer
        footer = tk.Label(
            self.root,
            text="Made with ❤️ for students | Support: contact@cvanalyzer.com",
            font=('Arial', 9),
            bg='#34495e',
            fg='white',
            pady=10
        )
        footer.pack(fill='x', side='bottom')
    
    def browse_file(self):
        """Open file dialog to select CV"""
        file_path = filedialog.askopenfilename(
            title="Select CV File",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("Word files", "*.docx"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Selected: {filename}", fg='#27ae60')
    
    def get_score_status(self, score):
        """Get status message based on score"""
        if score >= 80:
            return "Excellent! 🌟", '#27ae60'
        elif score >= 60:
            return "Good! Keep improving 👍", '#f39c12'
        elif score >= 40:
            return "Needs improvement ⚠️", '#e67e22'
        else:
            return "Needs major revision ❌", '#e74c3c'
    
    def analyze_cv(self):
        """Analyze the selected CV"""
        if not self.file_path:
            messagebox.showwarning("No File", "Please select a CV file first!")
            return
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.suggestions_text.delete(1.0, tk.END)
        
        try:
            # Analyze CV
            results = self.analyzer.analyze(self.file_path)
            
            if "error" in results:
                messagebox.showerror("Error", results["error"])
                return
            
            # Update score
            score = results['score']
            self.score_label.config(text=f"Score: {score}/100")
            self.progress['value'] = score
            
            status_text, status_color = self.get_score_status(score)
            self.score_status.config(text=status_text, fg=status_color)
            
            # Display results
            self.display_results(results)
            
            # Display suggestions
            self.display_suggestions(results['suggestions'])
            
            messagebox.showinfo("Success", "CV analysis completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    # def display_results(self, results):
    #     """Display analysis results"""
    #     self.results_text.insert(tk.END, "=== CV ANALYSIS RESULTS ===\n\n", 'header')
        
    #     # Contact Information
    #     self.results_text.insert(tk.END, "📧 CONTACT INFORMATION:\n", 'subheader')
    #     self.results_text.insert(tk.END, f"  Email: {results['email'] or 'Not found'}\n")
    #     self.results_text.insert(tk.END, f"  Phone: {results['phone'] or 'Not found'}\n\n")
        
    #     # Sections
    #     self.results_text.insert(tk.END, "📑 SECTIONS FOUND:\n", 'subheader')
    #     for section in results['found_sections']:
    #         self.results_text.insert(tk.END, f"  ✓ {section.capitalize()}\n", 'success')
        
    #     if results['missing_sections']:
    #         self.results_text.insert(tk.END, "\n❌ MISSING SECTIONS:\n", 'subheader')
    #         for section in results['missing_sections']:
    #             self.results_text.insert(tk.END, f"  ✗ {section.capitalize()}\n", 'error')
        
    #     # Technical Skills
    #     self.results_text.insert(tk.END, "\n🔧 TECHNICAL SKILLS FOUND:\n", 'subheader')
    #     if results['technical_skills']:
    #         for skill in results['technical_skills']:
    #             self.results_text.insert(tk.END, f"  • {skill}\n")
    #     else:
    #         self.results_text.insert(tk.END, "  No technical skills detected\n", 'error')
        
    #     # Action Verbs
    #     self.results_text.insert(tk.END, "\n💪 ACTION VERBS USED:\n", 'subheader')
    #     if results['action_verbs']:
    #         for verb in results['action_verbs']:
    #             self.results_text.insert(tk.END, f"  • {verb}\n")
    #     else:
    #         self.results_text.insert(tk.END, "  Few action verbs detected\n", 'warning')
        
    #     # Word Count
    #     self.results_text.insert(tk.END, f"\n📝 WORD COUNT: {results['word_count']}\n", 'subheader')
    #     self.results_text.insert(tk.END, f"  {results['length_message']}\n")
        
    #     # Configure tags
    #     self.results_text.tag_config('header', font=('Arial', 14, 'bold'), foreground='#2c3e50')
    #     self.results_text.tag_config('subheader', font=('Arial', 11, 'bold'), foreground='#34495e')
    #     self.results_text.tag_config('success', foreground='#27ae60')
    #     self.results_text.tag_config('error', foreground='#e74c3c')
    #     self.results_text.tag_config('warning', foreground='#f39c12')
    
    def display_results(self, results):
        """Display analysis results"""
        self.results_text.insert(tk.END, "=== CV ANALYSIS RESULTS ===\n\n", 'header')
        
        # Contact Information
        self.results_text.insert(tk.END, "📧 CONTACT INFORMATION:\n", 'subheader')
        
        # Email
        if results['email']:
            self.results_text.insert(tk.END, f"  ✓ Email: {results['email']}\n", 'success')
        else:
            self.results_text.insert(tk.END, f"  ✗ Email: Not found\n", 'error')
        
        # Phone
        if results['phone']:
            self.results_text.insert(tk.END, f"  ✓ Phone: {results['phone']}\n", 'success')
        else:
            self.results_text.insert(tk.END, f"  ✗ Phone: Not found\n", 'error')
        
        # LinkedIn
        if results['linkedin']:
            self.results_text.insert(tk.END, f"  ✓ LinkedIn: {results['linkedin']}\n", 'success')
        else:
            self.results_text.insert(tk.END, f"  ✗ LinkedIn: Not found\n", 'error')
        
        # GitHub
        if results['github']:
            self.results_text.insert(tk.END, f"  ✓ GitHub: {results['github']}\n", 'success')
        else:
            self.results_text.insert(tk.END, f"  ✗ GitHub: Not found\n", 'error')
        
        # Contact Status
        if results['contact_valid']:
            self.results_text.insert(tk.END, f"\n  ✅ Contact Section: VALID\n", 'success')
        else:
            self.results_text.insert(tk.END, f"\n  ❌ Contact Section: INVALID (need email + phone/LinkedIn/GitHub)\n", 'error')
        
        # Sections
        self.results_text.insert(tk.END, "\n📑 SECTIONS FOUND:\n", 'subheader')
        for section in results['found_sections']:
            self.results_text.insert(tk.END, f"  ✓ {section.capitalize()}\n", 'success')
        
        if results['missing_sections']:
            self.results_text.insert(tk.END, "\n❌ MISSING SECTIONS:\n", 'subheader')
            for section in results['missing_sections']:
                self.results_text.insert(tk.END, f"  ✗ {section.capitalize()}\n", 'error')
        
        # Technical Skills
        self.results_text.insert(tk.END, "\n🔧 TECHNICAL SKILLS FOUND:\n", 'subheader')
        if results['technical_skills']:
            for skill in results['technical_skills']:
                self.results_text.insert(tk.END, f"  • {skill}\n")
        else:
            self.results_text.insert(tk.END, "  No technical skills detected\n", 'error')
        
        # Action Verbs
        self.results_text.insert(tk.END, "\n💪 ACTION VERBS USED:\n", 'subheader')
        if results['action_verbs']:
            for verb in results['action_verbs']:
                self.results_text.insert(tk.END, f"  • {verb}\n")
        else:
            self.results_text.insert(tk.END, "  Few action verbs detected\n", 'warning')
        
        # Word Count
        self.results_text.insert(tk.END, f"\n📝 WORD COUNT: {results['word_count']}\n", 'subheader')
        self.results_text.insert(tk.END, f"  {results['length_message']}\n")
        
        # Configure tags
        self.results_text.tag_config('header', font=('Arial', 14, 'bold'), foreground='#2c3e50')
        self.results_text.tag_config('subheader', font=('Arial', 11, 'bold'), foreground='#34495e')
        self.results_text.tag_config('success', foreground='#27ae60')
        self.results_text.tag_config('error', foreground='#e74c3c')
        self.results_text.tag_config('warning', foreground='#f39c12')


    def display_suggestions(self, suggestions):
        """Display improvement suggestions"""
        self.suggestions_text.insert(tk.END, "=== IMPROVEMENT SUGGESTIONS ===\n\n", 'header')
        
        for i, suggestion in enumerate(suggestions, 1):
            self.suggestions_text.insert(tk.END, f"{i}. {suggestion}\n\n")
        
        self.suggestions_text.tag_config('header', font=('Arial', 14, 'bold'), foreground='#2c3e50')

def main():
    root = tk.Tk()
    app = CVAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()