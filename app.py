import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

class SpamClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Email/SMS Spam Classifier")
        self.root.geometry("800x600")
        
        # Load models
        try:
            self.legacy_tfidf = pickle.load(open('./models/vectorizer.pkl', 'rb'))
            self.legacy_model = pickle.load(open('./models/model.pkl', 'rb'))
            self.optimized_tfidf = pickle.load(open('./models/vectorizer_optimized.pkl', 'rb'))
            self.optimized_model = pickle.load(open('./models/model_optimized.pkl', 'rb'))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            root.destroy()
            return
            
        self.setup_ui()
        
    def setup_ui(self):
        # Configure dark theme styles using the 'clam' base theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Global background & text
        style.configure('.', background='#0f172a', foreground='#f8fafc', font=('Segoe UI', 10))
        
        # Frame
        style.configure('TFrame', background='#0f172a')
        
        # LabelFrames
        style.configure('TLabelframe', background='#1e293b', bordercolor='#334155', borderwidth=1, relief='flat')
        style.configure('TLabelframe.Label', background='#1e293b', foreground='#a5b4fc', font=('Segoe UI', 10, 'bold'))
        
        # Labels
        style.configure('TLabel', background='#0f172a', foreground='#f8fafc')
        style.configure('Card.TLabel', background='#1e293b', foreground='#f8fafc')
        style.configure('Header.TLabel', background='#0f172a', foreground='#f8fafc', font=('Segoe UI', 20, 'bold'))
        style.configure('Subheader.TLabel', background='#0f172a', foreground='#94a3b8', font=('Segoe UI', 10))
        
        # Radio buttons
        style.configure('TRadiobutton', background='#1e293b', foreground='#cbd5e1', font=('Segoe UI', 10))
        style.map('TRadiobutton',
                  background=[('active', '#1e293b'), ('selected', '#1e293b')],
                  foreground=[('active', '#a5b4fc'), ('selected', '#a5b4fc')])
        
        # Buttons
        style.configure('TButton', background='#6366f1', foreground='#ffffff', font=('Segoe UI', 11, 'bold'), borderwidth=0, padding=(20, 10))
        style.map('TButton',
                  background=[('active', '#4f46e5')],
                  foreground=[('active', '#ffffff')])

        # Apply dark background to root window
        self.root.configure(bg='#0f172a')
        
        # Header container
        header_frame = ttk.Frame(self.root, style='TFrame')
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        header_title = ttk.Label(header_frame, text="✉️ Spam Detection System", style='Header.TLabel')
        header_title.pack(anchor="w")
        
        header_sub = ttk.Label(header_frame, text="Classify messages instantly using optimized MBO or legacy ensemble models.", style='Subheader.TLabel')
        header_sub.pack(anchor="w", pady=(2, 0))
        
        # Model selection frame
        model_frame = ttk.LabelFrame(self.root, text=" MODEL SELECTION ", padding=12)
        model_frame.pack(fill="x", padx=20, pady=10)
        
        self.model_var = tk.StringVar(value="optimized")
        
        opt_radio = ttk.Radiobutton(model_frame, text="Optimized Model (MBO/Lite Preprocessing)", 
                       value="optimized", variable=self.model_var)
        opt_radio.pack(side=tk.LEFT, padx=15, pady=5)
        
        legacy_radio = ttk.Radiobutton(model_frame, text="Legacy Model (Traditional Preprocessing)", 
                       value="legacy", variable=self.model_var)
        legacy_radio.pack(side=tk.LEFT, padx=15, pady=5)
        
        # Input area frame
        input_frame = ttk.LabelFrame(self.root, text=" ENTER MESSAGE TO CLASSIFY ", padding=12)
        input_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Custom-styled Text widget to match dark mode
        self.input_text = scrolledtext.ScrolledText(
            input_frame, 
            height=8,
            bg='#0f172a',
            fg='#f8fafc',
            insertbackground='#f8fafc',
            relief='flat',
            borderwidth=0,
            font=('Segoe UI', 11),
            padx=10,
            pady=10
        )
        self.input_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Predict button frame
        btn_frame = ttk.Frame(self.root, style='TFrame')
        btn_frame.pack(fill="x", padx=20, pady=10)
        
        self.predict_btn = ttk.Button(btn_frame, text="Run Classification", command=self.predict)
        self.predict_btn.pack(anchor="center")
        
        # Result display frame
        result_frame = ttk.LabelFrame(self.root, text=" PREDICTION RESULT ", padding=12)
        result_frame.pack(fill="x", padx=20, pady=(10, 20))
        
        self.result_label = ttk.Label(
            result_frame, 
            text="Waiting for input...", 
            font=("Segoe UI", 14, "bold"),
            style='Card.TLabel',
            anchor="center"
        )
        self.result_label.pack(fill="x", pady=10)
        
    def transform_text_legacy(self, text):
        ps = PorterStemmer()
        text = text.lower()
        text = nltk.word_tokenize(text)
        
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
                
        text = y[:]
        y.clear()
        
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
                
        text = y[:]
        y.clear()
        
        for i in text:
            y.append(ps.stem(i))
            
        return " ".join(y)
        
    def transform_text_optimized(self, text):
        lemmatizer = WordNetLemmatizer()
        text = str(text).lower()
        words = nltk.word_tokenize(text)
        
        words = [lemmatizer.lemmatize(word) for word in words 
                if word.isalnum() and 
                word not in stopwords.words('english') and 
                word not in string.punctuation]
        
        return " ".join(words)
    
    def predict(self):
        text = self.input_text.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter a message")
            return
            
        try:
            if self.model_var.get() == "optimized":
                transformed_text = self.transform_text_optimized(text)
                vector = self.optimized_tfidf.transform([transformed_text])
                prediction = self.optimized_model.predict(vector)[0]
            else:
                transformed_text = self.transform_text_legacy(text)
                vector = self.legacy_tfidf.transform([transformed_text]).toarray()
                prediction = self.legacy_model.predict(vector)[0]
                
            result = "🚨 SPAM DETECTED" if prediction == 1 else "✅ NOT SPAM (CLEAN MESSAGE)"
            color = "#f43f5e" if prediction == 1 else "#10b981"
            self.result_label.config(text=result, foreground=color)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            
if __name__ == "__main__":
    try:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        
        root = tk.Tk()
        app = SpamClassifierUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")