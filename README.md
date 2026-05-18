resume-screening-nlp
AI-powered Resume Screening System using NLP (TF-IDF &amp; Cosine Similarity) to match resumes with job descriptions. Built with Python &amp; Streamlit with an interactive UI.

📄 AI Resume Screening System (NLP)
An AI-powered Resume Screening System that automates candidate evaluation using Natural Language Processing (NLP) and Machine Learning. It helps recruiters shortlist candidates based on job descriptions quickly and efficiently.

🚀 Demo
🔗 Add your deployed link here (Streamlit / Render / HuggingFace Spaces)  

📸 Screenshots

![Home UI](screenshots/home.png)  
![Upload UI](screenshots/upload.png)  
![Results UI](screenshots/results.png)

🧠 Features
- Upload resumes (PDF/DOCX)
- Automatic text extraction and preprocessing
- Keyword and skill matching
- Resume scoring based on job description
- NLP-based similarity (TF-IDF, Cosine Similarity)
- Simple and interactive UI (Streamlit)

🛠️ Tech Stack
- Python  
- scikit-learn  
- pandas  
- numpy  
- nltk / spacy  
- PyPDF2 / python-docx  
- Streamlit 

📂 Project Structure
resume-screening-nlp/
├── app.py
├── similarity.py
├── requirements.txt
├── README.md
├── screenshots/
└── sample_data/

⚙️ Installation & Setup

1. Clone the Repository
bash
git clone https://github.com/your-username/ai-resume-screening-system.git
cd ai-resume-screening-system

2. Create Virtual Environment
bash
python -m venv venv
Activate:
- Windows:
bash
venv\Scripts\activate
- Mac/Linux:
bash
source venv/bin/activate

3. Install Dependencies
bash
pip install -r requirements.txt

4. Run the App
bash
streamlit run app.py

📊 How It Works
1. Upload resumes  
2. Enter job description  
3. NLP processes the text  
4. Similarity score is calculated  
5. Candidates are ranked  

📌 Future Improvements
- Deep Learning models (BERT)
- Dashboard analytics
- Cloud deployment (AWS/Docker)
- Email notifications

📈 ATS Keywords
Artificial Intelligence, Machine Learning, NLP, Resume Screening, Recruitment Automation, Python, Scikit-learn, Data Science, Text Processing, Cosine Similarity, Streamlit

Contributing
Feel free to fork this repo and submit pull requests.

MIT License

👨‍💻 Author
Prayag Rajyaguru
LinkedIn:www.linkedin.com/in/prayag-rajyaguru
Email:prayagrajyaguru72@gmail.com  


⭐ Support
If you like this project, give it a ⭐ on GitHub!
