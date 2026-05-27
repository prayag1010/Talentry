import streamlit as st
import nltk
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

# ─── NLTK DOWNLOAD (runs every startup for cloud compatibility) ───────────────
def download_nltk_data():
    """
    Download required NLTK data. Runs at module level (not cached) so that
    Streamlit Cloud always has the data after a cold restart.
    """
    packages = ["stopwords", "punkt", "punkt_tab"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass  # already present or download failed gracefully

download_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Animated Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0818, #1a1040, #0d1b3e, #1a0a2e);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Decorative Blobs ── */
.stApp::before {
    content: '';
    position: fixed;
    top: -200px;
    left: -200px;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, #7c3aed22, transparent 70%);
    border-radius: 50%;
    animation: blobFloat 8s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

.stApp::after {
    content: '';
    position: fixed;
    bottom: -150px;
    right: -150px;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, #2563eb22, transparent 70%);
    border-radius: 50%;
    animation: blobFloat 10s ease-in-out infinite reverse;
    pointer-events: none;
    z-index: 0;
}

@keyframes blobFloat {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%       { transform: translate(30px, -40px) scale(1.05); }
    66%       { transform: translate(-20px, 20px) scale(0.96); }
}

/* ── Hero title ── */
.hero-title {
    text-align: center;
    font-size: 3.6rem;
    font-weight: 900;
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(90deg, #c084fc, #818cf8, #38bdf8, #34d399);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: titleGlow 4s ease infinite;
    margin-bottom: 0.3rem;
    line-height: 1.15;
    letter-spacing: -0.02em;
    text-shadow: none;
    filter: drop-shadow(0 0 30px #7c3aed44);
}

@keyframes titleGlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 0.8rem;
}

/* ── Pill badge ── */
.badge {
    display: inline-block;
    background: linear-gradient(90deg, #7c3aed33, #2563eb33);
    border: 1px solid #7c3aed55;
    color: #c4b5fd;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.badge-center {
    text-align: center;
}

/* ── Glass card wrapper ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 48px rgba(124,58,237,0.25), inset 0 1px 0 rgba(255,255,255,0.1);
}

/* ── Upload cards ── */
.upload-label {
    color: #c4b5fd;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 8px;
    letter-spacing: 0.02em;
}

/* ── Score circle ── */
.score-circle {
    width: 160px;
    height: 160px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1.5rem;
    font-weight: 800;
    border: 4px solid;
    box-shadow: 0 0 30px;
    transition: all 0.4s ease;
}

.score-high {
    color: #34d399;
    border-color: #34d399;
    box-shadow: 0 0 30px #34d39966;
    background: #34d39911;
}

.score-mid {
    color: #fbbf24;
    border-color: #fbbf24;
    box-shadow: 0 0 30px #fbbf2466;
    background: #fbbf2411;
}

.score-low {
    color: #f87171;
    border-color: #f87171;
    box-shadow: 0 0 30px #f8717166;
    background: #f8717111;
}

.score-number {
    font-size: 2.6rem;
    line-height: 1;
}

.score-label {
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 4px;
    opacity: 0.85;
}

/* ── Section headers ── */
.section-header {
    color: #e2e8f0;
    font-size: 1.0rem;
    font-weight: 700;
    border-left: 3px solid #7c3aed;
    padding-left: 10px;
    margin: 1.2rem 0 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Skill badges ── */
.skill-tag {
    display: inline-block;
    background: #1e1b4b;
    border: 1px solid #4338ca55;
    color: #a5b4fc;
    font-size: 0.78rem;
    padding: 3px 10px;
    border-radius: 999px;
    margin: 3px 3px;
    font-weight: 500;
}

.skill-tag-match {
    background: #064e3b;
    border-color: #059669;
    color: #6ee7b7;
}

/* ── Divider ── */
.fancy-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #7c3aed88, #60a5fa88, transparent);
    margin: 2rem 0;
    animation: dividerPulse 3s ease-in-out infinite;
}

@keyframes dividerPulse {
    0%, 100% { opacity: 0.5; }
    50%       { opacity: 1; }
}

/* ── Footer ── */
.footer-container {
    margin-top: 3rem;
    padding: 2.5rem 1rem 2rem;
    border-top: 1px solid rgba(124,58,237,0.25);
    background: linear-gradient(180deg, transparent, rgba(124,58,237,0.06));
    text-align: center;
    position: relative;
}

.footer-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #c084fc, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.footer-tagline {
    color: #64748b;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.footer-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, #38bdf8);
    border-radius: 999px;
    margin: 0.8rem auto 1.2rem;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}

.footer-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 500;
    text-decoration: none;
    padding: 5px 14px;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.15);
    transition: all 0.25s ease;
    background: rgba(255,255,255,0.03);
}

.footer-link:hover {
    color: #c4b5fd;
    border-color: #7c3aed66;
    background: rgba(124,58,237,0.12);
    transform: translateY(-2px);
}

.footer-copy {
    color: #334155;
    font-size: 0.74rem;
    margin-top: 0.8rem;
}

.footer-heart {
    color: #f43f5e;
    animation: heartBeat 1.4s ease-in-out infinite;
    display: inline-block;
}

@keyframes heartBeat {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.25); }
}

/* ── Button style ── */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #4f46e5, #2563eb) !important;
    background-size: 200% auto !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2.5rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.3s ease !important;
    width: 100%;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}

.stButton > button:hover {
    background-position: right center !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.6) !important;
    transform: translateY(-2px) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(124,58,237,0.05) !important;
    border: 1.5px dashed #7c3aed88 !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
    transition: border-color 0.3s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #a78bfa !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #7c3aed, #818cf8, #34d399) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 10px #7c3aed66 !important;
}

/* ── Stat chips ── */
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 4px 14px;
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 3px;
}
</style>
""", unsafe_allow_html=True)


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def extract_text_from_pdf(file) -> str:
    """Extract text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip()


def extract_text(uploaded_file) -> str:
    """Extract text from PDF or TXT uploaded file."""
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def preprocess(text: str) -> str:
    """Lowercase, tokenize, remove stopwords & non-alpha tokens."""
    tokens = word_tokenize(text.lower())
    filtered = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
    return " ".join(filtered)


def compute_similarity(resume_text: str, jd_text: str) -> float:
    """TF-IDF cosine similarity between resume and job description."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score) * 100, 2)


SKILL_KEYWORDS = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "r", "go",
    "rust", "swift", "kotlin", "php", "ruby", "scala", "matlab",
    # Web
    "html", "css", "react", "angular", "vue", "node", "django", "flask",
    "fastapi", "spring", "express", "nextjs",
    # Data / ML
    "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
    "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    "tableau", "powerbi", "sql", "mysql", "postgresql", "mongodb",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
    "ci/cd", "jenkins", "terraform", "ansible",
    # Soft skills
    "communication", "leadership", "teamwork", "problem solving",
    "time management", "agile", "scrum",
]


def extract_skills(text: str) -> list[str]:
    """Find known skill keywords present in the text."""
    text_lower = text.lower()
    return [skill for skill in SKILL_KEYWORDS if skill in text_lower]


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 AI Resume Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Instantly analyze how well a resume matches a job description</div>', unsafe_allow_html=True)
st.markdown('<div class="badge-center"><span class="badge">⚡ Powered by NLP &amp; TF-IDF</span></div>', unsafe_allow_html=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ─── UPLOAD SECTION ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="upload-label">📄 Upload Resume</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader(
        "Resume",
        type=["pdf", "txt"],
        key="resume",
        label_visibility="collapsed",
    )

with col2:
    st.markdown('<div class="upload-label">📋 Upload Job Description</div>', unsafe_allow_html=True)
    jd_file = st.file_uploader(
        "Job Description",
        type=["pdf", "txt"],
        key="jd",
        label_visibility="collapsed",
    )

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ─── ANALYZE BUTTON ───────────────────────────────────────────────────────────
both_uploaded = resume_file is not None and jd_file is not None

col_btn1, col_btn2, col_btn3 = st.columns([2, 3, 2])
with col_btn2:
    analyze = st.button(
        "🔍 Analyze Match" if both_uploaded else "⬆️ Upload both files to Analyze",
        disabled=not both_uploaded,
    )

# ─── RESULTS ──────────────────────────────────────────────────────────────────
if analyze and both_uploaded:
    with st.spinner("Extracting and analyzing text …"):
        progress = st.progress(0)
        for pct in range(0, 60, 10):
            time.sleep(0.05)
            progress.progress(pct / 100)

        resume_raw = extract_text(resume_file)
        jd_raw = extract_text(jd_file)

        for pct in range(60, 90, 10):
            time.sleep(0.05)
            progress.progress(pct / 100)

        resume_clean = preprocess(resume_raw)
        jd_clean = preprocess(jd_raw)
        score = compute_similarity(resume_clean, jd_clean)

        resume_skills = extract_skills(resume_raw)
        jd_skills = extract_skills(jd_raw)
        matched_skills = list(set(resume_skills) & set(jd_skills))
        missing_skills = list(set(jd_skills) - set(resume_skills))

        progress.progress(1.0)
        time.sleep(0.1)
        progress.empty()

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Score circle ──
    if score >= 70:
        circle_class = "score-high"
        label = "Excellent Match"
    elif score >= 40:
        circle_class = "score-mid"
        label = "Moderate Match"
    else:
        circle_class = "score-low"
        label = "Low Match"

    st.markdown(f"""
    <div class="score-circle {circle_class}">
        <div class="score-number">{score}%</div>
        <div class="score-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    st.progress(score / 100)

    # Verdict
    if score >= 70:
        st.success("🎉 **Excellent alignment!** Your resume strongly matches this job description. You're in a great position to apply — consider personalizing your cover letter to highlight your top matching skills.")
    elif score >= 40:
        st.warning("⚠️ **Moderate Match:** Your resume partially aligns with this role. Consider adding more relevant keywords and skills from the job description to strengthen your application.")
    else:
        st.error("❌ **Low Match:** Your resume doesn't closely match this job description. Significant tailoring is recommended — review the required skills and adjust your resume accordingly.")

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Skills breakdown ──
    r1, r2, r3 = st.columns(3, gap="medium")

    with r1:
        st.markdown('<div class="section-header">✅ Matched Skills</div>', unsafe_allow_html=True)
        if matched_skills:
            tags = "".join(f'<span class="skill-tag skill-tag-match">{s}</span>' for s in matched_skills)
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#64748b;font-size:0.85rem;">None detected</span>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="section-header">❌ Missing Skills</div>', unsafe_allow_html=True)
        if missing_skills:
            tags = "".join(f'<span class="skill-tag">{s}</span>' for s in missing_skills)
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#64748b;font-size:0.85rem;">None — great coverage!</span>', unsafe_allow_html=True)

    with r3:
        st.markdown('<div class="section-header">📄 Resume Skills</div>', unsafe_allow_html=True)
        if resume_skills:
            tags = "".join(f'<span class="skill-tag">{s}</span>' for s in resume_skills)
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#64748b;font-size:0.85rem;">None detected</span>', unsafe_allow_html=True)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Raw text preview ──
    with st.expander("📃 View Extracted Resume Text"):
        st.text_area("Resume Text", resume_raw[:3000] + ("…" if len(resume_raw) > 3000 else ""), height=200, disabled=True)

    with st.expander("📋 View Extracted Job Description Text"):
        st.text_area("JD Text", jd_raw[:3000] + ("…" if len(jd_raw) > 3000 else ""), height=200, disabled=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-container">
    <div class="footer-copy">
        <strong style="color:#a78bfa;">Prayag Rajyaguru</strong>
    </div>
</div>
""", unsafe_allow_html=True)
