import streamlit as st
import nltk
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import math

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
    page_title="Talentry — AI Resume Screener",
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

.stApp::before {
    content: '';
    position: fixed;
    top: -200px; left: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, #7c3aed22, transparent 70%);
    border-radius: 50%;
    animation: blobFloat 8s ease-in-out infinite;
    pointer-events: none; z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -150px; right: -150px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, #2563eb22, transparent 70%);
    border-radius: 50%;
    animation: blobFloat 10s ease-in-out infinite reverse;
    pointer-events: none; z-index: 0;
}
@keyframes blobFloat {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%       { transform: translate(30px, -40px) scale(1.05); }
    66%       { transform: translate(-20px, 20px) scale(0.96); }
}

/* ── Hero ── */
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
    margin-bottom: 1.2rem;
}
.badge-center { text-align: center; }

/* ── Step Indicator ── */
.step-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 0.8rem auto 2rem;
    max-width: 560px;
}
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    position: relative;
}
.step-circle {
    width: 40px; height: 40px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; font-weight: 700;
    transition: all 0.4s ease;
}
.step-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    white-space: nowrap;
    transition: color 0.4s ease;
}
.step-pending .step-circle {
    background: rgba(255,255,255,0.05);
    border: 2px solid rgba(255,255,255,0.15);
    color: #475569;
}
.step-pending .step-label { color: #475569; }

.step-active .step-circle {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    border: 2px solid #a78bfa;
    color: white;
    box-shadow: 0 0 20px #7c3aed66;
    animation: stepPulse 2s ease-in-out infinite;
}
.step-active .step-label { color: #c4b5fd; }

.step-done .step-circle {
    background: linear-gradient(135deg, #059669, #34d399);
    border: 2px solid #6ee7b7;
    color: white;
}
.step-done .step-label { color: #6ee7b7; }

@keyframes stepPulse {
    0%, 100% { box-shadow: 0 0 20px #7c3aed66; }
    50%       { box-shadow: 0 0 30px #7c3aed99; }
}
.step-line {
    width: 80px; height: 2px;
    background: rgba(255,255,255,0.1);
    margin: 0 4px;
    margin-bottom: 22px;
    border-radius: 999px;
    transition: background 0.4s ease;
}
.step-line-done {
    background: linear-gradient(90deg, #059669, #34d399);
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

/* ── Upload label ── */
.upload-label {
    color: #c4b5fd;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 8px;
    letter-spacing: 0.02em;
}

/* ── File Preview Card ── */
.file-preview-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 12px;
    padding: 10px 14px;
    margin-top: 10px;
    animation: fadeSlideIn 0.4s ease;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.file-preview-icon {
    font-size: 1.8rem;
    line-height: 1;
}
.file-preview-info { flex: 1; overflow: hidden; }
.file-preview-name {
    color: #e2e8f0;
    font-size: 0.85rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.file-preview-meta {
    color: #64748b;
    font-size: 0.75rem;
    margin-top: 2px;
}
.file-preview-check {
    font-size: 1.3rem;
}

/* ── Donut chart ── */
.donut-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 auto 1rem;
}
.donut-container svg {
    filter: drop-shadow(0 0 20px rgba(124,58,237,0.2));
}

/* ── Score stat chips ── */
.stat-chips-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 1.2rem;
}
.stat-chip {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 8px 18px;
    min-width: 90px;
}
.stat-chip-value {
    font-size: 1.3rem;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
    color: #c4b5fd;
    line-height: 1.2;
}
.stat-chip-label {
    font-size: 0.68rem;
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
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

/* ── Skill badges by category ── */
.skill-tag {
    display: inline-block;
    font-size: 0.78rem;
    padding: 3px 10px;
    border-radius: 999px;
    margin: 3px 3px;
    font-weight: 500;
    border: 1px solid;
}

/* ── Bar chart ── */
.bar-chart-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 1.2rem 0;
}
.bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 10px 0;
}
.bar-label {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 600;
    min-width: 140px;
    white-space: nowrap;
}
.bar-track {
    flex: 1;
    height: 10px;
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
    animation: barGrow 1.2s cubic-bezier(0.4,0,0.2,1) both;
}
@keyframes barGrow {
    from { width: 0% !important; }
}
.bar-pct {
    font-size: 0.82rem;
    font-weight: 700;
    min-width: 38px;
    text-align: right;
    font-family: 'Space Grotesk', sans-serif;
}
.bar-count {
    font-size: 0.72rem;
    color: #475569;
    min-width: 40px;
}

/* ── Recommendations card ── */
.rec-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(37,99,235,0.06));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin: 1.2rem 0;
}
.rec-title {
    color: #c4b5fd;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.rec-item {
    color: #cbd5e1;
    font-size: 0.85rem;
    line-height: 1.6;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.rec-item:last-child { border-bottom: none; }
.rec-item code {
    background: rgba(124,58,237,0.2);
    color: #c4b5fd;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.8rem;
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

/* ── Verdict banners ── */
.verdict-banner {
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
    font-size: 0.92rem;
    font-weight: 500;
    line-height: 1.6;
    animation: fadeSlideIn 0.5s ease;
}
.verdict-high {
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.35);
    color: #6ee7b7;
}
.verdict-mid {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.35);
    color: #fde68a;
}
.verdict-low {
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.35);
    color: #fca5a5;
}

/* ── Button ── */
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

/* ── Footer ── */
.footer-container {
    margin-top: 3rem;
    padding: 2.5rem 1rem 2rem;
    border-top: 1px solid rgba(124,58,237,0.25);
    background: linear-gradient(180deg, transparent, rgba(124,58,237,0.06));
    text-align: center;
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
.footer-copy {
    color: #334155;
    font-size: 0.74rem;
    margin-top: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ─── SKILL CATEGORIES ─────────────────────────────────────────────────────────
SKILL_CATEGORIES = {
    "💻 Programming": {
        "color": "#818cf8", "bg": "#1e1b4b", "border": "#4338ca",
        "skills": ["python", "java", "javascript", "typescript", "c", "c++", "c#",
                   "r", "go", "rust", "swift", "kotlin", "php", "ruby", "scala", "matlab"],
    },
    "🌐 Web & Frameworks": {
        "color": "#38bdf8", "bg": "#0c1a2e", "border": "#0284c7",
        "skills": ["html", "css", "react", "angular", "vue", "node", "django",
                   "flask", "fastapi", "spring", "express", "nextjs"],
    },
    "🤖 ML / AI": {
        "color": "#c084fc", "bg": "#2d1b4e", "border": "#9333ea",
        "skills": ["machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
                   "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn"],
    },
    "🗄️ Data & Databases": {
        "color": "#fb923c", "bg": "#2a1500", "border": "#ea580c",
        "skills": ["sql", "mysql", "postgresql", "mongodb", "tableau", "powerbi"],
    },
    "☁️ Cloud & DevOps": {
        "color": "#34d399", "bg": "#022c22", "border": "#059669",
        "skills": ["aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
                   "ci/cd", "jenkins", "terraform", "ansible"],
    },
    "🤝 Soft Skills": {
        "color": "#f472b6", "bg": "#2d0a1e", "border": "#ec4899",
        "skills": ["communication", "leadership", "teamwork", "problem solving",
                   "time management", "agile", "scrum"],
    },
}

ALL_SKILLS = [s for cat in SKILL_CATEGORIES.values() for s in cat["skills"]]


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip()


def extract_text(uploaded_file) -> str:
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def preprocess(text: str) -> str:
    tokens = word_tokenize(text.lower())
    filtered = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
    return " ".join(filtered)


def compute_similarity(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score) * 100, 2)


def extract_skills(text: str) -> list:
    text_lower = text.lower()
    return [s for s in ALL_SKILLS if s in text_lower]


def format_file_size(uploaded_file) -> str:
    try:
        size = len(uploaded_file.getvalue())
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size/1024:.1f} KB"
        return f"{size/(1024*1024):.1f} MB"
    except Exception:
        return "–"


def get_skill_style(skill: str) -> dict:
    for cat_data in SKILL_CATEGORIES.values():
        if skill in cat_data["skills"]:
            return cat_data
    return {"color": "#94a3b8", "bg": "#1e293b", "border": "#475569"}


def render_skill_badge(skill: str, override_color: str = None) -> str:
    style = get_skill_style(skill)
    color = override_color or style["color"]
    bg = style["bg"]
    border = style["border"]
    return (
        f'<span class="skill-tag" style="color:{color};background:{bg};'
        f'border-color:{border}55;">{skill}</span>'
    )


def score_donut_html(score: float, color: str, label: str) -> str:
    r = 80
    circ = 2 * math.pi * r
    offset = circ * (1 - score / 100)
    return f"""
    <div class="donut-container">
      <svg width="220" height="220" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="5" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <circle cx="100" cy="100" r="{r}" fill="none" stroke="#1e1b4b" stroke-width="18"/>
        <circle cx="100" cy="100" r="{r}" fill="none" stroke="{color}" stroke-width="18"
          stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"
          stroke-linecap="round" transform="rotate(-90 100 100)"
          filter="url(#glow)"
          style="transition: stroke-dashoffset 1.5s cubic-bezier(0.4,0,0.2,1);"/>
        <text x="100" y="90" text-anchor="middle" fill="{color}"
          font-size="38" font-weight="800" font-family="Space Grotesk, Inter, sans-serif">{score}%</text>
        <text x="100" y="118" text-anchor="middle" fill="#94a3b8"
          font-size="13" font-weight="500" font-family="Inter, sans-serif">Match Score</text>
        <text x="100" y="138" text-anchor="middle" fill="{color}"
          font-size="11" font-weight="600" font-family="Inter, sans-serif" opacity="0.8">{label}</text>
      </svg>
    </div>
    """


def skill_bar_chart_html(resume_skills: list, jd_skills: list) -> str:
    rows = ""
    for cat_name, cat_data in SKILL_CATEGORIES.items():
        jd_cat = [s for s in cat_data["skills"] if s in jd_skills]
        if not jd_cat:
            continue
        res_cat = [s for s in jd_cat if s in resume_skills]
        pct = round(len(res_cat) / len(jd_cat) * 100)
        color = cat_data["color"]
        rows += f"""
        <div class="bar-row">
          <div class="bar-label">{cat_name}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color}99,{color});box-shadow:0 0 8px {color}55;"></div>
          </div>
          <div class="bar-pct" style="color:{color};">{pct}%</div>
          <div class="bar-count">({len(res_cat)}/{len(jd_cat)})</div>
        </div>"""
    if not rows:
        return ""
    return f"""
    <div class="bar-chart-container">
      <div class="section-header">📊 Skills Coverage by Category</div>
      {rows}
    </div>"""


def recommendations_html(score: float, matched: list, missing: list, jd_skills: list) -> str:
    tips = []
    if score >= 70:
        tips.append("🌟 <strong>Excellent alignment!</strong> Your resume strongly matches this role. Personalise your cover letter to highlight your top matched skills.")
    elif score >= 40:
        tips.append(f"📈 <strong>Moderate match.</strong> Boosting to 70%+ is achievable — add the missing skills below to your resume.")
    else:
        tips.append(f"⚠️ <strong>Low match ({score}%).</strong> Consider a significant rewrite targeting the role's core requirements.")

    if missing:
        top = ", ".join([f"<code>{s}</code>" for s in missing[:5]])
        tips.append(f"🔧 <strong>Add these keywords</strong> to your resume: {top}.")

    if jd_skills:
        coverage = round(len(matched) / len(jd_skills) * 100)
        tips.append(
            f"✅ <strong>Skill coverage:</strong> You match {len(matched)}/{len(jd_skills)} "
            f"required skills ({coverage}%). {'Great coverage!' if coverage >= 75 else 'Room to improve.'}"
        )

    tips.append("📝 <strong>ATS tip:</strong> Mirror exact phrasing from the job description to pass automated screening filters.")

    items = "".join(f'<div class="rec-item">{t}</div>' for t in tips[:4])
    return f"""
    <div class="rec-card">
      <div class="rec-title">💡 Recommendations</div>
      {items}
    </div>"""


def step_indicator_html(step: int) -> str:
    steps = ["📁 Upload Files", "🔍 Analyze", "📊 Results"]
    parts = []
    for i, label in enumerate(steps, 1):
        if i < step:
            cls, icon = "step-done", "✓"
        elif i == step:
            cls, icon = "step-active", str(i)
        else:
            cls, icon = "step-pending", str(i)
        parts.append(f"""
        <div class="step-item {cls}">
          <div class="step-circle">{icon}</div>
          <div class="step-label">{label}</div>
        </div>""")
        if i < len(steps):
            line_cls = "step-line-done" if i < step else ""
            parts.append(f'<div class="step-line {line_cls}"></div>')
    return f'<div class="step-indicator">{"".join(parts)}</div>'


def file_preview_html(uploaded_file) -> str:
    name = uploaded_file.name
    size = format_file_size(uploaded_file)
    ext = name.split(".")[-1].upper()
    icon = "📄" if ext == "PDF" else "📝"
    return f"""
    <div class="file-preview-card">
      <div class="file-preview-icon">{icon}</div>
      <div class="file-preview-info">
        <div class="file-preview-name">{name}</div>
        <div class="file-preview-meta">{ext} &bull; {size}</div>
      </div>
      <div class="file-preview-check">✅</div>
    </div>"""


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 Talentry</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Instantly analyze how well a resume matches a job description</div>', unsafe_allow_html=True)
st.markdown('<div class="badge-center"><span class="badge">⚡ Powered by NLP &amp; TF-IDF</span></div>', unsafe_allow_html=True)

# ─── STATE ────────────────────────────────────────────────────────────────────
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ─── STEP INDICATOR ───────────────────────────────────────────────────────────
# Determine step after widgets are rendered (use placeholders)
step_placeholder = st.empty()

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ─── UPLOAD SECTION ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="upload-label">📄 Resume</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Resume", type=["pdf", "txt"], key="resume", label_visibility="collapsed")
    if resume_file:
        st.markdown(file_preview_html(resume_file), unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-label">📋 Job Description</div>', unsafe_allow_html=True)
    jd_file = st.file_uploader("Job Description", type=["pdf", "txt"], key="jd", label_visibility="collapsed")
    if jd_file:
        st.markdown(file_preview_html(jd_file), unsafe_allow_html=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ─── STEP INDICATOR (dynamic) ─────────────────────────────────────────────────
both_uploaded = resume_file is not None and jd_file is not None
if st.session_state.analyzed:
    current_step = 3
elif both_uploaded:
    current_step = 2
else:
    current_step = 1
step_placeholder.markdown(step_indicator_html(current_step), unsafe_allow_html=True)

# ─── ANALYZE BUTTON ───────────────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([2, 3, 2])
with col_btn2:
    analyze = st.button(
        "🔍 Analyze Match" if both_uploaded else "⬆️ Upload both files to Analyze",
        disabled=not both_uploaded,
    )

# ─── RESULTS ──────────────────────────────────────────────────────────────────
if analyze and both_uploaded:
    st.session_state.analyzed = True

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

        word_count = len(resume_raw.split())
        match_rank = "A+" if score >= 80 else "A" if score >= 70 else "B+" if score >= 60 else "B" if score >= 50 else "C+" if score >= 40 else "C"

        progress.progress(1.0)
        time.sleep(0.1)
        progress.empty()

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Score color ──
    if score >= 70:
        score_color, label = "#34d399", "Excellent Match"
        verdict_cls = "verdict-high"
        verdict_text = "🎉 <strong>Excellent alignment!</strong> Your resume strongly matches this job description. You're in a great position to apply — personalise your cover letter to highlight your top matching skills."
    elif score >= 40:
        score_color, label = "#fbbf24", "Moderate Match"
        verdict_cls = "verdict-mid"
        verdict_text = "⚠️ <strong>Moderate Match:</strong> Your resume partially aligns with this role. Consider adding more relevant keywords and skills from the job description to strengthen your application."
    else:
        score_color, label = "#f87171", "Low Match"
        verdict_cls = "verdict-low"
        verdict_text = "❌ <strong>Low Match:</strong> Your resume doesn't closely match this job description. Significant tailoring is recommended — review the required skills and adjust your resume accordingly."

    # ── Donut + stats layout ──
    dc1, dc2, dc3 = st.columns([1, 2, 1])
    with dc2:
        st.markdown(score_donut_html(score, score_color, label), unsafe_allow_html=True)

        # Stat chips
        st.markdown(f"""
        <div class="stat-chips-row">
          <div class="stat-chip">
            <div class="stat-chip-value">{word_count}</div>
            <div class="stat-chip-label">Words</div>
          </div>
          <div class="stat-chip">
            <div class="stat-chip-value">{len(resume_skills)}</div>
            <div class="stat-chip-label">Skills Found</div>
          </div>
          <div class="stat-chip">
            <div class="stat-chip-value">{len(matched_skills)}</div>
            <div class="stat-chip-label">Matched</div>
          </div>
          <div class="stat-chip">
            <div class="stat-chip-value" style="color:{score_color};">{match_rank}</div>
            <div class="stat-chip-label">Match Rank</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Verdict banner
        st.markdown(f'<div class="verdict-banner {verdict_cls}">{verdict_text}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Skills bar chart + Recommendations ──
    bar1, bar2 = st.columns([3, 2], gap="large")

    with bar1:
        bar_html = skill_bar_chart_html(resume_skills, jd_skills)
        if bar_html:
            st.markdown(bar_html, unsafe_allow_html=True)

    with bar2:
        st.markdown(recommendations_html(score, matched_skills, missing_skills, jd_skills), unsafe_allow_html=True)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Skills breakdown ──
    r1, r2, r3 = st.columns(3, gap="medium")

    with r1:
        st.markdown('<div class="section-header">✅ Matched Skills</div>', unsafe_allow_html=True)
        if matched_skills:
            tags = "".join(render_skill_badge(s, override_color="#34d399") for s in matched_skills)
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#64748b;font-size:0.85rem;">None detected</span>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="section-header">❌ Missing Skills</div>', unsafe_allow_html=True)
        if missing_skills:
            tags = "".join(render_skill_badge(s, override_color="#f87171") for s in missing_skills)
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#64748b;font-size:0.85rem;">None — great coverage!</span>', unsafe_allow_html=True)

    with r3:
        st.markdown('<div class="section-header">📄 Resume Skills</div>', unsafe_allow_html=True)
        if resume_skills:
            tags = "".join(render_skill_badge(s) for s in resume_skills)
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
  <div class="footer-logo">Talentry</div>
  <div class="footer-copy">Built by <strong style="color:#a78bfa;">Prayag Rajyaguru</strong></div>
</div>
""", unsafe_allow_html=True)
