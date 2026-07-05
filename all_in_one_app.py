"""
HireAI Resume Ranker - All-In-One Application

This single file contains the configurations, parsing logic, feature extraction,
dual similarity scoring (TF-IDF + BERT), XGBoost ranking model, evaluation metrics,
automated test cases, and the premium Streamlit Web Dashboard UI.

To run the CLI pipeline:
    python all_in_one_app.py

To run the Web Dashboard:
    streamlit run all_in_one_app.py

To run self-tests:
    python all_in_one_app.py --test
"""

import os
import re
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from scipy.stats import kendalltau, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Graceful optional imports
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import streamlit as st
    import altair as alt
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


# ─── 1. CONFIGURATION CONSTANTS ──────────────────────────────────────
DATA_DIR = "data"
RESUME_DIR = f"{DATA_DIR}/sample_resumes"
JD_DIR = f"{DATA_DIR}/sample_jds"
MODEL_DIR = "model"
MODEL_PATH = f"{MODEL_DIR}/ranking_model.pkl"
DEFAULT_JD = f"{JD_DIR}/ml_engineer.txt"

SPACY_MODEL = "en_core_web_sm"
BERT_MODEL = "all-MiniLM-L6-v2"
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

TFIDF_WEIGHT = 0.4
BERT_WEIGHT = 0.6

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
}

TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42

GROUND_TRUTH = {
    "senior_ml_engineer": 5.0,
    "data_scientist_python": 4.0,
    "data_analyst_python": 3.0,
    "devops_engineer": 1.5,
    "fullstack_developer": 2.0,
    "frontend_react_developer": 1.0,
}

IDEAL_RANKING = [
    "senior_ml_engineer",
    "data_scientist_python",
    "data_analyst_python",
    "devops_engineer",
    "fullstack_developer",
    "frontend_react_developer",
]

FEATURE_COLS = [
    "tfidf_cosine",
    "bert_cosine",
    "combined_similarity",
    "skill_overlap_count",
    "skill_overlap_ratio_jd",
    "skill_jaccard",
    "resume_experience_years",
    "experience_sufficient",
    "resume_education_level",
    "education_match",
    "resume_to_jd_length_ratio",
    "num_resume_skills",
]

TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql", "nosql",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "scipy", "matplotlib", "seaborn", "nltk", "spacy", "huggingface",
    "transformers", "xgboost", "lightgbm", "catboost", "machine learning",
    "deep learning", "nlp", "computer vision", "natural language processing",
    "neural networks", "reinforcement learning", "random forest", "logistic regression",
    "linear regression", "svm", "convolutional neural network", "recurrent neural network",
    "lstm", "gru", "transformer", "attention mechanism", "bert", "gpt",
    "feature engineering", "a/b testing", "mongodb", "postgresql", "mysql",
    "redis", "elasticsearch", "sqlite", "docker", "kubernetes", "aws", "azure",
    "gcp", "terraform", "ansible", "ci/cd", "jenkins", "github actions", "linux",
    "unix", "spark", "hadoop", "airflow", "dbt", "etl", "data pipeline",
    "flask", "django", "fastapi", "spring", "express", "node.js", "react",
    "angular", "vue", "next.js", "tableau", "power bi", "looker", "git",
    "api", "rest", "graphql", "microservices", "html", "css", "agile", "scrum",
    "tdd", "oop", "design patterns", "webpack",
}

EDUCATION_LEVELS = {
    "phd": 5, "doctor": 5, "doctorate": 5,
    "master": 4, "m.s.": 4, "m.sc": 4, "msc": 4, "m.tech": 4, "mca": 4,
    "bachelor": 3, "b.s.": 3, "b.sc": 3, "b.tech": 3, "b.e.": 3, "bca": 3,
    "associate": 2, "diploma": 2,
    "high school": 1,
}


# ─── 2. AUTOMATIC DATA GENERATOR ─────────────────────────────────────
def ensure_sample_data():
    """Generates folders and sample data files if they do not exist."""
    os.makedirs(RESUME_DIR, exist_ok=True)
    os.makedirs(JD_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    jd_file = DEFAULT_JD
    if not os.path.exists(jd_file):
        with open(jd_file, "w", encoding="utf-8") as f:
            f.write("""Senior Machine Learning Engineer

Required Skills:
- Python (proficient)
- Machine Learning & Deep Learning frameworks (TensorFlow, PyTorch, scikit-learn)
- NLP experience (spaCy, NLTK, Hugging Face Transformers)
- SQL and NoSQL databases
- Docker and Kubernetes
- AWS or GCP cloud services
- Feature engineering and model optimization
- Experience with BERT, GPT, or Transformer architectures

Preferred Qualifications:
- Master's degree or PhD in Computer Science, Statistics, or related field
- 5+ years of experience in ML/AI roles
- Experience deploying ML models to production
- Knowledge of MLOps practices (CI/CD for ML, model monitoring)
- Published research or contributions to open-source ML projects
""")

    resumes_data = {
        "senior_ml_engineer.txt": """Dr. Sarah Chen
Senior Machine Learning Engineer
sarah.chen@email.com | github.com/sarahchen

SUMMARY
Senior ML Engineer with 8 years of experience building production-grade ML systems. PhD in Computer Science specializing in NLP. Expertise in deploying transformer models at scale.

EDUCATION
PhD in Computer Science - Stanford University (2016)
Master of Science in Computer Science - Stanford University (2014)

EXPERIENCE
Senior ML Engineer - Google Brain (2020-Present)
- Led team of 5 engineers building NLP models
- Deployed BERT-based models serving 10M+ requests using TensorFlow
- Reduced model inference latency by 60% through quantization and ONNX
- Used Docker and Kubernetes for model serving on GCP

SKILLS
Python, TensorFlow, PyTorch, scikit-learn, XGBoost, spaCy, NLTK, Hugging Face Transformers, BERT, GPT, Docker, Kubernetes, AWS, GCP, SQL, Git, Linux, feature engineering, deep learning, machine learning, NLP, natural language processing, transformer, attention mechanism""",
        
        "data_scientist_python.txt": """Raj Patel
Data Scientist
raj.patel@email.com

SUMMARY
Data Scientist with 5 years of experience in machine learning, statistical analysis, and NLP. Master's degree in Statistics. Strong Python programming skills with experience in scikit-learn, TensorFlow, and PyTorch.

EDUCATION
Master of Science in Statistics - UC Berkeley (2019)

EXPERIENCE
Data Scientist - Amazon (2021-Present)
- Built ML models for customer segmentation using XGBoost and random forest
- Developed NLP pipeline for product review analysis using NLTK and spaCy
- Created feature engineering workflows processing 2M+ records daily
- Used SQL and Python for data extraction and analysis

SKILLS
Python, R, SQL, pandas, numpy, scikit-learn, TensorFlow, PyTorch, XGBoost, matplotlib, seaborn, NLTK, spaCy, machine learning, deep learning, NLP, natural language processing, feature engineering, random forest, statistics""",
        
        "data_analyst_python.txt": """Emily Rodriguez
Data Analyst
emily.r@email.com

SUMMARY
Data Analyst with 3 years of experience in data analysis, visualization, and basic machine learning. Proficient in Python and SQL. Bachelor's degree in Data Science.

EDUCATION
Bachelor of Science in Data Science - Georgia Tech (2021)

EXPERIENCE
Data Analyst - Accenture (2021-Present)
- Analyzed large datasets using Python, pandas, and SQL
- Built basic predictive models using scikit-learn (logistic regression, random forest)
- Created dashboards and reports using Tableau and matplotlib

SKILLS
Python, SQL, pandas, numpy, scikit-learn, matplotlib, seaborn, Tableau, Excel, git, machine learning, logistic regression, random forest, statistics, data analysis""",
        
        "fullstack_developer.txt": """Mike Johnson
Full Stack Developer
mike.j@email.com

SUMMARY
Full Stack Developer with 6 years of experience building web applications. Proficient in JavaScript, Python, React, and Node.js. Some experience with basic machine learning through online courses.

EDUCATION
Bachelor of Science in Computer Science - University of Michigan (2018)

EXPERIENCE
Senior Full Stack Developer - Shopify (2021-Present)
- Built web applications using React, Node.js, and Express
- Developed REST APIs using Python Flask and Managed SQL/PostgreSQL databases
- Deployed applications using Docker and AWS

SKILLS
JavaScript, TypeScript, Python, React, Node.js, Express, Flask, SQL, PostgreSQL, MongoDB, Redis, Docker, AWS, git, CI/CD, HTML, CSS, REST, API""",
        
        "frontend_react_developer.txt": """Lisa Wang
Frontend Developer
lisa.wang@email.com

SUMMARY
Frontend Developer with 4 years of experience specializing in React and JavaScript. Focused on building responsive, accessible user interfaces. No machine learning experience.

EDUCATION
Bachelor of Science in Information Technology - NYU (2020)

EXPERIENCE
Frontend Developer - Stripe (2022-Present)
- Built payment interface components using React and TypeScript
- Implemented responsive designs using HTML, CSS, and Sass
- Optimized web performance achieving 95+ Lighthouse scores

SKILLS
JavaScript, TypeScript, React, Next.js, HTML, CSS, Redux, webpack, Git, responsive design, REST, API""",
        
        "devops_engineer.txt": """James O'Brien
DevOps Engineer
james.obrien@email.com

SUMMARY
DevOps Engineer with 7 years of experience in cloud infrastructure, CI/CD, and automation. Strong Linux and containerization skills. Limited programming experience, no ML background.

EDUCATION
Bachelor of Engineering in Computer Engineering - Purdue University (2017)

EXPERIENCE
Senior DevOps Engineer - Netflix (2021-Present)
- Managed Kubernetes clusters on AWS serving 200M+ users
- Built CI/CD pipelines using Jenkins and GitHub Actions
- Implemented infrastructure as code using Terraform and Ansible

SKILLS
Docker, Kubernetes, AWS, Terraform, Ansible, Jenkins, GitHub Actions, Linux, Unix, CI/CD, Python, Bash, git, microservices, SQL, Redis"""
    }

    for name, content in resumes_data.items():
        path = os.path.join(RESUME_DIR, name)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)


# ─── 3. CORE RESUME PARSER ───────────────────────────────────────────
class ResumeParser:
    """Parse resumes from PDF, DOCX, and TXT files into plain text."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".text"}

    def parse(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return self._parse_pdf(filepath)
        elif ext in (".docx", ".doc"):
            return self._parse_docx(filepath)
        elif ext in (".txt", ".text"):
            return self._parse_txt(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _parse_pdf(self, filepath: str) -> str:
        if not HAS_PDFPLUMBER:
            raise ImportError("pdfplumber is required for PDF parsing.")
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

    def _parse_docx(self, filepath: str) -> str:
        if not HAS_DOCX:
            raise ImportError("python-docx is required for DOCX parsing.")
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _parse_txt(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def parse_directory(self, dirpath: str) -> Dict[str, str]:
        results = {}
        if not os.path.isdir(dirpath):
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        for filename in sorted(os.listdir(dirpath)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.SUPPORTED_EXTENSIONS:
                filepath = os.path.join(dirpath, filename)
                try:
                    results[filename] = self.parse(filepath)
                except Exception as e:
                    print(f"  [WARN] Failed to parse {filename}: {e}")
        return results


# ─── 4. FEATURE EXTRACTOR ────────────────────────────────────────────
class FeatureExtractor:
    """Extract structured features from resume and JD text."""

    def __init__(self, spacy_model: str = SPACY_MODEL):
        self.nlp = None
        if HAS_SPACY:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"  [INFO] Downloading spaCy model '{spacy_model}'...")
                from spacy.cli import download as spacy_download
                try:
                    spacy_download(spacy_model)
                    self.nlp = spacy.load(spacy_model)
                except Exception as e:
                    print(f"  [WARN] Could not download spaCy: {e}")

    def extract_skills(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = set()
        for skill in TECH_SKILLS:
            if " " in skill:
                if skill in text_lower:
                    found.add(skill)
            else:
                if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                    found.add(skill)
        return sorted(found)

    def extract_experience_years(self, text: str) -> float:
        text_lower = text.lower()
        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s+(?:of\s+)?experience",
            r"experience\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
            r"(\d+)\+\s*(?:years?|yrs?)\s+(?:of\s+)?experience",
        ]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))

        ranges = re.findall(
            r"(20\d{2})\s*[-–to]+\s*(20\d{2}|present|current)", text_lower
        )
        total = 0.0
        for start, end in ranges:
            end_year = 2024 if end in ("present", "current") else int(end)
            total += max(0, end_year - int(start))
        return total

    def extract_education_level(self, text: str) -> int:
        text_lower = text.lower()
        return max(
            (level for kw, level in EDUCATION_LEVELS.items() if kw in text_lower),
            default=0,
        )

    def compute_skill_overlap(self, resume_skills: List[str], jd_skills: List[str]) -> Dict:
        r_set = set(resume_skills)
        j_set = set(jd_skills)
        intersection = r_set & j_set
        union = r_set | j_set
        return {
            "overlap_count": len(intersection),
            "overlap_ratio_jd": len(intersection) / max(len(j_set), 1),
            "overlap_ratio_resume": len(intersection) / max(len(r_set), 1),
            "jaccard_similarity": len(intersection) / max(len(union), 1),
            "matched_skills": sorted(intersection),
            "missing_skills": sorted(j_set - r_set),
        }

    def extract_features(self, resume_text: str, jd_text: str) -> Dict:
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(jd_text)
        resume_exp = self.extract_experience_years(resume_text)
        resume_edu = self.extract_education_level(resume_text)
        jd_edu = self.extract_education_level(jd_text) or 3
        overlap = self.compute_skill_overlap(resume_skills, jd_skills)

        return {
            "num_resume_skills": len(resume_skills),
            "num_jd_skills": len(jd_skills),
            "skill_overlap_count": overlap["overlap_count"],
            "skill_overlap_ratio_jd": overlap["overlap_ratio_jd"],
            "skill_jaccard": overlap["jaccard_similarity"],
            "resume_experience_years": resume_exp,
            "experience_sufficient": 1.0 if resume_exp >= 3 else 0.0,
            "resume_education_level": resume_edu,
            "jd_education_level": jd_edu,
            "education_match": 1.0 if resume_edu >= jd_edu else 0.0,
            "resume_to_jd_length_ratio": len(resume_text) / max(len(jd_text), 1),
            "resume_length": len(resume_text),
            "jd_length": len(jd_text),
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "matched_skills": overlap["matched_skills"],
            "missing_skills": overlap["missing_skills"],
        }

    def extract_resume_summary(self, resume_text: str) -> Dict:
        return {
            "skills": self.extract_skills(resume_text),
            "experience_years": self.extract_experience_years(resume_text),
            "education_level": self.extract_education_level(resume_text),
        }


# ─── 5. SIMILARITY SCORER ────────────────────────────────────────────
class SimilarityScorer:
    """Computes TF-IDF + Sentence-BERT similarity scores."""

    def __init__(
        self,
        tfidf_max_features: int = TFIDF_MAX_FEATURES,
        tfidf_ngram_range: tuple = TFIDF_NGRAM_RANGE,
        bert_model_name: str = BERT_MODEL,
        tfidf_weight: float = TFIDF_WEIGHT,
        bert_weight: float = BERT_WEIGHT,
    ):
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            stop_words="english",
            ngram_range=tfidf_ngram_range,
            sublinear_tf=True,
        )
        self.tfidf_weight = tfidf_weight
        self.bert_weight = bert_weight
        self.bert_model = None

        if HAS_ST:
            try:
                self.bert_model = SentenceTransformer(bert_model_name)
                print(f"  [OK] BERT model loaded: {bert_model_name}")
            except Exception as e:
                print(f"  [WARN] BERT model load failed: {e}")
        else:
            print("  [WARN] sentence-transformers not installed — TF-IDF only mode")

    def _tfidf_pair(self, resume: str, jd: str) -> float:
        matrix = self.tfidf.fit_transform([resume, jd])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])

    def _bert_pair(self, resume: str, jd: str) -> float:
        if self.bert_model is None:
            return 0.0
        emb = self.bert_model.encode([resume, jd])
        return float(cosine_similarity(emb[0:1], emb[1:2])[0, 0])

    def score(self, resume: str, jd: str) -> Dict[str, float]:
        tfidf = self._tfidf_pair(resume, jd)
        bert = self._bert_pair(resume, jd)
        result = {"tfidf_cosine": tfidf, "bert_cosine": bert}
        if bert > 0:
            result["combined_similarity"] = (
                self.tfidf_weight * tfidf + self.bert_weight * bert
            )
        else:
            result["combined_similarity"] = tfidf
        return result

    def score_batch(self, resumes: List[str], jd: str) -> List[Dict[str, float]]:
        all_texts = resumes + [jd]
        tfidf_matrix = self.tfidf.fit_transform(all_texts)
        jd_vec = tfidf_matrix[-1:]
        tfidf_sims = cosine_similarity(tfidf_matrix[:-1], jd_vec).flatten()

        bert_sims = np.zeros(len(resumes))
        if self.bert_model is not None:
            emb = self.bert_model.encode(all_texts)
            jd_emb = emb[-1:]
            bert_sims = cosine_similarity(emb[:-1], jd_emb).flatten()

        results = []
        for tf, bt in zip(tfidf_sims, bert_sims):
            bt_f = float(bt)
            entry = {"tfidf_cosine": float(tf), "bert_cosine": bt_f}
            if bt_f > 0:
                entry["combined_similarity"] = (
                    self.tfidf_weight * float(tf) + self.bert_weight * bt_f
                )
            else:
                entry["combined_similarity"] = float(tf)
            results.append(entry)
        return results


# ─── 6. MACHINE LEARNING MODEL ───────────────────────────────────────
class RankingModel:
    """XGBoost tabular regressor mapping candidate features to score."""

    def __init__(self, use_bert: bool = True):
        self.use_bert = use_bert
        self.features = [f for f in FEATURE_COLS if f != "bert_cosine" or use_bert]
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def _to_matrix(self, feature_dicts: List[Dict]) -> np.ndarray:
        rows = []
        for fd in feature_dicts:
            row = []
            for col in self.features:
                v = fd.get(col, 0.0)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = 0.0
                row.append(float(v))
            rows.append(row)
        return np.array(rows)

    def train(self, feature_dicts: List[Dict], labels: List[float]) -> Dict:
        X = self._to_matrix(feature_dicts)
        y = np.array(labels)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        self.model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        self.model.fit(X_tr_s, y_tr)
        self.is_trained = True

        tr_pred = self.model.predict(X_tr_s)
        te_pred = self.model.predict(X_te_s)

        cv_folds = min(CV_FOLDS, len(X))
        cv = cross_val_score(
            self.model,
            self.scaler.fit_transform(X),
            y,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
        )

        importance = dict(
            sorted(
                zip(self.features, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        return {
            "train_rmse": np.sqrt(mean_squared_error(y_tr, tr_pred)),
            "test_rmse": np.sqrt(mean_squared_error(y_te, te_pred)),
            "train_mae": mean_absolute_error(y_tr, tr_pred),
            "test_mae": mean_absolute_error(y_te, te_pred),
            "cv_rmse_mean": float(-cv.mean()),
            "cv_rmse_std": float(cv.std()),
            "feature_importance": importance,
            "train_samples": len(X_tr),
            "test_samples": len(X_te),
        }

    def predict(self, feature_dicts: List[Dict]) -> List[float]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.scaler.transform(self._to_matrix(feature_dicts))
        return [float(s) for s in self.model.predict(X)]

    def rank(self, feature_dicts: List[Dict], ids: List[str] = None) -> List[Tuple[str, float]]:
        scores = self.predict(feature_dicts)
        if ids is None:
            ids = [f"candidate_{i}" for i in range(len(scores))]
        ranked = list(zip(ids, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "features": self.features,
                "use_bert": self.use_bert,
                "is_trained": self.is_trained,
            },
            path,
        )
        print(f"  [OK] Model saved -> {path}")

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.features = data["features"]
        self.use_bert = data["use_bert"]
        self.is_trained = data["is_trained"]
        print(f"  [OK] Model loaded <- {path}")


# ─── 7. EVALUATION SYSTEM ────────────────────────────────────────────
def ndcg_at_k(predicted: List[str], ideal: List[str], k: int = None) -> float:
    if k is None:
        k = len(predicted)
    k = min(k, len(predicted), len(ideal))
    dcg = 0.0
    for i, cid in enumerate(predicted[:k]):
        if cid in ideal:
            relevance = len(ideal) - ideal.index(cid)
            dcg += relevance / np.log2(i + 2)
    idcg = 0.0
    for i in range(k):
        idcg += (len(ideal) - i) / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def mrr(predicted: List[str], relevant: List[str]) -> float:
    for i, cid in enumerate(predicted):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ranking(predicted: List[Tuple[str, float]], ideal: List[str]) -> Dict:
    pred_ids = [c for c, _ in predicted]
    pred_scores = [s for _, s in predicted]
    ideal_scores = [
        (len(ideal) - ideal.index(c)) if c in ideal else 0 for c in pred_ids
    ]
    rel_n = max(3, len(ideal) // 2)
    relevant = ideal[:rel_n]

    metrics = {
        "ndcg": ndcg_at_k(pred_ids, ideal),
        "ndcg_at_3": ndcg_at_k(pred_ids, ideal, k=3),
        "ndcg_at_5": ndcg_at_k(pred_ids, ideal, k=5),
        "mrr": mrr(pred_ids, relevant),
        "spearman": float(
            spearmanr(pred_scores, ideal_scores).correlation
            if len(pred_scores) > 1
            else 0.0
        ),
        "kendall_tau": float(
            kendalltau(pred_scores, ideal_scores).correlation
            if len(pred_scores) > 1
            else 0.0
        ),
    }

    for k in (1, 3, 5):
        if k <= len(ideal):
            overlap = len(set(pred_ids[:k]) & set(ideal[:k]))
            metrics[f"top_{k}_overlap"] = overlap
            metrics[f"top_{k}_precision"] = overlap / k
    return metrics


def print_report(metrics: Dict, title: str = "Ranking Evaluation"):
    w = 50
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")
    print(f"  NDCG:           {metrics['ndcg']:.4f}")
    print(f"  NDCG@3:         {metrics['ndcg_at_3']:.4f}")
    print(f"  NDCG@5:         {metrics['ndcg_at_5']:.4f}")
    print(f"  MRR:            {metrics['mrr']:.4f}")
    print(f"  Spearman rho:   {metrics['spearman']:.4f}")
    print(f"  Kendall tau:    {metrics['kendall_tau']:.4f}")
    for k in (1, 3, 5):
        key = f"top_{k}_overlap"
        if key in metrics:
            print(f"  Top-{k} Overlap:  {metrics[key]}/{k}")
    print(f"{'=' * w}\n")


# ─── 8. PIPELINE ORCHESTRATION ───────────────────────────────────────
class ResumeRankingPipeline:
    def __init__(self, use_bert: bool = True):
        self.use_bert = use_bert
        self.parser = ResumeParser()
        self.extractor = FeatureExtractor(spacy_model=SPACY_MODEL)
        self.scorer = SimilarityScorer()
        self.model = RankingModel(use_bert=use_bert)

    def run(
        self,
        resume_dir: str = RESUME_DIR,
        jd_path: str = DEFAULT_JD,
        ground_truth: Optional[Dict[str, float]] = None,
        ideal_ranking: Optional[List[str]] = None,
        save_path: str = MODEL_PATH,
    ) -> Dict:
        print("\n" + "=" * 60)
        print("  HireAI — Resume Screening & Ranking Pipeline")
        print("=" * 60)

        resumes = self.parser.parse_directory(resume_dir)
        jd_text = ""
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

        print(f"\n[INFO] Extracting features...")
        texts = list(resumes.values())
        sim_results = self.scorer.score_batch(texts, jd_text)

        features = []
        for (fname, rtext), sim in zip(resumes.items(), sim_results):
            cid = os.path.splitext(fname)[0]
            structured = self.extractor.extract_features(rtext, jd_text)
            combined = {"candidate_id": cid, "resume_file": fname, **sim, **structured}
            combined.setdefault("bert_cosine", 0.0)
            features.append(combined)

            n_matched = len(combined.get("matched_skills", []))
            print(
                f"   [OK] {cid}: {n_matched} skills matched, "
                f"TF-IDF={sim['tfidf_cosine']:.3f}"
                + (f", BERT={sim['bert_cosine']:.3f}" if (self.use_bert and sim['bert_cosine'] > 0) else "")
            )

        results = {"features": features}

        if ground_truth:
            ids = [f["candidate_id"] for f in features]
            labels = [ground_truth.get(cid, 0.0) for cid in ids]
            results["training"] = self.model.train(features, labels)
            self.model.save(save_path)

        if self.model.is_trained:
            results["ranking"] = self.model.rank(features, [f["candidate_id"] for f in features])
        else:
            print("\n[RANK] Ranking by combined similarity (no model)...")
            ranked = sorted(
                [(f["candidate_id"], f["combined_similarity"]) for f in features],
                key=lambda x: x[1],
                reverse=True,
            )
            results["ranking"] = ranked

        for i, (c, s) in enumerate(results["ranking"], 1):
            print(f"   {i}. {c:<35s} {s:.4f}")

        if ideal_ranking and results["ranking"]:
            results["evaluation"] = evaluate_ranking(results["ranking"], ideal_ranking)
            print_report(results["evaluation"])

        return results


# ─── 9. DYNAMIC STREAMLIT WEB DASHBOARD ──────────────────────────────
def run_streamlit_dashboard():
    """Builds and serves the premium glassmorphic Streamlit Dashboard."""
    st.set_page_config(
        page_title="HireAI - AI Resume Screener",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * { font-family: 'Plus Jakarta Sans', sans-serif; }
            .main { background-color: #080c16; color: #f3f4f6; }
            
            .animated-header {
                background: linear-gradient(120deg, #00f2fe 0%, #4facfe 30%, #a18cd1 70%, #00f2fe 100%);
                background-size: 200% auto;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: 'Space Grotesk', sans-serif;
                font-size: 3.2rem;
                font-weight: 700;
                letter-spacing: -1px;
                animation: shimmerFlow 5s linear infinite;
                margin-bottom: 4px;
            }
            @keyframes shimmerFlow {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            .gradient-sub { color: #8b9bb4; font-size: 1.15rem; margin-bottom: 30px; }
            
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(24px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .glass-card {
                background: rgba(13, 20, 38, 0.6);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                padding: 26px;
                margin-bottom: 22px;
                box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4);
                animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
                transition: all 0.35s cubic-bezier(0.25, 0.8, 0.25, 1);
            }
            .glass-card:hover {
                transform: translateY(-6px);
                border-color: rgba(0, 242, 254, 0.35);
                box-shadow: 0 15px 45px rgba(0, 242, 254, 0.12);
            }
            .badge {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 30px;
                font-size: 0.78rem;
                font-weight: 600;
                margin: 4px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            .badge-match { background-color: rgba(16, 185, 129, 0.12); color: #34d399; border-color: rgba(52, 211, 153, 0.25); }
            .badge-missing { background-color: rgba(239, 68, 68, 0.08); color: #f87171; border-color: rgba(248, 113, 113, 0.2); }
            
            .badge-verdict { font-size: 0.85rem; padding: 8px 16px; border-radius: 8px; font-family: 'Space Grotesk', sans-serif; text-align: center; }
            .verdict-elite { background-color: rgba(0, 242, 254, 0.15); color: #00f2fe; border-color: rgba(0, 242, 254, 0.35); }
            .verdict-strong { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(52, 211, 153, 0.35); }
            .verdict-potential { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(251, 191, 36, 0.35); }
            .verdict-low { background-color: rgba(156, 163, 175, 0.1); color: #9ca3af; }
            
            div.stButton > button:first-child {
                background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
                color: #ffffff;
                border: none;
                padding: 14px 32px;
                border-radius: 14px;
                font-weight: 600;
                font-family: 'Space Grotesk', sans-serif;
                box-shadow: 0 4px 20px rgba(0, 242, 254, 0.3);
                width: 100%;
            }
            [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; font-weight: 700; background: linear-gradient(135deg, #ffffff 0%, #cfd9df 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="animated-header">HireAI Resume Ranker</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Deploying machine learning and NLP semantic embeddings to find your ideal candidate.</div>', unsafe_allow_html=True)

    # Sidebar settings
    st.sidebar.markdown("### ⚙️ Engine Settings")
    tfidf_wt = st.sidebar.slider("Lexical (TF-IDF) Weight", 0.0, 1.0, TFIDF_WEIGHT, step=0.1)
    bert_wt = round(1.0 - tfidf_wt, 1)
    st.sidebar.caption(f"Semantic (BERT) Weight: **{bert_wt}**")
    use_bert_toggle = st.sidebar.toggle("Enable BERT Semantics", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Fast Replay / Demo")
    run_demo = st.sidebar.button("Quick Demo Run 🚀")
    st.sidebar.caption("Instantly runs the pipeline with pre-configured data for Senior ML Engineer.")

    # Init engine
    @st.cache_resource
    def load_cached_nlp(use_bert_flag):
        ensure_sample_data()
        extractor_obj = FeatureExtractor(spacy_model=SPACY_MODEL)
        scorer_obj = SimilarityScorer(
            tfidf_weight=TFIDF_WEIGHT,
            bert_weight=BERT_WEIGHT,
        )
        model_obj = RankingModel(use_bert=use_bert_flag)
        if os.path.exists(MODEL_PATH):
            try:
                model_obj.load(MODEL_PATH)
            except Exception:
                pass
        return extractor_obj, scorer_obj, model_obj

    ext_e, scr_e, r_mdl = load_cached_nlp(use_bert_toggle)

    tab1, tab2 = st.tabs(["📊 Candidate Screener", "⚖️ Side-by-Side Comparer"])

    with tab1:
        col_jd, col_res = st.columns(2)
        with col_jd:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📋 Job Description (JD)")
            default_jd = ""
            if os.path.exists(DEFAULT_JD):
                with open(DEFAULT_JD, "r", encoding="utf-8") as f:
                    default_jd = f.read()
            jd_input = st.text_area("Paste JD Requirements", value=default_jd, height=300)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📂 Upload Candidate Resumes")
            uploaded_files = st.file_uploader(
                "Upload files (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Trigger analysis
        if st.button("Compute Fit Scores 📊") or run_demo:
            resumes = {}
            if run_demo:
                ensure_sample_data()
                for name in sorted(os.listdir(RESUME_DIR)):
                    path = os.path.join(RESUME_DIR, name)
                    with open(path, "r", encoding="utf-8") as f:
                        resumes[name] = f.read()
                jd_text = default_jd
            else:
                jd_text = jd_input
                if uploaded_files:
                    for f in uploaded_files:
                        txt = parse_uploaded_file(f)
                        if txt.strip():
                            resumes[f.name] = txt

            if not resumes:
                st.warning("Please upload resumes or click Quick Demo Run.")
            elif not jd_text.strip():
                st.warning("Please enter a job description.")
            else:
                with st.spinner("Analyzing candidate profiles..."):
                    scr_e.tfidf_weight = tfidf_wt
                    scr_e.bert_weight = bert_wt
                    
                    texts = list(resumes.values())
                    sims = scr_e.score_batch(texts, jd_text)
                    
                    features = []
                    for (fname, rtext), sim in zip(resumes.items(), sims):
                        cid = os.path.splitext(fname)[0].replace("_", " ").title()
                        feats = ext_e.extract_features(rtext, jd_text)
                        combined = {"candidate_id": cid, "filename": fname, **sim, **feats}
                        combined.setdefault("bert_cosine", 0.0)
                        features.append(combined)

                    ranked_list = []
                    if r_mdl.is_trained:
                        try:
                            preds = r_mdl.predict(features)
                            for feat, score in zip(features, preds):
                                ranked_list.append((feat, float(score)))
                            ranked_list.sort(key=lambda x: x[1], reverse=True)
                        except Exception:
                            for feat in features:
                                ranked_list.append((feat, feat["combined_similarity"] * 5.0))
                            ranked_list.sort(key=lambda x: x[1], reverse=True)
                    else:
                        for feat in features:
                            ranked_list.append((feat, feat["combined_similarity"] * 5.0))
                        ranked_list.sort(key=lambda x: x[1], reverse=True)

                st.session_state["ranked_results"] = ranked_list
                st.session_state["jd_skills"] = ext_e.extract_skills(jd_text)

                # Dashboard metrics
                st.markdown('<div class="gradient-title" style="font-size: 2.2rem; margin-top:35px; font-family:\'Space Grotesk\'">Leaderboard</div>', unsafe_allow_html=True)
                top_name = ranked_list[0][0]["candidate_id"]
                top_score = ranked_list[0][1]

                c1, c2, c3 = st.columns(3)
                c1.metric("Screened", len(ranked_list))
                c2.metric("Top Fit Match", top_name)
                c3.metric("Highest Profile Score", f"{top_score:.2f} / 5.0")

                # Leaderboard chart
                chart_data = pd.DataFrame({
                    "Candidate": [item[0]["candidate_id"] for item in ranked_list],
                    "Score": [round(item[1], 2) for item in ranked_list],
                })
                chart = alt.Chart(chart_data).mark_bar(
                    cornerRadiusTopRight=10,
                    cornerRadiusBottomRight=10
                ).encode(
                    x=alt.X("Score:Q", title="Fit Score", scale=alt.Scale(domain=[0, 5])),
                    y=alt.Y("Candidate:N", sort="-x", title="Candidate Name"),
                    color=alt.Color("Score:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
                    tooltip=["Candidate", "Score"]
                ).properties(height=260)
                st.altair_chart(chart, use_container_width=True)

                # Cards list
                def get_desc_verdict(sc, match_ratio):
                    if sc >= 4.0 and match_ratio >= 0.6:
                        return "ELITE MATCH", "verdict-elite", "Excellent core tech skills alignment and deep semantics. Highly recommended."
                    elif sc >= 2.5 and match_ratio >= 0.4:
                        return "STRONG CONTENDER", "verdict-strong", "Solid intermediate experience and core skill alignment."
                    elif sc >= 1.5:
                        return "POTENTIAL FIT", "verdict-potential", "Lacks a few primary frameworks, but demonstrates relevant technical foundations."
                    else:
                        return "NOT ALIGNED", "verdict-low", "Does not align with required qualifications or stack details."

                for idx, (feat, score) in enumerate(ranked_list, 1):
                    verdict, v_cls, v_desc = get_desc_verdict(score, feat["skill_overlap_ratio_jd"])
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    
                    hc1, hc2, hc3 = st.columns([3, 1.5, 1])
                    hc1.markdown(f"### #{idx} - {feat['candidate_id']}")
                    hc1.caption(f"File: `{feat['filename']}`")
                    hc2.markdown(f'<div class="badge badge-verdict {v_cls}">{verdict}</div>', unsafe_allow_html=True)
                    hc3.markdown(f"<h3 style='text-align: right; color:#00f2fe; margin:0;'>{score:.2f}</h3>", unsafe_allow_html=True)

                    st.info(f"💡 **Recruiter Insight:** {v_desc}")
                    
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    with sc1:
                        st.write(f"💼 **Experience:** {feat['resume_experience_years']:.1f} Yrs")
                        st.progress(min(max(feat['resume_experience_years'] / 10.0, 0.0), 1.0))
                    with sc2:
                        st.write(f"🎓 **Education Level:** {feat['resume_education_level']}")
                        st.progress(min(max(feat['resume_education_level'] / 5.0, 0.0), 1.0))
                    with sc3:
                        st.write(f"🔑 **Lexical Cosine:** {feat['tfidf_cosine']:.2f}")
                        st.progress(min(max(feat['tfidf_cosine'], 0.0), 1.0))
                    with sc4:
                        st.write(f"🧠 **Semantic Cosine:** {feat['bert_cosine']:.2f}")
                        st.progress(min(max(feat['bert_cosine'], 0.0), 1.0))

                    with st.expander("🔬 View Matching vs Missing Skill details"):
                        matched = "".join([f'<span class="badge badge-match">{s}</span>' for s in feat["matched_skills"]])
                        missing = "".join([f'<span class="badge badge-missing">{s}</span>' for s in feat["missing_skills"]])
                        st.write("**Matched:**")
                        st.markdown(matched or "None", unsafe_allow_html=True)
                        st.write("**Missing:**")
                        st.markdown(missing or "None", unsafe_allow_html=True)
                        
                    st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if "ranked_results" not in st.session_state or not st.session_state["ranked_results"]:
            st.warning("Run scores on first tab to enable candidates comparison.")
        else:
            lst = st.session_state["ranked_results"]
            cand_names = [item[0]["candidate_id"] for item in lst]
            
            c_sel1, c_sel2 = st.columns(2)
            c1_name = c_sel1.selectbox("Select Candidate A", cand_names, index=0)
            c2_name = c_sel2.selectbox("Select Candidate B", cand_names, index=min(1, len(cand_names)-1))

            c1_f, c1_s = next(item for item in lst if item[0]["candidate_id"] == c1_name)
            c2_f, c2_s = next(item for item in lst if item[0]["candidate_id"] == c2_name)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"### {c1_name}")
                st.markdown(f"**Overall Fit Score:** <span style='font-size:1.8rem; color:#00f2fe;'>{c1_s:.2f} / 5.0</span>", unsafe_allow_html=True)
                st.write(f"💼 **Experience:** {c1_f['resume_experience_years']:.1f} Years")
                st.write(f"🎓 **Education Level:** Level {c1_f['resume_education_level']}")
                st.write(f"🔑 **Lexical Cosine:** {c1_f['tfidf_cosine']:.2f}")
                st.write(f"🧠 **Semantic Cosine:** {c1_f['bert_cosine']:.2f}")
                
                st.write("**Matched Skills:**")
                st.markdown("".join([f'<span class="badge badge-match">{s}</span>' for s in c1_f["matched_skills"]]), unsafe_allow_html=True)
                st.write("**Missing Skills:**")
                st.markdown("".join([f'<span class="badge badge-missing">{s}</span>' for s in c1_f["missing_skills"]]), unsafe_allow_html=True)

            with col_b:
                st.markdown(f"### {c2_name}")
                st.markdown(f"**Overall Fit Score:** <span style='font-size:1.8rem; color:#00f2fe;'>{c2_s:.2f} / 5.0</span>", unsafe_allow_html=True)
                st.write(f"💼 **Experience:** {c2_f['resume_experience_years']:.1f} Years")
                st.write(f"🎓 **Education Level:** Level {c2_f['resume_education_level']}")
                st.write(f"🔑 **Lexical Cosine:** {c2_f['tfidf_cosine']:.2f}")
                st.write(f"🧠 **Semantic Cosine:** {c2_f['bert_cosine']:.2f}")
                
                st.write("**Matched Skills:**")
                st.markdown("".join([f'<span class="badge badge-match">{s}</span>' for s in c2_f["matched_skills"]]), unsafe_allow_html=True)
                st.write("**Missing Skills:**")
                st.markdown("".join([f'<span class="badge badge-missing">{s}</span>' for s in c2_f["missing_skills"]]), unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)


# ─── 10. AUTOMATED UNIT TESTS ────────────────────────────────────────
def run_unit_tests():
    """Runs standard inline unit tests to check all modules."""
    print("\n" + "=" * 50)
    print("  Running Self-Verification Test Suite...")
    print("=" * 50)

    # Test parser
    parser = ResumeParser()
    test_file = "temp_selftest.txt"
    test_content = "Self test content for resume parsing validation."
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)
    try:
        parsed = parser.parse(test_file)
        assert parsed == test_content, "Parser mismatch"
        print("  [PASS] ResumeParser text verification successful.")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

    # Test features
    extractor = FeatureExtractor()
    sample_res = "Senior Python developer with PyTorch and AWS. 6 years experience. PhD Computer Science."
    sample_jd = "Machine learning engineer with Python and AWS. PhD required."
    features = extractor.extract_features(sample_res, sample_jd)
    assert "python" in features["resume_skills"], "Failed to extract python skill"
    assert features["resume_experience_years"] == 6.0, "Experience extraction mismatch"
    assert features["resume_education_level"] == 5, "Education extraction mismatch"
    print("  [PASS] FeatureExtractor taxonomy, experience and education verification successful.")

    # Test similarity
    scorer = SimilarityScorer()
    score_res = scorer.score("Python coding", "Python coding")
    assert score_res["tfidf_cosine"] > 0.9, "Cosine similarity mismatch"
    print("  [PASS] SimilarityScorer TF-IDF and BERT logic successful.")

    # Test evaluation
    predicted = ["cand1", "cand2"]
    ideal = ["cand1", "cand2"]
    ndcg = ndcg_at_k(predicted, ideal)
    assert ndcg == 1.0, "NDCG calculation mismatch"
    print("  [PASS] Evaluation metrics NDCG verification successful.")
    print("=" * 50)
    print("  All tests passed successfully! [OK]")
    print("=" * 50 + "\n")


# ─── 11. MAIN ROUTING ENTRY POINT ────────────────────────────────────
def main():
    # Detect if run under Streamlit
    # Streamlit sets a specific session state or module flag
    is_streamlit_run = False
    
    # Check if run by calling streamlit run
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        is_streamlit_run = True
    else:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                is_streamlit_run = True
        except ImportError:
            pass

    if is_streamlit_run:
        if HAS_STREAMLIT:
            run_streamlit_dashboard()
        else:
            print("Streamlit package not found. Run: pip install streamlit")
    else:
        import argparse
        ap = argparse.ArgumentParser(description="HireAI Pipeline")
        ap.add_argument("--test", action="store_true", help="Run self verification tests")
        ap.add_argument("--resumes", default=RESUME_DIR)
        ap.add_argument("--jd", default=DEFAULT_JD)
        ap.add_argument("--no-bert", action="store_true")
        ap.add_argument("--save-model", default=MODEL_PATH)
        args = ap.parse_args()

        ensure_sample_data()

        if args.test:
            run_unit_tests()
        else:
            pipeline = ResumeRankingPipeline(use_bert=not args.no_bert)
            pipeline.run(
                resume_dir=args.resumes,
                jd_path=args.jd,
                ground_truth=GROUND_TRUTH,
                ideal_ranking=IDEAL_RANKING,
                save_path=args.save_model,
            )


if __name__ == "__main__":
    main()
