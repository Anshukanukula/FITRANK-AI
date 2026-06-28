import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
import random

random.seed(42)

# ─────────────────────────────────────────────
# PATH RESOLUTION UTILITY
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

def find_file(filename: str, start: Path = BASE_DIR) -> Path:
    for path in start.rglob(filename):
        return path
    return start / filename

# ─────────────────────────────────────────────
# SKILL TAXONOMY
# ─────────────────────────────────────────────
taxonomy_path = find_file('skill_taxonomy.json')
taxonomy = {}
if taxonomy_path.exists():
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)

# ─────────────────────────────────────────────
# JD PARSER
# ─────────────────────────────────────────────
def parse_job_description(jd_path):
    if not Path(jd_path).exists():
        return {
            "title_keywords": ["ai","ml","machine learning","nlp","search","retrieval"],
            "required_skills": ["embeddings","vector search","python","evaluation","vector database"],
            "preferred_skills": ["fine-tuning","learning-to-rank"],
            "min_experience": 5.0, "max_experience": 9.0
        }
    with open(jd_path, 'r', encoding='utf-8') as f:
        jd_text = f.read().lower()
    min_exp, max_exp = 5.0, 9.0
    m = re.search(r'(\d+)\s*[-–—]\s*(\d+)\s*years', jd_text)
    if m:
        min_exp, max_exp = float(m.group(1)), float(m.group(2))
    required_skills, preferred_skills = [], []
    for key, synonyms in taxonomy.items():
        matched = key in jd_text or any(s in jd_text for s in synonyms)
        if matched:
            pref = re.search(r'(?:preferred|like you to have|nice to have).*?' + re.escape(key), jd_text, re.DOTALL)
            (preferred_skills if pref else required_skills).append(key)
    if not required_skills:
        required_skills = ["embeddings","vector search","python","evaluation","vector database"]
    if not preferred_skills:
        preferred_skills = ["fine-tuning","learning-to-rank"]
    return {
        "title_keywords": ["ai","ml","machine learning","nlp","search","retrieval"],
        "required_skills": required_skills, "preferred_skills": preferred_skills,
        "min_experience": min_exp, "max_experience": max_exp
    }

# ─────────────────────────────────────────────
# BM25
# ─────────────────────────────────────────────
def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())

class CustomBM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.corpus_size = len(corpus_tokens)
        self.avg_doc_len = sum(len(d) for d in corpus_tokens) / self.corpus_size
        self.doc_lens = [len(d) for d in corpus_tokens]
        df = {}
        for doc in corpus_tokens:
            for w in set(doc):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: np.log((self.corpus_size - f + 0.5)/(f + 0.5) + 1.0) for w, f in df.items()}

    def get_scores(self, doc_tokens_list, query_tokens):
        scores = []
        for i, doc in enumerate(doc_tokens_list):
            score = 0.0
            dl = self.doc_lens[i]
            tf = {}
            for w in doc: tf[w] = tf.get(w,0)+1
            for w in query_tokens:
                if w in tf:
                    idf = self.idf.get(w, 0.0)
                    tf_w = tf[w]
                    denom = tf_w + self.k1*(1 - self.b + self.b*dl/self.avg_doc_len)
                    score += idf * (tf_w * (self.k1+1)) / denom
            scores.append(score)
        return np.array(scores)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FOUNDING_YEARS = {
    'Krutrim': 2023, 'Sarvam AI': 2023, 'Observe.AI': 2017, 'Rephrase.ai': 2019,
    'Glance': 2019, 'Aganitha': 2017, 'Haptik': 2013, 'Mad Street Den': 2013,
    'Niramai': 2016, 'Saarthi.ai': 2017, 'Verloop.io': 2015, 'Wysa': 2015,
    'Yellow.ai': 2016, 'CRED': 2018, 'Swiggy': 2014, 'Razorpay': 2014,
    'Zomato': 2008, 'Flipkart': 2007, 'Meesho': 2015, 'Nykaa': 2012,
    'InMobi': 2007, "BYJU'S": 2011, 'PolicyBazaar': 2008, 'Ola': 2010,
    'Paytm': 2010, 'PharmEasy': 2015, 'PhonePe': 2015, 'Unacademy': 2015,
    'Vedantu': 2011, 'upGrad': 2015, 'Freshworks': 2010, 'Dream11': 2008
}
CONSULTING_FIRMS = {
    'TCS','Infosys','Wipro','Accenture','Cognizant',
    'Capgemini','Tech Mahindra','Mphasis','HCL','Genpact AI'
}
STARTUP_COMPANIES = set(FOUNDING_YEARS.keys())
ENTERPRISE_COMPANIES = {
    'Google','Microsoft','Amazon','Meta','Apple','Adobe','Oracle','SAP',
    'IBM','Salesforce','Netflix','LinkedIn','Twitter','Uber','Lyft'
}

# ─────────────────────────────────────────────
# TRUST ENGINE
# ─────────────────────────────────────────────
def check_trust_score(cand):
    profile = cand.get('profile', {})
    career  = cand.get('career_history', [])
    skills  = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})
    for job in career:
        comp = job.get('company','')
        start_str = job.get('start_date','')
        if comp in FOUNDING_YEARS and start_str:
            try:
                if datetime.strptime(start_str,"%Y-%m-%d").year < FOUNDING_YEARS[comp]:
                    return 0
            except ValueError: pass
    total_yoe = profile.get('years_of_experience', 0)
    for s in skills:
        if s.get('duration_months',0)/12.0 > total_yoe + 5.0:
            return 0
    score = 100
    if sum(1 for s in skills if s.get('proficiency')=='expert' and s.get('duration_months',0)==0) >= 5:
        score = min(score, 40)
    sal = signals.get('expected_salary_range_inr_lpa', {})
    if sal.get('min',0) > sal.get('max',0):
        score = min(score, 40)
    signup = signals.get('signup_date'); active = signals.get('last_active_date')
    if signup and active:
        try:
            if datetime.strptime(signup,"%Y-%m-%d") > datetime.strptime(active,"%Y-%m-%d"):
                score = min(score, 40)
        except ValueError: pass
    for job in career:
        s_, e_ = job.get('start_date'), job.get('end_date')
        if s_ and e_:
            try:
                if datetime.strptime(s_,"%Y-%m-%d") > datetime.strptime(e_,"%Y-%m-%d"):
                    score = min(score, 40); break
            except ValueError: pass
    for s in skills:
        if total_yoe + 1.0 < s.get('duration_months',0)/12.0 <= total_yoe + 5.0:
            score = min(score, 85); break
    return score

# ─────────────────────────────────────────────
# COMPANY INTELLIGENCE ENGINE
# ─────────────────────────────────────────────
def classify_company(company_name: str) -> str:
    if company_name in STARTUP_COMPANIES:
        return "Startup"
    if company_name in CONSULTING_FIRMS:
        return "Consulting/IT Services"
    if company_name in ENTERPRISE_COMPANIES:
        return "Big Tech / Enterprise"
    return "Mid-size / Other"

def company_fit_score(career: list) -> float:
    """Startup-heavy background = better fit for founding team role."""
    if not career:
        return 0.5
    cats = [classify_company(j.get('company','')) for j in career]
    startup_frac   = cats.count("Startup")             / len(cats)
    bigtech_frac   = cats.count("Big Tech / Enterprise") / len(cats)
    consulting_frac = cats.count("Consulting/IT Services") / len(cats)
    score = 0.5 + 0.4 * startup_frac + 0.2 * bigtech_frac - 0.3 * consulting_frac
    return round(min(1.0, max(0.0, score)), 4)

def get_company_environment_tag(career: list) -> str:
    if not career:
        return "Unknown"
    cats = [classify_company(j.get('company','')) for j in career]
    dominant = max(set(cats), key=cats.count)
    return dominant

# ─────────────────────────────────────────────
# CANDIDATE RISK ANALYZER
# ─────────────────────────────────────────────
def compute_risk_score(cand: dict) -> tuple[int, list]:
    """Returns (risk_score 0-100, list_of_risk_warnings)."""
    career  = cand.get('career_history', [])
    signals = cand.get('redrob_signals', {})
    warnings = []
    risk = 0

    # Title regression: later title looks more junior than earlier one
    SENIORITY = ['intern','junior','associate','trainee','sde 1','sde i',
                 'software engineer','developer','sde 2','sde ii',
                 'senior','lead','staff','principal','manager','director','head','vp']
    def seniority_rank(title):
        t = (title or '').lower()
        for i, kw in enumerate(SENIORITY):
            if kw in t:
                return i
        return len(SENIORITY) // 2
    if len(career) >= 2:
        latest_rank = seniority_rank(career[0].get('title') or '')
        oldest_rank = seniority_rank(career[-1].get('title') or '')
        if latest_rank < oldest_rank - 2:
            risk += 30
            warnings.append("Title regression detected (latest role appears more junior than past roles)")

    # Job hopping: avg tenure < 12 months
    if career:
        avg_tenure = sum(j.get('duration_months',0) for j in career) / len(career)
        if avg_tenure < 12:
            risk += 25
            warnings.append(f"Job hopping risk — avg tenure {avg_tenure:.1f} months (<12)")
        elif avg_tenure < 18:
            risk += 10
            warnings.append(f"Short average tenure ({avg_tenure:.1f} months)")

    # Long notice period
    notice = signals.get('notice_period_days', 0)
    if notice > 90:
        risk += 20
        warnings.append(f"Long notice period ({notice} days)")
    elif notice > 60:
        risk += 10
        warnings.append(f"Moderate notice period ({notice} days)")

    # Consulting-only background
    if career and all(j.get('company','') in CONSULTING_FIRMS for j in career):
        risk += 25
        warnings.append("Consulting-only background — limited startup/product experience")

    risk = min(100, risk)
    return risk, warnings

# ─────────────────────────────────────────────
# JOB INTELLIGENCE ENGINE (GAP ANALYSIS)
# ─────────────────────────────────────────────
def evaluate_candidate_jd_gap(cand, jd_specs):
    skills = cand.get('skills', [])
    profile = cand.get('profile', {})
    total_yoe = profile.get('years_of_experience', 0.0)
    
    skill_names = [(s.get('name') or '').lower() for s in skills if (s.get('duration_months') or 0) > 0]
    
    matched_required = []
    missing_required = []
    for r in jd_specs['required_skills']:
        if any(r in n or n in r for n in skill_names):
            matched_required.append(r)
        else:
            missing_required.append(r)
            
    matched_preferred = []
    missing_preferred = []
    for p in jd_specs['preferred_skills']:
        if any(p in n or n in p for n in skill_names):
            matched_preferred.append(p)
        else:
            missing_preferred.append(p)
            
    min_exp = jd_specs['min_experience']
    max_exp = jd_specs['max_experience']
    if total_yoe < min_exp:
        exp_gap = f"Underqualified ({total_yoe} YOE, required min is {min_exp} YOE)"
        exp_status = "Under"
    elif total_yoe > max_exp:
        exp_gap = f"Overqualified ({total_yoe} YOE, typical max is {max_exp} YOE)"
        exp_status = "Over"
    else:
        exp_gap = f"Meets experience range ({total_yoe} YOE, JD asks for {min_exp}-{max_exp} YOE)"
        exp_status = "Meets"
        
    return {
        'matched_required': matched_required,
        'missing_required': missing_required,
        'matched_preferred': matched_preferred,
        'missing_preferred': missing_preferred,
        'yoe': total_yoe,
        'exp_gap': exp_gap,
        'exp_status': exp_status
    }

# ─────────────────────────────────────────────
# RECRUITER COPILOT ENGINE
# ─────────────────────────────────────────────
def generate_recruiter_copilot_content(cand, gap_analysis, risk_warnings, archetypes):
    profile = cand.get('profile', {})
    name = profile.get('anonymized_name', 'Candidate')
    current_title = profile.get('current_title', 'Software Engineer')
    current_company = profile.get('current_company', 'their company')
    primary_archetype = archetypes[0]
    
    matched_req_str = ", ".join(gap_analysis['matched_required'][:3])
    skills_para = f"I was particularly impressed by your deep background in {matched_req_str}." if matched_req_str else "I was particularly impressed by your strong background in AI engineering."
    
    email_template = (
        f"Subject: Founding AI Engineer Opportunity at Redrob AI\n\n"
        f"Hi {name},\n\n"
        f"I came across your profile and noticed your impressive career progression as a {current_title} at {current_company}. "
        f"Given your background, you stand out as a strong fit for our founding team at Redrob AI.\n\n"
        f"{skills_para} Your profile aligns very closely with our **{primary_archetype}** profile, and your experience "
        f"with startup environments would make you an invaluable founding team member.\n\n"
        f"We are building the next generation of trust-aware talent platforms. I'd love to tell you more about our vision. "
        f"Would you be open to a quick 15-minute chat sometime next week?\n\n"
        f"Best regards,\n"
        f"[Recruiter Name]\n"
        f"Redrob AI Founding Team"
    )
    
    questions = []
    if primary_archetype == "Deep Tech Specialist":
        questions.append("How do you approach designing and testing scalable RAG pipelines, particularly around vector databases like Qdrant/FAISS?")
    elif primary_archetype == "Startup AI Builder":
        questions.append("As a founding engineer, you'll have to build rapidly under high ambiguity. Can you tell us about a time you took an AI product from 0 to 1 with minimal direction?")
    elif primary_archetype == "Leadership Candidate":
        questions.append("In a founding team, mentoring junior members and aligning technical architecture with business goals is critical. How do you balance hands-on coding with leadership?")
    else:
        questions.append("What are the most critical engineering choices you make when deploying large language models or embeddings pipelines into production?")
        
    if risk_warnings:
        warn = str(risk_warnings[0]).lower()
        if "job hopping" in warn or "tenure" in warn:
            questions.append("I noticed you've had a few shorter tenures in your recent roles. Could you share what factors drove those transitions and what you're looking for in terms of long-term alignment?")
        elif "notice" in warn:
            questions.append("You have a notice period listed of over 60 days. If selected, are there any options or standard practices at your company to buy out or shorten this notice period so we can get you onboarded sooner?")
        elif "consulting" in warn:
            questions.append("You've spent significant time in consulting/IT services. How do you plan to transition to the fast-paced, high-ownership product-development environment of an early-stage startup?")
        else:
            questions.append("What are your expectations for career growth and stability in your next role?")
    else:
        questions.append("We value profile trust and data integrity. Can you describe a project where you had to debug and validate contradictory datasets or model outputs?")
        
    if gap_analysis['missing_required']:
        missing_s = gap_analysis['missing_required'][0]
        questions.append(f"Our stack relies heavily on {missing_s}. While your profile shows strong adjacent skills, what is your experience with {missing_s} or how would you ramp up on it?")
    elif gap_analysis['missing_preferred']:
        missing_p = gap_analysis['missing_preferred'][0]
        questions.append(f"We are looking for exposure to {missing_p} (preferred skill). Have you had a chance to work with it in side projects, or how would you approach integrating it into your workflow?")
    else:
        questions.append("How do you evaluate and benchmark the quality of your embeddings/retrieval systems to ensure high precision?")
        
    return {
        'outreach_email': email_template,
        'interview_questions': questions
    }

# ─────────────────────────────────────────────
# ARCHETYPE ENGINE
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
def determine_archetypes(cand, features_row):
    profile = cand.get('profile', {})
    career  = cand.get('career_history', [])
    skills  = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})
    headline = (profile.get('headline') or '').lower()
    summary  = (profile.get('summary') or '').lower()
    skill_names = [(s.get('name') or '').lower() for s in skills]
    career_titles = [(j.get('title') or '').lower() for j in career]
    career_text   = " ".join((j.get('description') or '').lower() for j in career)

    archetypes = []
    deep_tech_kw = ['vector search','rag','embeddings','faiss','qdrant','chromadb','milvus','llm evaluation','research']
    if (any(any(kw in s for kw in deep_tech_kw) for s in skill_names) or
            any(kw in headline or kw in summary for kw in deep_tech_kw)):
        github = features_row.get('github', -1)
        if isinstance(github, float) and np.isnan(github): github = -1
        if features_row.get('skill_score', 0) > 0.5 or github > 60:
            archetypes.append("Deep Tech Specialist")

    startup_kw = ['startup','early stage','early-stage','founding','built from scratch',
                  'full stack','full-stack','ownership','product launch','first engineer']
    if any(kw in headline or kw in summary or kw in career_text for kw in startup_kw):
        archetypes.append("Startup AI Builder")

    if features_row.get('growth_score', 0) >= 0.8:
        archetypes.append("Career Growth Standout")

    lead_kw = ['lead','manager','director','head','vp','architect','principal']
    if any(any(kw in t for kw in lead_kw) for t in career_titles[:2]):
        archetypes.append("Leadership Candidate")
    elif any(kw in career_text or kw in headline for kw in ['mentoring','led a team','managing']):
        archetypes.append("Leadership Candidate")

    notice = signals.get('notice_period_days', 90)
    if signals.get('open_to_work_flag', False) or notice <= 30:
        archetypes.append("Immediate Availability")
    else:
        try:
            active_dt = datetime.strptime(signals.get('last_active_date',''), "%Y-%m-%d")
            if (datetime(2026, 6, 12) - active_dt).days <= 30:
                archetypes.append("Immediate Availability")
        except ValueError:
            pass

    if not archetypes:
        archetypes.append("AI Engineering Generalist")
    return archetypes

# ─────────────────────────────────────────────
# STRUCTURED REASONING BUILDER
# ─────────────────────────────────────────────
def build_retrieval_consensus(retrieval_agreement: float) -> str:
    if retrieval_agreement >= 0.80:
        return "High"
    elif retrieval_agreement >= 0.55:
        return "Medium"
    return "Low"

def generate_reasoning(cand, score, conf, retrieval_agreement,
                        matched_skills, archetypes, risk_score, risk_warnings,
                        company_env, skill_score, yoe_score, trust_score, title_score):
    profile  = cand.get('profile', {})
    signals  = cand.get('redrob_signals', {})
    name     = profile.get('anonymized_name', 'Candidate')
    yoe      = profile.get('years_of_experience', 0.0)
    company  = profile.get('current_company', 'their company')
    title    = profile.get('current_title', 'Engineer')
    notice   = signals.get('notice_period_days', 90)
    primary  = archetypes[0] if archetypes else "AI Engineering Generalist"
    skills_str = ", ".join(matched_skills[:3]) if matched_skills else "AI engineering"
    cand_id  = cand.get('candidate_id', name)
    
    # Deterministic choice based on candidate ID hash to generate diverse templates
    hash_idx = sum(ord(c) for c in cand_id)
    
    # Lead-in variations
    lead_ins = [
        f"{name} offers {yoe:.1f} years of engineering experience, currently operating as a {title} at {company}.",
        f"Currently working as a {title} at {company}, {name} brings a strong background of {yoe:.1f} YOE.",
        f"With a solid track record of {yoe:.1f} years, {name} (currently {title} at {company}) presents a compelling profile."
    ]
    lead_in = lead_ins[hash_idx % len(lead_ins)]
    
    # Strength/Archetype variations
    archetype_descs = {
        "Startup AI Builder": [
            f"Classified as a {primary}, they show strong exposure to fast-paced product environments and hands-on delivery.",
            f"As a {primary}, they have proven experience building in high-growth startup ecosystems.",
            f"Their profile aligns as a {primary}, suggesting versatility and capability to build from scratch."
        ],
        "Deep Tech Specialist": [
            f"As a {primary}, they demonstrate deep technical focus on complex algorithms and machine learning architecture.",
            f"Classified as a {primary}, they possess advanced domain expertise suited for hard AI challenges.",
            f"They show a strong {primary} profile with research or specialized implementation skills."
        ],
        "Immediate Availability": [
            f"They represent an {primary} profile with short notice period, making them highly actionable.",
            f"Classified under {primary}, their quick availability is paired with robust engineering fundamentals.",
            f"Their immediate availability status is highly favorable for rapid onboarding requirements."
        ],
        "Career Growth Standout": [
            f"A {primary} candidate demonstrating stable job tenures and upward career progression.",
            f"As a {primary}, they show a consistent track record of growing responsibilities and stability.",
            f"They represent a {primary} with balanced experience and a history of steady career development."
        ],
        "AI Engineering Generalist": [
            f"They bring a versatile set of skills as an {primary}, spanning software engineering and applied AI systems.",
            f"As an {primary}, they show equal strength in software craftsmanship and model deployment.",
            f"They present a solid {primary} profile, capable of crossing stack boundaries effectively."
        ]
    }
    
    arch_options = archetype_descs.get(primary, archetype_descs["AI Engineering Generalist"])
    arch_desc = arch_options[hash_idx % len(arch_options)]
    
    # Skill variations
    skill_options = [
        f"They demonstrate solid alignment with our required stack, particularly in {skills_str}.",
        f"Technical alignment is strong, showing direct experience with {skills_str}.",
        f"Key engineering strengths include active experience in {skills_str}."
    ]
    skill_desc = skill_options[hash_idx % len(skill_options)]
    
    # Trust engine text
    if trust_score >= 0.95:
        trust_desc = f"Trust verification is complete with a clean {trust_score*100:.0f}% rating."
    elif trust_score >= 0.50:
        trust_desc = f"Timeline checks reveal minor warnings (trust rating: {trust_score*100:.0f}%)."
    else:
        trust_desc = f"Caution: Trust engine flagged significant timeline/experience inconsistencies (trust rating: {trust_score*100:.0f}%)."
        
    # Notice & Relocation text
    if risk_score > 25:
        if notice > 60:
            notice_desc = f"Notice period of {notice} days presents a potential timeline risk."
        else:
            notice_desc = f"Notice period is {notice} days; relocation and team integration checks are advised."
    else:
        notice_desc = f"Notice period ({notice} days) and location alignment are favorable."
        
    # Assemble final reasoning dynamically
    parts = [lead_in, arch_desc, skill_desc, trust_desc]
    if risk_score > 25:
        parts.append(notice_desc)
        
    return " ".join(parts)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FitRank AI Ranking Engine")
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--out",        type=str, required=True)
    args = parser.parse_args()

    print("Loading job description...")
    jd_path  = find_file('job_description.txt')
    jd_specs = parse_job_description(jd_path)

    expanded = []
    for skill in jd_specs['required_skills'] + jd_specs['preferred_skills']:
        expanded.append(skill)
        expanded.extend(taxonomy.get(skill, []))
    query_tokens = tokenize(" ".join(set(expanded)))

    print(f"Reading candidates from {args.candidates}...")
    candidates = []
    with open(args.candidates, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    if content.startswith('['):
        try:
            candidates = json.loads(content)
        except Exception as e:
            print(f"Error parsing candidates as JSON list: {e}")
            sys.exit(1)
    else:
        for line in content.split('\n'):
            if line.strip():
                try:
                    candidates.append(json.loads(line))
                except Exception as e:
                    print(f"Error parsing line: {line[:50]}... - {e}")
                    
    num_candidates = len(candidates)
    print(f"Loaded {num_candidates} candidates.")

    # Detect precomputed artefacts
    use_precomputed = (
        num_candidates == 100000 and
        Path('./artifacts/candidate_data.pkl').exists() and
        Path('./artifacts/candidate_embeddings.npy').exists() and
        Path('./artifacts/jd_embedding.npy').exists()
    )

    # Initialize BM25 model
    with open('./models/bm25_model.pkl','rb') as f:
        bm25 = pickle.load(f)

    corpus_texts, corpus_tokens = [], []
    for cand in candidates:
        p = cand.get('profile',{})
        h, s = p.get('headline',''), p.get('summary','')
        career_str = " ".join(f"{j.get('title','')} {j.get('description','')}"
                               for j in cand.get('career_history',[]))
        corpus_texts.append(f"{h} {s}")
        corpus_tokens.append(tokenize(f"{h} {s} {career_str}"))

    # Compute BM25 scores
    bm25_scores = bm25.get_scores(corpus_tokens, query_tokens)
    if bm25_scores.max() > 0:
        bm25_scores /= bm25_scores.max()
    bm25_ranks  = np.argsort(np.argsort(-bm25_scores))

    cosine_sims = None

    if use_precomputed:
        print("Using precomputed artefacts...")
        try:
            with open('./artifacts/candidate_data.pkl','rb') as f:
                pre_ids, pre_tokens = pickle.load(f)
            input_ids = [c['candidate_id'] for c in candidates]
            if input_ids == pre_ids:
                embeddings    = np.load('./artifacts/candidate_embeddings.npy')
                jd_embedding  = np.load('./artifacts/jd_embedding.npy')
                cosine_sims = np.dot(embeddings, jd_embedding) / (
                    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(jd_embedding)
                )
                cosine_sims = (cosine_sims + 1.0) / 2.0
                print("Successfully loaded precomputed embeddings and computed cosine similarities.")
            else:
                print("[WARN] Input candidate IDs do not match precomputed IDs! Forcing on-the-fly computation.")
                use_precomputed = False
        except BaseException as e:
            print(f"[WARN] Failed to load precomputed embeddings: {e}. Forcing on-the-fly computation.")
            use_precomputed = False

    if cosine_sims is None:
        # We need on-the-fly BGE-M3 encoding
        dense_loaded = False
        try:
            print("Loading BGE model for on-the-fly encoding...")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('./models/bge-m3')
            dense_loaded = True
        except BaseException as e:
            print(f"[WARN] BGE-M3/PyTorch failed to load ({type(e).__name__}: {e}). Falling back to Pure BM25 pipeline.")
            
        if dense_loaded:
            try:
                embeddings   = model.encode(corpus_texts, show_progress_bar=False)
                jd_embedding = np.load('./artifacts/jd_embedding.npy')
                cosine_sims = np.dot(embeddings, jd_embedding) / (
                    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(jd_embedding)
                )
                cosine_sims = (cosine_sims + 1.0) / 2.0
            except BaseException as e:
                print(f"[WARN] BGE-M3 encoding failed: {e}. Falling back to BM25.")
                cosine_sims = bm25_scores
        else:
            cosine_sims = bm25_scores

    embed_ranks = np.argsort(np.argsort(-cosine_sims))
    rrf_scores  = 1.0 / (60 + bm25_ranks) + 1.0 / (60 + embed_ranks)
    rrf_scores  /= rrf_scores.max()

    gbm = lgb.Booster(model_file='./models/ltr_model.txt')

    # Feature extraction
    candidate_features = []
    skipped_honeypots  = 0
    print("Building feature matrices & Trust Engine pass...")

    for idx, cand in enumerate(candidates):
        cid     = cand['candidate_id']
        profile = cand.get('profile',{})
        career  = cand.get('career_history',[])
        skills  = cand.get('skills',[])
        signals = cand.get('redrob_signals',{})

        trust = check_trust_score(cand)
        if trust == 0:
            skipped_honeypots += 1
            continue

        total_yoe = profile.get('years_of_experience', 0.0)
        if jd_specs['min_experience'] <= total_yoe <= jd_specs['max_experience']:
            yoe_fit = 1.0
        elif jd_specs['min_experience'] - 1 <= total_yoe <= jd_specs['max_experience'] + 3:
            yoe_fit = 0.7
        else:
            yoe_fit = 0.2
            
        if total_yoe < jd_specs['min_experience']:
            yoe_fit *= 0.4

        avg_tenure = sum(j.get('duration_months',0) for j in career)/len(career) if career else 24.0
        tenure_fit = min(1.0, avg_tenure / 36.0)

        growth_fit = 0.5
        if len(career) > 1:
            roles = [(j.get('title') or '').lower() for j in career]
            has_lead   = any(kw in roles[0]  for kw in ['lead','senior','staff','principal','manager','head'])
            has_junior = any(kw in roles[-1] for kw in ['junior','intern','associate','sde 1','trainee'])
            if has_lead and has_junior: growth_fit = 1.0
            elif has_lead:              growth_fit = 0.8

        skill_names = [(s.get('name') or '').lower() for s in skills if (s.get('duration_months') or 0) > 0]
        matched_req  = sum(1 for r in jd_specs['required_skills']  if any(r in n or n in r for n in skill_names))
        matched_pref = sum(1 for p in jd_specs['preferred_skills'] if any(p in n or n in p for n in skill_names))
        req_cov  = matched_req  / len(jd_specs['required_skills'])  if jd_specs['required_skills']  else 1.0
        pref_cov = matched_pref / len(jd_specs['preferred_skills']) if jd_specs['preferred_skills'] else 1.0
        skill_fit = 0.7 * req_cov + 0.3 * pref_cov

        loc = (profile.get('location') or '').lower()
        country = (profile.get('country') or '').lower()
        willing_reloc = signals.get('willing_to_relocate', False)
        is_india = (country == 'india' or any(c in loc for c in ['pune','noida','delhi','ncr','gurgaon','bangalore','bengaluru','hyderabad','mumbai','chennai','kolkata']))
        tier1_cities = ['pune','noida','delhi','ncr','gurgaon','bangalore','bengaluru','hyderabad','mumbai','chennai']
        is_tier1 = any(c in loc for c in tier1_cities)
        
        if is_tier1 and is_india:
            loc_fit = 1.0
        elif is_india:
            loc_fit = 0.8
        elif willing_reloc and is_india:
            loc_fit = 0.7
        elif willing_reloc:
            loc_fit = 0.4
        else:
            loc_fit = 0.1

        notice_days = signals.get('notice_period_days', 90)
        if notice_days <= 30:
            notice_fit = 1.0
        elif notice_days <= 60:
            notice_fit = 0.8
        elif notice_days <= 90:
            notice_fit = 0.5
        else:
            notice_fit = 0.1

        active_days_ago = 180
        try:
            active_dt = datetime.strptime(signals.get('last_active_date',''), "%Y-%m-%d")
            active_days_ago = (datetime(2026, 6, 12) - active_dt).days
        except ValueError:
            pass
        active_fit    = max(0.1, 1.0 - active_days_ago/180.0) if active_days_ago <= 180 else 0.1
        resp_rate     = signals.get('recruiter_response_rate', 0.0)
        open_to_work  = 1.0 if signals.get('open_to_work_flag', False) else 0.7
        interview_rt  = signals.get('interview_completion_rate', 0.0)
        
        saved_score = min(1.0, signals.get('saved_by_recruiters_30d', 0) / 20.0)
        search_score = min(1.0, signals.get('search_appearance_30d', 0) / 300.0)
        
        behavioral_fit = (
            0.2 * active_fit + 
            0.2 * resp_rate + 
            0.1 * open_to_work + 
            0.1 * interview_rt +
            0.2 * saved_score +
            0.2 * search_score
        )

        current_title = (profile.get('current_title') or '').lower()
        if any(kw in current_title for kw in ['ai','ml','machine learning','nlp','search','retrieval','data scientist']):
            title_fit = 1.0
        elif 'software engineer' in current_title or 'developer' in current_title:
            title_fit = 0.6
        else:
            title_fit = 0.0

        assess_scores = signals.get('skill_assessment_scores', {})
        matched_assess_scores = []
        for sname, score in assess_scores.items():
            sname_lower = (sname or '').lower()
            if any(req in sname_lower or sname_lower in req for req in jd_specs['required_skills'] + jd_specs['preferred_skills']):
                matched_assess_scores.append(score / 100.0)
                
        if matched_assess_scores:
            assess_fit = sum(matched_assess_scores) / len(matched_assess_scores)
        else:
            assess_fit = 0.0

        matched_skills_names = [
            s.get('name') for s in skills
            if any((s.get('name') or '').lower() == r for r in jd_specs['required_skills'])
        ]

        comp_fit   = company_fit_score(career)
        comp_env   = get_company_environment_tag(career)

        risk_score, risk_warnings = compute_risk_score(cand)

        candidate_features.append({
            'candidate_id': cid,
            'semantic_score': cosine_sims[idx],
            'rrf_score': rrf_scores[idx],
            'skill_score': skill_fit,
            'yoe_score': yoe_fit,
            'tenure_score': tenure_fit,
            'growth_score': growth_fit,
            'location_score': loc_fit,
            'notice_score': notice_fit,
            'behavioral_score': behavioral_fit,
            'trust_score': trust / 100.0,
            'title_score': title_fit,
            'assessment_score': assess_fit,
            'company_fit': comp_fit,
            'company_env': comp_env,
            'risk_score': risk_score,
            'risk_warnings': risk_warnings,
            'matched_skills_names': matched_skills_names,
            'notice_days': notice_days,
            'github': signals.get('github_activity_score', -1),
            'rank_bm25': bm25_ranks[idx],
            'rank_embed': embed_ranks[idx],
            'candidate_dict': cand
        })

    print(f"Trust Engine: excluded {skipped_honeypots} honeypots.")
    df_pred = pd.DataFrame(candidate_features)

    # Model selection
    use_lightgbm = True
    cfg_path = Path('./models/best_model_config.json')
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            use_lightgbm = cfg.get('use_lightgbm', True)
            print(f"Model Selection Layer: use_lightgbm = {use_lightgbm}")
        except Exception as e:
            print(f"Config load error: {e}. Defaulting to LightGBM.")

    FEATURE_COLS = [
        'semantic_score','rrf_score','skill_score','yoe_score',
        'tenure_score','growth_score','location_score','notice_score',
        'behavioral_score','trust_score','title_score','assessment_score'
    ]
    if use_lightgbm:
        print("Scoring with LightGBM LTR...")
        df_pred['fit_score'] = gbm.predict(df_pred[FEATURE_COLS])
    else:
        print("Scoring with Weighted Baseline...")
        def baseline(row):
            cand   = row['candidate_dict']
            career = cand.get('career_history',[])
            b = (
                0.20*row['semantic_score'] + 0.20*row['rrf_score'] +
                0.20*row['skill_score']    + 0.10*row['yoe_score']  +
                0.05*row['tenure_score']   + 0.10*row['location_score'] +
                0.05*row['notice_score']   + 0.10*row['behavioral_score']
            ) * row['trust_score']
            b = 0.8*b + 0.2*row['title_score']
            if career and all(j.get('company','') in CONSULTING_FIRMS for j in career):
                b *= 0.3
            return b
        df_pred['fit_score'] = df_pred.apply(baseline, axis=1)

    # Scale fit scores to [0.0, 1.0] range
    if use_lightgbm:
        max_fit = df_pred['fit_score'].max()
        min_fit = df_pred['fit_score'].min()
        if max_fit > min_fit:
            df_pred['fit_score'] = (df_pred['fit_score'] - min_fit) / (max_fit - min_fit)
        else:
            df_pred['fit_score'] = 1.0
    else:
        df_pred['fit_score'] = np.clip(df_pred['fit_score'], 0.0, 1.0)
        
    # Apply hard title match penalty to prevent non-technical titles from ranking high
    df_pred['fit_score'] = df_pred['fit_score'] * (0.1 + 0.9 * df_pred['title_score'])

    # Confidence score (log-scale retrieval agreement)
    max_rank = num_candidates
    df_pred['retrieval_agreement'] = 1.0 - (
        np.abs(np.log(df_pred['rank_bm25']+1) - np.log(df_pred['rank_embed']+1)) /
        np.log(max_rank + 1)
    )
    df_pred['feature_consistency'] = (
        df_pred['semantic_score'] + df_pred['skill_score'] + df_pred['title_score']
    ) / 3.0
    df_pred['confidence_score'] = (
        0.40 * df_pred['retrieval_agreement'] +
        0.30 * df_pred['trust_score'] +
        0.20 * df_pred['feature_consistency'] +
        0.10 * df_pred['fit_score']
    ) * 100.0
    df_pred['confidence_score'] = np.clip(df_pred['confidence_score'], 0.0, 100.0)

    df_pred.sort_values(by=['fit_score','candidate_id'], ascending=[False,True], inplace=True)
    df_pred.reset_index(drop=True, inplace=True)
    top_100 = df_pred.head(100).copy()
    top_100['rank'] = range(1, len(top_100) + 1)

    print("Generating explainable AI reasonings...")
    reasons = []
    for i in range(len(top_100)):
        row = top_100.iloc[i]
        archetypes = determine_archetypes(row['candidate_dict'], row)
        reason = generate_reasoning(
            cand=row['candidate_dict'],
            score=row['fit_score'],
            conf=row['confidence_score'],
            retrieval_agreement=row['retrieval_agreement'],
            matched_skills=row['matched_skills_names'],
            archetypes=archetypes,
            risk_score=row['risk_score'],
            risk_warnings=row['risk_warnings'],
            company_env=row['company_env'],
            skill_score=row['skill_score'],
            yoe_score=row['yoe_score'],
            trust_score=row['trust_score'],
            title_score=row['title_score']
        )
        reasons.append(reason)

    top_100['reasoning'] = reasons
    output_df = top_100[['candidate_id','rank','fit_score','reasoning']].rename(
        columns={'fit_score': 'score'}
    )
    output_df.to_csv(args.out, index=False, encoding='utf-8')
    
    # Write features dataset for app.py diagnostics support
    features_out = args.out.replace('.csv', '_features.csv')
    df_pred.drop(columns=['candidate_dict'], errors='ignore').to_csv(features_out, index=False, encoding='utf-8')
    
    print(f"Done! Top 100 candidates saved to {args.out}")

if __name__ == '__main__':
    main()
