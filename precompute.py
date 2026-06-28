import os
import sys

# Resolve OpenMP conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch.set_num_threads(os.cpu_count())

import json
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import lightgbm as lgb

# ─────────────────────────────────────────────
# PATH RESOLUTION UTILITY
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

def find_file(filename: str, start: Path = BASE_DIR) -> Path:
    """Recursively search for a file starting from BASE_DIR."""
    for path in start.rglob(filename):
        return path
    return start / filename   # return a non-existent path; callers check .exists()

os.makedirs('./models', exist_ok=True)
os.makedirs('./artifacts', exist_ok=True)

# ─────────────────────────────────────────────
# 1. DOWNLOAD / LOAD BGE-M3 MODEL
model = None
if not (Path('./artifacts/candidate_embeddings.npy').exists() and Path('./artifacts/jd_embedding.npy').exists()):
    print("Downloading/loading BAAI/bge-m3 model...")
    if os.path.exists('./models/bge-m3/pytorch_model.bin') and os.path.getsize('./models/bge-m3/pytorch_model.bin') > 2 * 1024 * 1024 * 1024:
        print("Found complete BGE-M3 local weights. Loading offline...")
        model = SentenceTransformer('./models/bge-m3')
    else:
        model = SentenceTransformer('BAAI/bge-m3')
        model.save('./models/bge-m3')
    print("Model loaded and saved.")
else:
    print("Precomputed embeddings found. Bypassing BGE-M3 model loading/downloading.")

# ─────────────────────────────────────────────
# 2. CUSTOM BM25
# ─────────────────────────────────────────────
class CustomBM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.avg_doc_len = sum(len(d) for d in corpus_tokens) / self.corpus_size
        self.doc_lens = [len(d) for d in corpus_tokens]
        df = {}
        for doc in corpus_tokens:
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
        self.idf = {
            w: np.log((self.corpus_size - f + 0.5) / (f + 0.5) + 1.0)
            for w, f in df.items()
        }

    def get_scores(self, doc_tokens_list, query_tokens):
        scores = []
        for i, doc in enumerate(doc_tokens_list):
            score = 0.0
            doc_len = self.doc_lens[i]
            tf = {}
            for w in doc:
                tf[w] = tf.get(w, 0) + 1
            for w in query_tokens:
                if w in tf:
                    idf = self.idf.get(w, 0.0)
                    tf_w = tf[w]
                    denom = tf_w + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                    score += idf * (tf_w * (self.k1 + 1)) / denom
            scores.append(score)
        return np.array(scores)

def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())

# ─────────────────────────────────────────────
# 3. SKILL TAXONOMY + JD PARSING
# ─────────────────────────────────────────────
taxonomy_path = find_file('skill_taxonomy.json')
with open(taxonomy_path, 'r', encoding='utf-8') as f:
    taxonomy = json.load(f)

def parse_job_description(jd_path):
    print(f"Parsing job description from {jd_path}...")
    if not Path(jd_path).exists():
        print("  [WARN] JD file not found — using built-in defaults.")
        return {
            "title_keywords": ["ai", "ml", "machine learning", "nlp", "search", "retrieval"],
            "required_skills": ["embeddings", "vector search", "python", "evaluation", "vector database"],
            "preferred_skills": ["fine-tuning", "learning-to-rank"],
            "min_experience": 5.0,
            "max_experience": 9.0
        }
    with open(jd_path, 'r', encoding='utf-8') as f:
        jd_text = f.read().lower()
    min_exp, max_exp = 5.0, 9.0
    m = re.search(r'(\d+)\s*[-–—]\s*(\d+)\s*years', jd_text)
    if m:
        min_exp, max_exp = float(m.group(1)), float(m.group(2))
        print(f"  Parsed YOE: {min_exp}–{max_exp} years")
    required_skills, preferred_skills = [], []
    for key, synonyms in taxonomy.items():
        matched = key in jd_text or any(s in jd_text for s in synonyms)
        if matched:
            pref_match = re.search(
                r'(?:preferred|like you to have|nice to have).*?' + re.escape(key),
                jd_text, re.DOTALL
            )
            (preferred_skills if pref_match else required_skills).append(key)
    if not required_skills:
        required_skills = ["embeddings", "vector search", "python", "evaluation", "vector database"]
    if not preferred_skills:
        preferred_skills = ["fine-tuning", "learning-to-rank"]
    print("  Required:", required_skills)
    print("  Preferred:", preferred_skills)
    return {
        "title_keywords": ["ai", "ml", "machine learning", "nlp", "search", "retrieval"],
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "min_experience": min_exp,
        "max_experience": max_exp
    }

jd_path = find_file('job_description.txt')
jd_specs = parse_job_description(jd_path)

expanded_query_terms = []
for skill in jd_specs['required_skills'] + jd_specs['preferred_skills']:
    expanded_query_terms.append(skill)
    expanded_query_terms.extend(taxonomy.get(skill, []))
query_tokens = tokenize(" ".join(set(expanded_query_terms)))

# ─────────────────────────────────────────────
# 4. TRUST ENGINE
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
    'TCS', 'Infosys', 'Wipro', 'Accenture', 'Cognizant',
    'Capgemini', 'Tech Mahindra', 'Mphasis', 'HCL', 'Genpact AI'
}

STARTUP_COMPANIES = set(FOUNDING_YEARS.keys())

def check_trust_score(cand):
    profile = cand.get('profile', {})
    career = cand.get('career_history', [])
    skills = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})
    # Hard honeypot: founding year violation
    for job in career:
        comp = job.get('company', '')
        start_str = job.get('start_date', '')
        if comp in FOUNDING_YEARS and start_str:
            try:
                start_yr = datetime.strptime(start_str, "%Y-%m-%d").year
                if start_yr < FOUNDING_YEARS[comp]:
                    return 0
            except ValueError:
                pass
    # Hard honeypot: extreme skill duration
    total_yoe = profile.get('years_of_experience', 0)
    for s in skills:
        if s.get('duration_months', 0) / 12.0 > total_yoe + 5.0:
            return 0
    score = 100
    # Major contradictions → 40
    if sum(1 for s in skills if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0) >= 5:
        score = min(score, 40)
    sal = signals.get('expected_salary_range_inr_lpa', {})
    if sal.get('min', 0) > sal.get('max', 0):
        score = min(score, 40)
    signup = signals.get('signup_date')
    active = signals.get('last_active_date')
    if signup and active:
        try:
            if datetime.strptime(signup, "%Y-%m-%d") > datetime.strptime(active, "%Y-%m-%d"):
                score = min(score, 40)
        except ValueError:
            pass
    for job in career:
        s_, e_ = job.get('start_date'), job.get('end_date')
        if s_ and e_:
            try:
                if datetime.strptime(s_, "%Y-%m-%d") > datetime.strptime(e_, "%Y-%m-%d"):
                    score = min(score, 40); break
            except ValueError:
                pass
    # Minor contradictions → 85
    for s in skills:
        dur = s.get('duration_months', 0) / 12.0
        if total_yoe + 1.0 < dur <= total_yoe + 5.0:
            score = min(score, 85)
            break
    return score

# ─────────────────────────────────────────────
# 5. GOLD LABEL FUNCTION  (replaces pseudo-label baseline)
# ─────────────────────────────────────────────
def compute_gold_label(cand, trust, skill_fit, req_coverage, pref_coverage,
                        growth_fit, title_fit, loc_fit, notice_fit, recruiter_engaged):
    """
    Recruiter-inspired rubric-generated relevance labels — NOT derived from the baseline formula.
    Max raw points = 14; normalised to [0, 1].
    """
    if trust == 0:
        return 0.0

    pts = 0.0

    # Skill match (max 3 pts)
    if req_coverage >= 0.7:
        pts += 3
    elif req_coverage >= 0.4:
        pts += 2
    if pref_coverage >= 0.5:
        pts += 1

    # Growth trajectory (max 2 pts)
    if growth_fit >= 1.0:
        pts += 2
    elif growth_fit >= 0.8:
        pts += 1

    # Title alignment (max 2 pts)
    if title_fit >= 1.0:
        pts += 2
    elif title_fit >= 0.6:
        pts += 1

    # Startup company experience (max 2 pts)
    career = cand.get('career_history', [])
    startup_matches = sum(1 for j in career if j.get('company', '') in STARTUP_COMPANIES)
    if startup_matches >= 2:
        pts += 2
    elif startup_matches == 1:
        pts += 1

    # Location alignment (max 2 pts)
    if loc_fit >= 1.0:
        pts += 2
    elif loc_fit >= 0.7:
        pts += 1

    # Notice period alignment (max 2 pts)
    if notice_fit >= 1.0:
        pts += 2
    elif notice_fit >= 0.8:
        pts += 1

    # Recruiter engagement (max 1 pt)
    if recruiter_engaged:
        pts += 1

    # Contradiction penalty (−3 pts if trust degraded)
    if trust < 100:
        pts -= 3

    # Hard location penalty (if candidate is outside India)
    if loc_fit < 0.5:
        pts -= 4

    # Hard notice penalty (if notice period is > 90 days)
    if notice_fit < 0.2:
        pts -= 4

    pts = max(0.0, pts)
    return round(pts / 14.0, 6)

# ─────────────────────────────────────────────
# 6. LOAD CANDIDATES
# ─────────────────────────────────────────────
candidates_file = find_file('candidates.jsonl')
print(f"Loading candidates from: {candidates_file}")

corpus_texts, corpus_tokens, candidate_ids, candidates = [], [], [], []
with open(candidates_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        cand = json.loads(line)
        candidates.append(cand)
        candidate_ids.append(cand['candidate_id'])
        profile = cand.get('profile', {})
        headline = profile.get('headline', '')
        summary = profile.get('summary', '')
        career_str = " ".join(
            f"{j.get('title','')} {j.get('description','')}"
            for j in cand.get('career_history', [])
        )
        corpus_texts.append(f"{headline} {summary}")
        corpus_tokens.append(tokenize(f"{headline} {summary} {career_str}"))

with open('./artifacts/candidate_data.pkl', 'wb') as f:
    pickle.dump((candidate_ids, corpus_tokens), f)

# ─────────────────────────────────────────────
# 7. BM25 + EMBEDDINGS + RRF
# ─────────────────────────────────────────────
print("Fitting custom BM25 Model...")
bm25 = CustomBM25(corpus_tokens)
with open('./models/bm25_model.pkl', 'wb') as f:
    pickle.dump(bm25, f)
print("BM25 saved.")

embeddings_path = Path('./artifacts/candidate_embeddings.npy')
if embeddings_path.exists():
    print("Loading precomputed candidate embeddings from ./artifacts/candidate_embeddings.npy...")
    embeddings = np.load(embeddings_path)
else:
    print("Encoding candidates with BGE-M3...")
    embeddings = []
    batch_size = 512
    for i in tqdm(range(0, len(corpus_texts), batch_size)):
        embeddings.append(model.encode(corpus_texts[i:i+batch_size], show_progress_bar=False))
    embeddings = np.vstack(embeddings)
    np.save(embeddings_path, embeddings)

if Path('./artifacts/jd_embedding.npy').exists():
    print("Loading precomputed JD embedding...")
    jd_embedding = np.load('./artifacts/jd_embedding.npy')
else:
    jd_text_raw = jd_path.read_text(encoding='utf-8') if jd_path.exists() else \
        "Senior AI Engineer Founding Team embeddings vector search python evaluation vector database"
    jd_embedding = model.encode([jd_text_raw])[0]
    np.save('./artifacts/jd_embedding.npy', jd_embedding)

print("Computing cosine similarities...")
norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(jd_embedding)
cosine_sims = np.dot(embeddings, jd_embedding) / norms
cosine_sims = (cosine_sims + 1.0) / 2.0

print("Computing BM25 scores...")
bm25_scores = bm25.get_scores(corpus_tokens, query_tokens)
if bm25_scores.max() > 0:
    bm25_scores /= bm25_scores.max()

print("Fusing ranks via RRF...")
bm25_ranks = np.argsort(np.argsort(-bm25_scores))
embed_ranks = np.argsort(np.argsort(-cosine_sims))
rrf_scores = 1.0 / (60 + bm25_ranks) + 1.0 / (60 + embed_ranks)
rrf_scores /= rrf_scores.max()

# ─────────────────────────────────────────────
# 8. FEATURE EXTRACTION + GOLD LABELS
# ─────────────────────────────────────────────
features_csv_path = Path('./artifacts/features_dataset.csv')
if features_csv_path.exists():
    print("Features dataset already exists at `./artifacts/features_dataset.csv`. Loading directly...")
    df_features = pd.read_csv(features_csv_path)
else:
    print("Extracting features and computing Gold Labels...")
    features = []

    for idx, cand in enumerate(tqdm(candidates)):
        cid = cand['candidate_id']
        profile = cand.get('profile', {})
        career = cand.get('career_history', [])
        skills = cand.get('skills', [])
        signals = cand.get('redrob_signals', {})

        trust = check_trust_score(cand)
        total_yoe = profile.get('years_of_experience', 0.0)

        # YOE fit with experience gating
        if jd_specs['min_experience'] <= total_yoe <= jd_specs['max_experience']:
            yoe_fit = 1.0
        elif jd_specs['min_experience'] - 1 <= total_yoe <= jd_specs['max_experience'] + 3:
            yoe_fit = 0.7
        else:
            yoe_fit = 0.2
            
        if total_yoe < jd_specs['min_experience']:
            yoe_fit *= 0.4

        # Tenure
        avg_tenure = sum(j.get('duration_months', 0) for j in career) / len(career) if career else 24.0
        tenure_fit = min(1.0, avg_tenure / 36.0)

        # Growth
        growth_fit = 0.5
        if len(career) > 1:
            roles = [(j.get('title') or '').lower() for j in career]
            has_lead = any(kw in roles[0] for kw in ['lead','senior','staff','principal','manager','head'])
            has_junior = any(kw in roles[-1] for kw in ['junior','intern','associate','sde 1','trainee'])
            if has_lead and has_junior:
                growth_fit = 1.0
            elif has_lead:
                growth_fit = 0.8

        # Skill match
        skill_names = [(s.get('name') or '').lower() for s in skills if (s.get('duration_months') or 0) > 0]
        matched_req = sum(1 for r in jd_specs['required_skills'] if any(r in n or n in r for n in skill_names))
        matched_pref = sum(1 for p in jd_specs['preferred_skills'] if any(p in n or n in p for n in skill_names))
        req_coverage = matched_req / len(jd_specs['required_skills']) if jd_specs['required_skills'] else 1.0
        pref_coverage = matched_pref / len(jd_specs['preferred_skills']) if jd_specs['preferred_skills'] else 1.0
        skill_fit = 0.7 * req_coverage + 0.3 * pref_coverage

        # Location (hard location checks)
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

        # Notice (strict notice period thresholds)
        notice_days = signals.get('notice_period_days', 90)
        if notice_days <= 30:
            notice_fit = 1.0
        elif notice_days <= 60:
            notice_fit = 0.8
        elif notice_days <= 90:
            notice_fit = 0.5
        else:
            notice_fit = 0.1

        # Behavioral (recruiter engagement signals integrated)
        active_date_str = signals.get('last_active_date', '')
        active_days_ago = 180
        if active_date_str:
            try:
                active_days_ago = (datetime(2026, 6, 12) - datetime.strptime(active_date_str, "%Y-%m-%d")).days
            except ValueError:
                pass
        active_fit = max(0.1, 1.0 - active_days_ago / 180.0) if active_days_ago <= 180 else 0.1
        resp_rate = signals.get('recruiter_response_rate', 0.0)
        open_to_work = 1.0 if signals.get('open_to_work_flag', False) else 0.7
        interview_rate = signals.get('interview_completion_rate', 0.0)
        
        saved_score = min(1.0, signals.get('saved_by_recruiters_30d', 0) / 20.0)
        search_score = min(1.0, signals.get('search_appearance_30d', 0) / 300.0)
        
        behavioral_fit = (
            0.2 * active_fit + 
            0.2 * resp_rate + 
            0.1 * open_to_work + 
            0.1 * interview_rate +
            0.2 * saved_score +
            0.2 * search_score
        )

        # Title
        current_title = (profile.get('current_title') or '').lower()
        if any(kw in current_title for kw in ['ai','ml','machine learning','nlp','search','retrieval','data scientist']):
            title_fit = 1.0
        elif 'software engineer' in current_title or 'developer' in current_title:
            title_fit = 0.6
        else:
            title_fit = 0.0

        # Skill assessment score feature
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

        # Baseline score (updated with location & notice period weights)
        is_consulting_only = len(career) > 0 and all(j.get('company','') in CONSULTING_FIRMS for j in career)
        baseline_score = (
            0.20 * cosine_sims[idx] + 0.20 * rrf_scores[idx] +
            0.20 * skill_fit + 0.10 * yoe_fit + 0.05 * tenure_fit +
            0.10 * loc_fit + 0.05 * notice_fit + 0.10 * behavioral_fit
        ) * (trust / 100.0)
        baseline_score = 0.8 * baseline_score + 0.2 * title_fit
        if is_consulting_only:
            baseline_score *= 0.3

        # Gold label (rubric-generated relevance labels)
        recruiter_engaged = signals.get('saved_by_recruiters_30d', 0) >= 5 or signals.get('search_appearance_30d', 0) >= 100
        gold_score = compute_gold_label(
            cand, trust, skill_fit, req_coverage, pref_coverage, growth_fit, title_fit, loc_fit, notice_fit, recruiter_engaged
        )

        # Retrieval agreement
        rank_bm25 = bm25_ranks[idx]
        rank_embed = embed_ranks[idx]
        max_rank = len(candidates)
        retrieval_agreement = 1.0 - (
            abs(np.log(rank_bm25 + 1) - np.log(rank_embed + 1)) / np.log(max_rank + 1)
        )

        features.append({
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
            'retrieval_agreement': retrieval_agreement,
            'target_score': baseline_score,
            'gold_score': gold_score,
        })

    df_features = pd.DataFrame(features)
    df_features.to_csv(features_csv_path, index=False)
    print("Features dataset saved.")

# ─────────────────────────────────────────────
# 9. LIGHTGBM TRAINING ON RECRUITER PREFERENCES (LAMBDARANK)
# ─────────────────────────────────────────────
print("Processing recruiter preference dataset...")
FEATURE_COLS = [
    'semantic_score', 'rrf_score', 'skill_score', 'yoe_score',
    'tenure_score', 'growth_score', 'location_score', 'notice_score',
    'behavioral_score', 'trust_score', 'title_score', 'assessment_score'
]

# Load recruiter preference dataset
df_pairs = pd.read_csv('./artifacts/recruiter_preference_dataset.csv')

# Calculate win ratios
stats = {}
for idx, row in df_pairs.iterrows():
    a, b, w = row['cand_a_id'], row['cand_b_id'], row['winner']
    for cid in [a, b]:
        if cid not in stats:
            stats[cid] = {'wins': 0, 'losses': 0, 'ties': 0}
    if w == 'A':
        stats[a]['wins'] += 1
        stats[b]['losses'] += 1
    elif w == 'B':
        stats[b]['wins'] += 1
        stats[a]['losses'] += 1
    else:
        stats[a]['ties'] += 1
        stats[b]['ties'] += 1
        
candidate_utilities = {}
for cid, s in stats.items():
    total = s['wins'] + s['losses'] + s['ties']
    if total > 0:
        candidate_utilities[cid] = (s['wins'] + 0.5 * s['ties']) / total
    else:
        candidate_utilities[cid] = 0.5
        
# Add utility to df_features
df_features['recruiter_utility'] = df_features['candidate_id'].map(candidate_utilities)

# Train a regression model on candidates with actual utilities to predict utilities for the rest
df_labeled = df_features[df_features['recruiter_utility'].notna()].copy()
X_lab = df_labeled[FEATURE_COLS]
y_lab = df_labeled['recruiter_utility']
reg_dataset = lgb.Dataset(X_lab, label=y_lab)
reg_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 15,
    'learning_rate': 0.05,
    'verbose': -1
}
reg_model = lgb.train(reg_params, reg_dataset, num_boost_round=100)

# Propagate utility to all 100K candidates
df_features['recruiter_utility_pred'] = reg_model.predict(df_features[FEATURE_COLS])
df_features.loc[df_features['recruiter_utility'].notna(), 'recruiter_utility_pred'] = df_features['recruiter_utility']

# Bin recruiter_utility_pred into discrete relevance grades (0 to 3) for LTR training
cohort_size = 10
num_candidates = len(df_features)
num_cohorts = num_candidates // cohort_size

df_features = df_features.head(num_cohorts * cohort_size).copy()
df_features['query_id'] = np.repeat(range(num_cohorts), cohort_size)
df_features['relevance_grade'] = pd.qcut(df_features['recruiter_utility_pred'], q=4, labels=[0, 1, 2, 3]).astype(int)

# Sort by query_id for LightGBM grouping requirements
df_features = df_features.sort_values(by='query_id').reset_index(drop=True)
df_features.to_csv('./artifacts/features_dataset.csv', index=False)
print("Updated features_dataset.csv with relevance grades and query cohorts.")

# Training splits
train_queries = int(num_cohorts * 0.8)
split_idx = train_queries * cohort_size

train_df = df_features.iloc[:split_idx]
test_df = df_features.iloc[split_idx:]

train_groups = train_df.groupby('query_id').size().tolist()
test_groups = test_df.groupby('query_id').size().tolist()

train_data = lgb.Dataset(train_df[FEATURE_COLS], label=train_df['relevance_grade'], group=train_groups)
test_data = lgb.Dataset(test_df[FEATURE_COLS], label=test_df['relevance_grade'], group=test_groups, reference=train_data)

# LambdaRank LTR model parameters
ltr_params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

print("Training LightGBM LambdaRank LTR model on Recruiter Preference Utilities...")
gbm = lgb.train(
    ltr_params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data]
)

gbm.save_model('./models/ltr_model.txt')
print("LightGBM LambdaRank model saved to ./models/ltr_model.txt")
print("Precomputation complete!")
