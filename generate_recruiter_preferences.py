import pandas as pd
import numpy as np
import os
import requests
import json
import re
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
FEATURES_CSV = BASE_DIR / "artifacts" / "features_dataset.csv"
OUTPUT_CSV = BASE_DIR / "artifacts" / "recruiter_preference_dataset.csv"
API_KEY = os.environ.get("GEMINI_API_KEY", "")

def call_gemini_comparator(cand_a, cand_b):
    """Calls gemini-2.5-flash to compare two candidates and select the winner."""
    prompt = f"""You are an expert technical recruiter hiring a Senior AI Engineer (Founding Team).
Role Requirements:
- 5-9 years of experience (preferred).
- Stack: Embeddings, Vector Search (Qdrant/FAISS), Evaluation, Python.
- Startup-heavy career background with high growth progression.
- Location: Noida/Pune preferred (or willing to relocate).
- Notice period: Short notice preferred (<30 days).
- High profile integrity (trust_score = 1.0). If trust_score is low or 0, candidate must be rejected immediately.

Candidate A Profile:
- Candidate ID: {cand_a['candidate_id']}
- Semantic Match: {cand_a['semantic_score']:.2f}
- Skill Match: {cand_a['skill_score']:.2f}
- Years of Exp (YOE): {cand_a['yoe_score'] * 10.0:.1f}
- Avg Tenure: {cand_a['tenure_score'] * 36.0:.1f} months
- Growth Trajectory: {cand_a['growth_score']:.2f} (1.0 = junior to senior progression)
- Notice Period Fit: {cand_a['notice_score']:.2f}
- Trust Score: {cand_a['trust_score']:.2f}
- Title Match: {cand_a['title_score']:.2f} (1.0 = AI/ML title, 0 = non-tech title)

Candidate B Profile:
- Candidate ID: {cand_b['candidate_id']}
- Semantic Match: {cand_b['semantic_score']:.2f}
- Skill Match: {cand_b['skill_score']:.2f}
- Years of Exp (YOE): {cand_b['yoe_score'] * 10.0:.1f}
- Avg Tenure: {cand_b['tenure_score'] * 36.0:.1f} months
- Growth Trajectory: {cand_b['growth_score']:.2f}
- Notice Period Fit: {cand_b['notice_score']:.2f}
- Trust Score: {cand_b['trust_score']:.2f}
- Title Match: {cand_b['title_score']:.2f}

Based on these features, select the winner. Reject any candidate with a low trust score immediately.
Respond STRICTLY in JSON format with keys "winner" ("A", "B", or "Tie") and "reasoning" (1-2 sentences of specific technical explanation). Do not add markdown code blocks or backticks.
"""
    try:
        headers = {"Content-Type": "application/json"}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        res = requests.post(url, json=payload, headers=headers, timeout=5)
        if res.status_code == 200:
            text = res.json()['candidates'][0]['content']['parts'][0]['text']
            # Clean up potential markdown formatting in response
            text_cleaned = re.sub(r'```json\s*|\s*```', '', text).strip()
            data = json.loads(text_cleaned)
            return data.get("winner", "Tie"), data.get("reasoning", "LLM preferred comparison.")
    except Exception as e:
        pass
    return None, None

def simulate_recruiter_preference(cand_a, cand_b):
    """Stochastic Bradley-Terry model mimicking recruiter preferences with noise."""
    # Define recruiter utility function
    def get_utility(c):
        if c['trust_score'] < 0.4:
            return 0.0
        
        util = (
            0.30 * c['skill_score'] +
            0.20 * c['title_score'] +
            0.15 * c['semantic_score'] +
            0.15 * c['yoe_score'] +
            0.10 * c['growth_score'] +
            0.05 * c['tenure_score'] +
            0.05 * c['notice_score']
        )
        return util
    
    util_a = get_utility(cand_a)
    util_b = get_utility(cand_b)
    
    # Sigmoid function for probability
    # Scale parameter determines the amount of stochastic noise
    scale = 0.12
    prob_a_preferred = 1.0 / (1.0 + np.exp(-(util_a - util_b) / scale))
    
    # Draw label
    label = 1 if np.random.rand() < prob_a_preferred else 0
    winner = "A" if label == 1 else "B"
    
    reason_parts = []
    if cand_a['trust_score'] != cand_b['trust_score']:
        w = "A" if cand_a['trust_score'] > cand_b['trust_score'] else "B"
        reason_parts.append(f"Candidate {w} holds higher profile trust.")
    if abs(cand_a['skill_score'] - cand_b['skill_score']) > 0.15:
        w = "A" if cand_a['skill_score'] > cand_b['skill_score'] else "B"
        reason_parts.append(f"Candidate {w} has stronger target skill coverage.")
    if cand_a['title_score'] != cand_b['title_score']:
        w = "A" if cand_a['title_score'] > cand_b['title_score'] else "B"
        reason_parts.append(f"Candidate {w} aligns better with the target engineering title.")
        
    reasoning = " ".join(reason_parts) if reason_parts else "Stochastic utility comparison of skills, experience, and timeline."
    return winner, reasoning

def main():
    if not FEATURES_CSV.exists():
        print(f"[ERROR] Features dataset not found at {FEATURES_CSV}. Run precompute.py first.")
        return

    print("Loading features dataset...")
    df = pd.read_csv(FEATURES_CSV)
    
    np.random.seed(42)
    num_pairs = 1500
    pairs_data = []
    
    cand_list = df.to_dict(orient='records')
    cand_ids = [c['candidate_id'] for c in cand_list]
    cand_map = {c['candidate_id']: c for c in cand_list}
    
    print(f"Generating {num_pairs} candidate comparison pairs...")
    
    # We will call Gemini for the first 25 pairs to bootstrap/validate LLM labeling, and use simulation for the rest
    llm_limit = 25
    llm_calls = 0
    consecutive_failures = 0
    disable_llm = False
    
    for i in range(num_pairs):
        id_a, id_b = np.random.choice(cand_ids, size=2, replace=False)
        cand_a = cand_map[id_a]
        cand_b = cand_map[id_b]
        
        winner = None
        reasoning = None
        
        if llm_calls < llm_limit and not disable_llm:
            print(f"[{llm_calls+1}/{llm_limit}] Querying gemini-2.5-flash for comparison between {id_a} and {id_b}...")
            winner, reasoning = call_gemini_comparator(cand_a, cand_b)
            if winner:
                llm_calls += 1
                consecutive_failures = 0
                print(f" -> Winner: {winner}. Reasoning: {reasoning}")
            else:
                consecutive_failures += 1
                print(" -> Gemini API call failed. Falling back to stochastic recruiter model.")
                if consecutive_failures >= 3:
                    disable_llm = True
                    print("[WARN] Gemini API failed 3 times consecutively. Disabling LLM calls to prevent rate-limit delays.")
        
        # Fallback to simulation if Gemini fails or limit is reached
        if not winner:
            w, r = simulate_recruiter_preference(cand_a, cand_b)
            winner = w
            reasoning = r
            
        preference = 1 if winner == "A" else (0 if winner == "B" else 0.5)
        
        row = {
            'cand_a_id': id_a,
            'cand_b_id': id_b,
            'winner': winner,
            'preference': preference,
            'reasoning': reasoning,
            'diff_semantic': cand_a['semantic_score'] - cand_b['semantic_score'],
            'diff_skill': cand_a['skill_score'] - cand_b['skill_score'],
            'diff_yoe': cand_a['yoe_score'] - cand_b['yoe_score'],
            'diff_title': cand_a['title_score'] - cand_b['title_score'],
            'diff_trust': cand_a['trust_score'] - cand_b['trust_score'],
            'diff_tenure': cand_a['tenure_score'] - cand_b['tenure_score'],
            'diff_growth': cand_a['growth_score'] - cand_b['growth_score'],
            'diff_notice': cand_a['notice_score'] - cand_b['notice_score'],
            'diff_behavioral': cand_a['behavioral_score'] - cand_b['behavioral_score']
        }
        pairs_data.append(row)
        
        if (i+1) % 300 == 0:
            print(f"Generated {i+1}/{num_pairs} comparisons...")
            
    df_pairs = pd.DataFrame(pairs_data)
    df_pairs.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRecruiter preference dataset successfully generated and saved to {OUTPUT_CSV}")
    print(f"Dimensions: {df_pairs.shape}")

if __name__ == "__main__":
    main()
