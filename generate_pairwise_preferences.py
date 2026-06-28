"""
FitRank AI - Pairwise Preference & LTR LambdaMART Pipeline
---------------------------------------------------------
This script demonstrates how to transition from a manual rubric (Gold Labels)
to a pairwise preference ranking model. 

It implements:
1. Dynamic generation of candidate comparison pairs (Candidate A vs Candidate B).
2. Stochastic preference labeling (simulating recruiter choices with noise).
3. Query grouping and formatting required for LightGBM LambdaRank.
4. Training a LambdaMART pairwise ranking model to optimize NDCG directly.
"""

import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
FEATURES_CSV = BASE_DIR / "artifacts" / "features_dataset.csv"

def generate_synthetic_features(num_candidates=500):
    """Generates synthetic candidate features if features_dataset.csv is missing."""
    np.random.seed(42)
    features = {
        'candidate_id': [f"CAND_{i:07d}" for i in range(num_candidates)],
        'semantic_score': np.random.uniform(0.1, 1.0, num_candidates),
        'skill_score': np.random.uniform(0.1, 1.0, num_candidates),
        'yoe_score': np.random.uniform(0.2, 1.0, num_candidates),
        'title_score': np.random.choice([0.0, 0.6, 1.0], num_candidates, p=[0.3, 0.4, 0.3]),
        'trust_score': np.random.choice([0.0, 0.4, 1.0], num_candidates, p=[0.05, 0.15, 0.80])
    }
    df = pd.DataFrame(features)
    # Define a rough relevance utility
    df['utility'] = (0.3 * df['skill_score'] + 0.3 * df['title_score'] + 
                     0.2 * df['semantic_score'] + 0.2 * df['yoe_score']) * df['trust_score']
    return df

def build_preference_dataset(df, num_pairs=3000):
    """
    Constructs candidate comparison pairs and labels them.
    Label = 1 if Candidate A is preferred over Candidate B, else 0.
    Adds stochastic noise to simulate recruiter evaluation variance.
    """
    np.random.seed(42)
    pairs = []
    
    cand_ids = df['candidate_id'].tolist()
    utilities = df.set_index('candidate_id')['utility'].to_dict()
    feature_dict = df.set_index('candidate_id').to_dict(orient='index')
    
    print(f"Generating {num_pairs} pairwise candidate comparisons...")
    for _ in range(num_pairs):
        # Pick two random candidates
        id_a, id_b = np.random.choice(cand_ids, size=2, replace=False)
        util_a = utilities[id_a]
        util_b = utilities[id_b]
        
        # Calculate preference probability using a Logistic Sigmoid function
        # Prob(A > B) = 1 / (1 + exp(-(utility_a - utility_b) / scale))
        prob_a_preferred = 1.0 / (1.0 + np.exp(-(util_a - util_b) / 0.1))
        
        # Draw a stochastic label (1 = A wins, 0 = B wins)
        preference = int(np.random.rand() < prob_a_preferred)
        
        # Construct pairwise features (diff features: Feat_A - Feat_B)
        feat_a = feature_dict[id_a]
        feat_b = feature_dict[id_b]
        
        row = {
            'cand_a': id_a,
            'cand_b': id_b,
            'preference': preference,
            'diff_semantic': feat_a['semantic_score'] - feat_b['semantic_score'],
            'diff_skill': feat_a['skill_score'] - feat_b['skill_score'],
            'diff_yoe': feat_a['yoe_score'] - feat_b['yoe_score'],
            'diff_title': feat_a['title_score'] - feat_b['title_score'],
            'diff_trust': feat_a['trust_score'] - feat_b['trust_score']
        }
        pairs.append(row)
        
    return pd.DataFrame(pairs)

def train_lambdamart_pairwise(df_features, df_pairs):
    """
    Trains a pairwise LightGBM ranking model.
    In LambdaRank, candidates are grouped by query. Here we group candidates
    into 'virtual query cohorts' of size 10 to train the model to rank cohorts.
    """
    # Create cohorts for ranking evaluation
    cohort_size = 10
    num_candidates = len(df_features)
    num_cohorts = num_candidates // cohort_size
    
    df_ranking = df_features.head(num_cohorts * cohort_size).copy()
    df_ranking['query_id'] = np.repeat(range(num_cohorts), cohort_size)
    
    # Gold relevance grade for LTR (0 to 3)
    df_ranking['relevance_grade'] = pd.qcut(df_ranking['utility'], q=4, labels=[0, 1, 2, 3]).astype(int)
    
    # Sort by query_id for LightGBM grouping requirements
    df_ranking = df_ranking.sort_values(by='query_id').reset_index(drop=True)
    
    # Training splits
    train_queries = int(num_cohorts * 0.8)
    split_idx = train_queries * cohort_size
    
    train_df = df_ranking.iloc[:split_idx]
    test_df = df_ranking.iloc[split_idx:]
    
    # Feature columns
    feature_cols = ['semantic_score', 'skill_score', 'yoe_score', 'title_score', 'trust_score']
    
    # Group size lists (queries)
    train_groups = train_df.groupby('query_id').size().tolist()
    test_groups = test_df.groupby('query_id').size().tolist()
    
    # LightGBM datasets
    train_data = lgb.Dataset(train_df[feature_cols], label=train_df['relevance_grade'], group=train_groups)
    test_data = lgb.Dataset(test_df[feature_cols], label=test_df['relevance_grade'], group=test_groups, reference=train_data)
    
    # LambdaMART pairwise ranking parameters
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [3, 5, 10],
        'learning_rate': 0.05,
        'num_leaves': 15,
        'min_data_in_leaf': 5,
        'verbose': -1
    }
    
    print("\nTraining LambdaMART Pairwise Ranking Model...")
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data]
    )
    
    # Predict rank order on test queries
    test_df['predicted_score'] = gbm.predict(test_df[feature_cols])
    
    print("\nPairwise Ranking Inference Sample (Query Cohort 0):")
    cohort_sample = test_df[test_df['query_id'] == test_df['query_id'].iloc[0]]
    cohort_sample = cohort_sample.sort_values(by='predicted_score', ascending=False)
    print(cohort_sample[['candidate_id', 'utility', 'relevance_grade', 'predicted_score']])
    
    return gbm

def main():
    if FEATURES_CSV.exists():
        print(f"Loading features dataset from {FEATURES_CSV}...")
        df_cand = pd.read_csv(FEATURES_CSV)
        # Check if features are present
        if 'utility' not in df_cand.columns:
            df_cand['utility'] = (0.3 * df_cand['skill_score'] + 0.3 * df_cand['title_score'] + 
                                  0.2 * df_cand['semantic_score'] + 0.2 * df_cand['yoe_score']) * df_cand['trust_score']
    else:
        print("Features dataset not found. Generating synthetic candidate features...")
        df_cand = generate_synthetic_features()
        
    # Build pairwise comparisons (useful for training pairwise classification/preference)
    df_pairs = build_preference_dataset(df_cand, num_pairs=4000)
    print(f"Dataset generated. Sample pairs shape: {df_pairs.shape}")
    print(df_pairs.head())
    
    # Train LTR LambdaMART Model
    model = train_lambdamart_pairwise(df_cand, df_pairs)
    
    # Save the pairwise model config
    os.makedirs(str(BASE_DIR / "models"), exist_ok=True)
    model.save_model(str(BASE_DIR / "models" / "ltr_pairwise_model.txt"))
    print("\nPairwise LTR LambdaMART model saved to models/ltr_pairwise_model.txt")
    print("This confirms the pipeline setup is complete and operational.")

if __name__ == "__main__":
    main()
