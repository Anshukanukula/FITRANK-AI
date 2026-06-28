# FitRank AI — ML Rigor & Evaluation Report

This report documents the quantitative validation, data leakage audit, target-feature circularity analysis, and system limitations of the FitRank AI ranking models for the **Redrob India Runs Data & AI Challenge**.

---

## 1. Data Leakage Audit (Split Validation)
To verify the statistical validity of our evaluation, we conducted a row-level audit of the training and testing candidate splits used in the ranking evaluation pipeline.

*   **Total Corpus Size:** 100,000 candidate profiles.
*   **Training Set (80%):** 80,000 unique candidates.
*   **Test Set (20%):** 20,000 unique candidates.

### Split Intersection Verification
The splits are constructed programmatically using the candidate's unique UUID (`candidate_id`) as the partition key. To ensure no record leakage, we checked the intersection between the training set candidate IDs ($T_{\text{train}}$) and test set candidate IDs ($T_{\text{test}}$):

$$T_{\text{train}} \cap T_{\text{test}} = \emptyset$$

```text
========================================
SPLIT VERIFICATION LOG:
========================================
Train Set Unique Records: 80,000
Test Set Unique Records: 20,000
Overlapping Candidate IDs: 0 (0.00%)
Status: COMPLETE SPLIT ISOLATION VERIFIED
========================================
```
There is **zero row-level data leakage** or record overlap between the training set and the test set.

---

## 2. Target-Feature Circularity (NDCG = 1.0000 Analysis)
In our ablation study, the LightGBM Learning-to-Rank (LTR) model achieved a perfect **NDCG@10 = 1.0000** and **NDCG@50 = 0.9951** on the unseen test set. 

### Why is the NDCG Score Perfect?
This perfect score is caused by **target-feature circularity** in our synthetic ground truth labelling, not data leakage:
1.  **Rubric-Based Gold Labels:** In the absence of real-world recruiter interaction data (like click-through logs, shortlist acceptances, or hiring outcomes), we constructed an recruiter-inspired **rubric-generated relevance labels** ($0.0$ to $1.0$) using objective feature parameters (e.g., target skill coverage, current title keywords, startup exposure, notice period).
2.  **Deterministic Mapping:** This rubric is a deterministic piecewise step function of the candidate features.
3.  **LTR Learning Capacity:** Because the LightGBM LambdaMART regressor is trained on the exact same feature columns (e.g., `skill_score`, `yoe_score`, `title_score`) that calculate the target label, the gradient boosted trees easily converge to the exact step-function boundaries. 
4.  **Ideal Sorting:** Because the model predicts `gold_score` with near-zero root-mean-squared error, sorting candidates by the predicted LTR score yields the exact same ordering as sorting by the ground truth relevance grades. This produces a perfect NDCG.

### Why LTR is Still Superior to a Formula
Although the LTR model is learning a rubric, using LightGBM is mathematically superior to using a static linear weighted formula in production:
*   **Non-Linear Feature Interaction:** GBDTs can learn conditional thresholds. For example, the model can learn: *"If years of experience is under 5 years, required skill match must be 100% to rank; if experience is over 8 years, a 70% skill match is acceptable."*
*   **Pluggable Target Variable:** In a production deployment, the target variable `gold_score` is simply swapped for a stochastic variable (e.g., `hired_flag` or `interviewed_flag`). The LightGBM pipeline remains identical, but it will learn to rank candidates based on actual recruiter behavior instead of the hand-crafted rubric.

---

## 3. System Limitations & Future Roadmap

During the development of FitRank AI, we identified the following limitations and established their corresponding mitigation strategies:

### 1. Synthetic Gold Labels
*   **Limitation:** Ground-truth labels are defined by a set of heuristic rules rather than real recruiter feedback.
*   **Mitigation:** Swap target labels with recruiter click logs, candidate save rates, and offer acceptance data to transition to feedback-driven ranking.

### 2. Static Skill Synonyms
*   **Limitation:** Skill mapping is constrained by a pre-compiled taxonomy of synonyms (`skill_taxonomy.json`).
*   **Mitigation:** Integrate a dynamic LLM-based taxonomy generator that expands synonyms on-the-fly based on industry trends (e.g., automatically grouping 'Dense Retrieval' under 'Vector Search').

### 3. Isolated Candidate Evaluation
*   **Limitation:** Candidates are evaluated and ranked in isolation, ignoring cohort-level relationships.
*   **Mitigation:** Implement Graph Neural Networks (GNNs) mapping candidates, skills, and companies to discover latent talent density relationships.
