# FitRank AI — Systematic Failure & False Positive Analysis

This document provides a systematic review of the ranking engine's failure modes, false positive scenarios, and our implemented architectural mitigations. It outlines the limitations of dense retrieval and learn-to-rank systems in sourcing pipelines and how FitRank AI secures model decisions against them.

---

## 1. Identified Failure Modes & Mitigations

### 1.1 Candidate Keyword Stuffing (Semantic Inflation)
*   **The Problem:** Dense retrieval models (such as BGE-M3) encode candidate summaries and project descriptions into high-dimensional semantic spaces. If a candidate (e.g., a Graphic Designer or a Civil Engineer) inserts technical keywords (e.g., *"highly passionate about RAG, LangChain, FAISS, and vector databases"*) into their summary, dense semantic search overweights these claims and ranks them at the top.
*   **Why It Happens:** Semantic models lack structural context; they match tokens and semantic concepts but do not verify whether the candidate has actual professional experience utilizing those capabilities.
*   **Our Mitigation:**
    *   **Title Match Feature:** We extract the candidate's current title and compute a `title_score`. Candidates holding non-technical titles get a `title_score = 0.0`.
    *   **LambdaRank Penalization:** The LTR model is trained on recruiter choices that heavily favor candidates with senior AI/ML titles. Non-technical title-stuffed profiles are automatically demoted.
    *   **RRF Fusion:** The BM25 keyword index tokenizes project descriptions across career history, requiring actual descriptions of work, not just summary claims.

### 1.2 Sparse Profiles & Short Resumes
*   **The Problem:** Candidates with extremely brief resumes or missing sections (e.g. only listing a name, current title, and 2-3 skills without descriptions) produce very sparse feature vectors, leading to low semantic similarity scores.
*   **Why It Happens:** Dense embeddings require rich text context to match concepts. When text is sparse, cosine similarity drops significantly.
*   **Our Mitigation:**
    *   **BM25 Keyword Fallback:** BM25 handles exact keyword matching. Even if a candidate has a short resume, if they explicitly list the exact required skills (e.g. *"Python, FAISS, Qdrant"*), BM25 ranks them highly, boosting their RRF score.
    *   **Behavioral Score Integration:** We integrate platform-level signals (`behavioral_score`) such as signup completeness and responsiveness to ensure sparse but active/responsive candidates are not entirely buried.

### 1.3 Timeline & Chronological Discrepancies
*   **The Problem:** Candidates reporting overlapping job tenures, incorrect career start/end dates, or inflated durations with specific skills (e.g., claiming 15 years of Python experience with only 5 years of total professional YOE).
*   **Why It Happens:** Machine learning ranking models assume input features are valid. They cannot detect chronological contradictions on their own.
*   **Our Mitigation:**
    *   **Trust Engine Audit:** Programmatically checks employment history timelines:
        *   Blocks candidates whose career histories overlap chronologically.
        *   Blocks candidates whose skill durations exceed their total YOE + 5 years.
        *   Blocks candidates claiming to work at companies before those companies were founded (e.g., Krutrim before 2023).
    *   **Trust Score Block:** If a trust violation is detected, `trust_score` is set to `0.0`. The candidate's final ranking utility drops to zero, and they are completely excluded from the recruiter's view.

---

## 2. False Positive Analysis

A **False Positive** in FitRank AI is defined as a candidate who is ranked in the top shortlist (Top 100) but is actually unqualified or unsuitable for the Senior AI Engineer role.

### 2.1 The "Chronologically Inflated SDE"
*   **Scenario:** A software engineer with 15 years of generic SDE experience (e.g. database administration, web development) who recently spent 3 months on a basic LangChain tutorial. 
*   **The Risk:** The candidate meets the YOE range (5-9+ years) and has keyword matches. Naive models might rank them as a "Senior AI specialist."
*   **Mitigation:** 
    *   **Growth Trajectory Feature:** We calculate a `growth_score` tracking title regression. If the candidate went from a tech lead role back to a junior SDE role, or if their career lacks SDE -> Senior AI/ML progression, they are penalized.
    *   **Target Skill Duration Filter:** We compute skill experience duration. A candidate must have at least 12+ months of direct experience with embeddings or vector search to get a high `skill_score`.

### 2.2 Relocation & Notice Period Mismatches
*   **Scenario:** A candidate with excellent technical credentials who is located outside of India (with no relocation willingness) and has a notice period exceeding 90 days.
*   **The Risk:** While technically qualified, they cannot fill the founding team role immediately.
*   **Mitigation:**
    *   **Location & Notice Score:** We incorporate `location_score` and `notice_score` directly into the LTR feature matrix. Recruiter preferences trained the model to penalize candidates who are geographically unavailable or have long notice periods, moving them down the shortlist in favor of local, immediate joiners.
