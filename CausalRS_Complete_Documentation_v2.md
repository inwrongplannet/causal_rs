# CAUSAL-INFORMED RECOMMENDATION SYSTEM
## FOR MITIGATING ECHO CHAMBERS

> **Complete Technical Project Documentation**
> Powered by **DoWhy v0.14** · **PyTorch PPO** · **MIND Dataset**
> Causal Discovery → Effect Estimation → Counterfactual Simulation → RL Policy

| Project Type | Domain | Stack | Version |

| Research + Engineering | Responsible AI / RecSys | Python · PyTorch · DoWhy | v2.0 · April 2026 |

**Author:** Abhishek M G · Acharya Institute of Technology, Bengaluru · B.E. Computer Science
**GitHub:** Abhishek-M-29 · abhishekmurali2006@gmail.com

---

## Table of Contents

1. Executive Summary 3

2. Problem Statement & Motivation 4

3. Literature Review & Research Gaps 5

4. DoWhy v0.14 — Technology Deep Dive 6

5. System Architecture: The DoWhy-RL Loop 8

6. Phase 1: Data Pipeline & Feature Engineering 10

7. Phase 2: Causal Modeling with DoWhy 13

8. Phase 3: Counterfactual Engine (DoWhy GCM) 17

9. Phase 4: Reinforcement Learning Agent (PPO) 20

10. Phase 5: Evaluation Framework 24

11. Five Critical Iteration Cycles 27

12. Repository Structure & Setup 34

13. Installation Guide 36

14. Project Timeline (20 Weeks) 37

15. Risks, Mitigations & Ethical Considerations 39

16. Conclusion & References 41


---


## 1. Executive Summary
This document is the complete technical blueprint for a Causal-Informed Recommendation System (CIRS) that mitigates algorithmic echo chambers in news consumption. The project reframes the recommendation problem from correlation-based prediction to causal intervention: instead of asking 'which article will this user click?', the system asks 'which article will cause the most beneficial long-term shift in this user's information diversity?'

The system is built on a three-pillar architecture:

-   DoWhy v0.14 — Microsoft's end-to-end causal inference library. Handles causal graph modeling, automatic effect identification (backdoor/frontdoor criteria), propensity-weighted estimation, and mandatory refutation testing. Replaces all custom causal machinery from the original plan.

-   DoWhy GCM (Graphical Causal Models) — The counterfactual simulation engine. Fits a Structural Causal Model to data, then answers 'what-if' queries: What would a user's diversity score be if we showed article X instead of Y?

-   PyTorch PPO (Proximal Policy Optimization) — The decision-making RL agent. Trained offline via replay on MIND interaction logs. Optimizes a composite reward that weights immediate click probability against the DoWhy-derived causal diversity impact.
> **✅ TIP:** The key innovation over prior work: DoWhy's refutation step is mandatory between every iteration cycle. A causal estimate that fails a placebo or bootstrap test cannot be used as a reward signal — the cycle must be fixed before proceeding. This enforces epistemic discipline that most RL-for-RecSys papers skip.
#### Key Outcomes Targeted
  **Metric**                          **Target vs CF Baseline**

  NDCG@10 (ranking accuracy)          < 5% degradation

  Precision@10 (relevance)            < 8% degradation

  Intra-List Diversity (ILD)          +0.15 absolute improvement

  Homogeneity Score Trend             Downward trend (CF = flat/upward)

  Placebo Refutation (CDI ≈ 0)        Pass (DoWhy mandatory check)

  Bootstrap CI excludes zero          Pass at 95% confidence

  GCM Counterfactual MAE              < 0.08 on KuaiRand-Pure holdout

---


## 2. Problem Statement & Motivation
### 2.1 The Echo Chamber Mechanism
Modern recommender systems (RS) are trained to maximize engagement — clicks, dwell time, completion rate. They are exceptionally good at this. But 'maximize engagement' is not the same as 'serve the user well.' The mechanism that creates echo chambers is precisely this optimization pressure:

1.  System observes: user clicks political article A.

2.  System infers: user prefers political content.

3.  System recommends: more articles similar to A.

4.  User clicks again (because it's available and relevant).

5.  System reinforces: political content is right. Narrows further.

After N iterations, the user's information horizon has collapsed. They have not 'chosen' this narrow diet — they have been funneled into it by an optimization process that conflates availability with preference.

### 2.2 The Technical Root Cause: Correlation is Not Causation
The root flaw is that standard RS operate on association: P(click | user, item). They cannot answer causal questions:

-   Does showing this article cause interest expansion, or merely confirm existing bias?

-   Is the user's engagement with political content a genuine preference, or a result of the system's prior choices (exposure bias)?

-   Would this user have engaged with science articles if they had been shown them with equal prominence?
> **⚠️ WARNING:** A confounding variable Z (e.g., socioeconomic context, time of day, social circle) influences both what the system recommends and what the user clicks. Without modeling Z causally, no correlational model can separate user preference from system-induced behavior.
### 2.3 The Matthew Effect & Filter Bubble Entrenchment
Exposure bias creates a 'rich get richer' dynamic: items that are shown more are clicked more, which causes them to be shown more. Diverse or niche content is systematically suppressed — not because users dislike it, but because it was never given a fair chance. This is the Matthew Effect in recommender systems, and it is a direct consequence of training on biased observational data without causal correction.

### 2.4 Research Question
  **RQ**     Which news article recommendation, at each time step, will cause the maximum positive causal shift in a user's long-term Interest Diversity Score, while preserving sufficient relevance to sustain engagement?


### 2.5 Scope & Constraints
-   Domain: English-language news recommendation (MIND dataset).

-   Setting: Offline simulation using historical interaction logs (no live A/B test).

-   Ethical framing: Recommendations are nudges, not mandates. Relevance is a hard constraint, not a soft preference.

-   Non-goal: This project does not claim to eliminate echo chambers — it demonstrates a causal framework for actively reducing them over time.


---


## 3. Literature Review & Research Gaps
  **Paper**                                      **Key Contribution**                                                              **Gap**                                                                           **Our Advance**

  Zhang et al. 2022 Dynamic Causal CF            Temporal unfolding of causal DAG; back-door adjustment for unbiased preferences   Heuristic intervention (replace item with least-similar). No RL.                  Replace heuristic with DoWhy + PPO optimal policy

  Mureddu et al. 2022 People Recommenders        Simulation evidence linking RS to polarization; PROD framework                    People (social graph) recommenders only — not content RS                        Apply causal validation to content RS on MIND

  Chen et al. 2021 MAC R                         Counterfactual debiasing of item popularity; static correction at inference       Static single-step bias only; no longitudinal diversity modeling                  Generalize to dynamic sequence planning via GCM

  Woo et al. 2025 LAAC (LLM+RL)                  LLM reference policy guides RL exploration toward diverse items                   Not causally grounded — diversity is a metric target, not a modeled mechanism   Causal diversity signal from DoWhy drives RL reward

  Cavenaghi et al. 2024 Causal Discovery in RS   Hybrid HC+BIC causal discovery on KuaiRand; domain-tiered DAG                     Discovery only — graph is the output, not used for intervention                 Use DoWhy to operationalize discovery into estimation + refutation + RL

  Sharma & Kiciman 2020 DoWhy Paper              Four-step causal API with mandatory refutation; first-class assumptions           General library — not applied to RecSys diversity problem                       First application of full DoWhy pipeline to echo chamber mitigation

> **📌 NOTE:** Research Gap This Project Fills: No prior work combines (1) DoWhy's four-step validated causal pipeline, (2) GCM counterfactual simulation, and (3) PPO-based RL policy optimization into a single echo-chamber mitigation system for news recommendation.
---


## 4. DoWhy v0.14 — Technology Deep Dive
DoWhy is a Python library developed by Microsoft Research that enforces rigorous causal thinking through a four-step API. Unlike libraries that provide causal estimators as interchangeable modules, DoWhy makes causal assumptions explicit, testable, and auditable. This is precisely what the project needs: a framework that will refuse to give you a causal estimate if your graph is misspecified.

### 4.1 The Four-Step API
  **Step**   **Method**          **What It Does**                                                                                                                                                                         **In This Project**

  **1**      CausalModel()       Declares the causal graph (DAG), treatment variable, outcome variable, and common causes. Assumptions are first-class — you must state them explicitly.                                Tiered DAG: U→A→Y; I→A; I→Y; Z (latent confounder)

  **2**      identify_effect()   Automatically checks whether the causal effect is identifiable from observational data using backdoor, frontdoor, or instrumental variable criteria. Fails loudly if not.                Identify E\[Y_diversity | do(A=i')\] via backdoor through U

  **3**      estimate_effect()   Plugs the identified estimand into a statistical estimator. Supports IPW, linear regression, propensity stratification, DML (via EconML), CausalML, and more.                            Primary: propensity_score_weighting. Secondary: linear_regression (cross-check)

  **4**      refute_estimate()   Runs robustness checks: placebo treatment (should return \~0), data subset test, bootstrap confidence interval, add unobserved confounder (E-value). Mandatory — skip at your peril.   All four refuters run per iteration cycle. Fail = block cycle progression.


### 4.2 DoWhy GCM (Graphical Causal Models)
Beyond the four-step API, DoWhy v0.11+ includes a GCM module that allows fitting full Structural Causal Models with explicit functional mechanisms (linear, nonlinear, ML-based). The GCM enables:

-   counterfactual_samples(): Given a user state, what would their diversity score be under a different intervention? This is the core of the Counterfactual Engine.

-   anomaly_scores(): Detect when a user's current state is an outlier relative to the population — useful for echo chamber detection.

-   auto.assign_causal_mechanisms(): Automatically fits appropriate mechanisms (linear, AdaBoost, etc.) per node using cross-validation.

### 4.3 DoWhy vs Manual IPW — Why This Matters
  **Concern**                     **Manual IPW**                **DoWhy IPW**

  Assumption transparency         Implicit in code              Explicit in graph declaration

  Identifiability check           Developer's responsibility   Automatic — fails loudly

  Refutation                      Often skipped                 Built-in; enforced in our workflow

  Positivity violations           Silent failure                Detected via propensity overlap checks

  Estimator swapping              Rewrite code                  Change method_name string

  EconML / CausalML integration   Manual glue code              Native support via estimate_effect()


---


## 5. System Architecture: The DoWhy-RL Loop
The system is a closed feedback loop in which every recommendation is backed by a validated causal estimate and informs RL policy updates. The loop has five stages, three of which are new relative to a standard collaborative filtering pipeline.

### 5.1 Architecture Overview
  **\#**   **Stage**                  **Core Logic**                                                                                                                                                                              **Output**

  **1**    Data Pipeline              Parse MIND logs → extract (U, I, A, Y) tensors. Compute Interest Diversity Score D = 1 − cosine(v_user_history, v_item). Build propensity training dataset.                                 Structured DataFrame with all SCM variables

  **2**    DoWhy: Model + Identify    Declare tiered causal DAG in GML. Instantiate CausalModel. Call identify_effect() — verifies backdoor criterion holds through U. Graph validated with independence tests.                 Verified causal estimand

  **3**    DoWhy: Estimate + Refute   estimate_effect() with propensity_score_weighting → Causal Diversity Impact (CDI) per item. refute_estimate() with placebo + bootstrap + sensitivity. Fit GCM for counterfactual queries.   CDI signal + GCM simulator

  **4**    RL Agent (PPO)             State s_t = \[user_history_embedding ∥ D_t\]. Action a_t = select article from candidate pool. Reward = w·r_click + (1−w)·CDI. Policy gradient via PPO. Offline training on MIND replay.    Trained policy π(a|s)

  **5**    Evaluation & Output        Replay evaluation: NDCG@10, Precision@10, ILD, Homogeneity Reduction. CLI demo: Baseline CF vs Causal-RL side-by-side with explanation of diversity shift.                                  Dual-metric report + demo


### 5.2 Information Flow Between Stages
The following shows exactly what data object flows between each stage boundary. This is the contract between modules — each file in the repo must produce these exact objects.

> Stage 1 → Stage 2: Pandas DataFrame df with columns:
>
> user_id, item_id, A (treatment binary), Y_click (0/1),
>
> Y_diversity (float), U_history_emb (768-dim), U_dwell_mean,
>
> I_category (str), I_sentiment (float), I_entity_emb (100-dim)
>
> Stage 2 → Stage 3: dowhy.CausalModel object + IdentifiedEstimand
>
> Stage 3 → Stage 4: CDI_lookup dict {(user_id, item_id) -> float}
>
> gcm.StructuralCausalModel fitted object
>
> Stage 4 → Stage 5: Trained PPO policy checkpoint (.pt)
>
> Replay session records with recommended lists
>
> Stage 5 → Output: metrics_report.json + CLI demo

### 5.3 Causal Graph — The Tiered DAG
The DAG is the foundation of the entire system. It must be specified before any code runs. Edges encode directional causal claims; the absence of an edge is an equally important claim (no direct causal path).

> causal_graph_gml = \"\"\"
>
> graph \[
>
> node \[id \"U_history_emb\" label \"U_history_emb\"\] \# Tier 1: Static user state
>
> node \[id \"U_dwell_mean\" label \"U_dwell_mean\"\] \# Tier 1
>
> node \[id \"I_category\" label \"I_category\"\] \# Tier 2: Item features
>
> node \[id \"I_entity_emb\" label \"I_entity_emb\"\] \# Tier 2
>
> node \[id \"A\" label \"A\"\] \# Tier 3: Treatment (recommendation)
>
> node \[id \"Y_diversity\" label \"Y_diversity\"\] \# Tier 4: Outcome
>
> node \[id \"Y_click\" label \"Y_click\"\] \# Tier 4
>
> edge \[source \"U_history_emb\" target \"A\"\] \# User state influences what legacy RS shows
>
> edge \[source \"U_history_emb\" target \"Y_click\"\] \# User state influences click
>
> edge \[source \"U_history_emb\" target \"Y_diversity\"\]
>
> edge \[source \"U_dwell_mean\" target \"A\"\]
>
> edge \[source \"I_category\" target \"A\"\] \# Item properties influence display
>
> edge \[source \"I_category\" target \"Y_diversity\"\]# Item category directly determines diversity
>
> edge \[source \"I_entity_emb\" target \"Y_diversity\"\]
>
> edge \[source \"A\" target \"Y_diversity\"\]# Treatment -> Outcome (the causal edge of interest)
>
> edge \[source \"A\" target \"Y_click\"\]
>
> \]
>
> \"\"\"
> **📌 NOTE:** Key identifiability: The backdoor path A ← U_history_emb → Y_diversity is blocked by conditioning on U_history_emb. DoWhy's identify_effect() will verify this automatically. If you add an edge that creates an unblockable backdoor path, identify_effect() will raise an IdentificationError — the system's built-in sanity check.
---


## 6. Phase 1: Data Pipeline & Feature Engineering
  **PHASE 1: Data Pipeline & Feature Engineering**

  **Timeline**                                       Weeks 1--3                             pandas · sentence-transformers · NLTK · scikit-learn


### 6.1 Dataset Selection
  **Dataset**          **Scale**                      **Role**              **Justification**

  MIND-small           65K articles 300K users        Dev + Eval            Rich entity embeddings from KG, category labels, and impression logs. Fast iteration cycle. All causal variables extractable directly.

  MIND-large           160K articles 1M users         Full training         Used after causal pipeline validated on MIND-small. Same schema — zero migration cost.

  KuaiRand-Pure        7.5K users 1.2M interactions   Causal ground-truth   Randomized exposure logs: items assigned randomly to users. Allows true ATE estimation without propensity correction — the gold standard for validating our DoWhy estimates.

  Amazon News (2023)   \~500K reviews                 Transfer test         Cross-domain diversity generalization. Diverse topic clusters. Tests robustness of trained policy outside MIND domain.


### 6.2 MIND Dataset Structure
MIND ships with two files. Every column used in this project is documented below.

**6.2.1 news.tsv**

> Columns: NewsID | Category | SubCategory | Title | Abstract | URL | TitleEntities | AbstractEntities
>
> Extraction targets:
>
> I_category <- Category (e.g., 'politics', 'sports', 'finance')
>
> I_subcategory <- SubCategory (fine-grained label)
>
> I_sentiment <- VADER sentiment score on Title (float -1 to +1)
>
> I_entity_emb <- Mean-pool TitleEntities WikiData embeddings (100-dim)
>
> I_title_emb <- SentenceTransformer('all-MiniLM-L6-v2') on Title (384-dim)

**6.2.2 behaviors.tsv**

> Columns: ImpressionID | UserID | Time | History | Impressions
>
> Extraction targets:
>
> user_id <- UserID
>
> U_history_emb <- Mean-pool I_title_emb over all items in History (384-dim)
>
> U_dwell_mean <- Proxy: avg impression count in session (no raw dwell in MIND)
>
> U_click_count <- Count of clicked items in History
>
> A <- Binary: 1 if item shown in impression, 0 for non-shown negatives
>
> Y_click <- Binary: 1 if clicked, 0 if shown but not clicked
>
> Y_diversity <- 1 - cosine_similarity(U_history_emb, I_title_emb) \[0..1\]

### 6.3 Feature Engineering Pipeline
#### 6.3.1 Interest Diversity Score
This is the primary causal outcome variable. It measures how semantically distant the candidate article is from the user's historical consumption pattern.

> \# diversity_scorer.py
>
> import numpy as np
>
> from sklearn.metrics.pairwise import cosine_similarity
>
> def compute_diversity_score(user_history_emb: np.ndarray,
>
> item_title_emb: np.ndarray) -> float:
>
> \"\"\"
>
> Homogeneity(u, i) = cosine_similarity(v_user, v_item)
>
> Diversity(u, i) = 1 - Homogeneity(u, i)
>
> Range: 0 (identical to history) to 1 (maximally diverse)
>
> \"\"\"
>
> u = user_history_emb.reshape(1, -1)
>
> i = item_title_emb.reshape(1, -1)
>
> homogeneity = cosine_similarity(u, i)\[0\]\[0\]
>
> return float(1.0 - homogeneity)
>
> def update_user_history_emb(history_emb: np.ndarray,
>
> new_item_emb: np.ndarray,
>
> alpha: float = 0.1) -> np.ndarray:
>
> \"\"\"Exponential moving average update after a click.\"\"\"
>
> return (1 - alpha) \* history_emb + alpha \* new_item_emb

#### 6.3.2 Propensity Dataset Construction
The propensity model needs to know what the legacy RS would have shown. We approximate the legacy policy by treating each impression log as the 'treatment assignment'. Non-shown items sampled from the article pool serve as controls.

> \# data_prep/build_propensity_df.py
>
> \# For each impression session:
>
> \# Shown items: A=1, Y_click from log, Y_diversity computed
>
> \# Sampled negatives: A=0, Y_click=0 (never shown), Y_diversity computed
>
> \# Ratio: 1 shown : 4 sampled negatives (standard in RS literature)
>
> def build_scm_dataframe(behaviors_df, news_df,
>
> user_emb_map, item_emb_map,
>
> neg_ratio=4) -> pd.DataFrame:
>
> records = \[\]
>
> for \_, row in behaviors_df.iterrows():
>
> u_emb = user_emb_map\[row.user_id\]
>
> for item_id, click in parse_impressions(row.impressions):
>
> i_emb = item_emb_map\[item_id\]
>
> records.append({
>
> 'user_id': row.user_id, 'item_id': item_id,
>
> 'A': 1, 'Y_click': click,
>
> 'Y_diversity': compute_diversity_score(u_emb, i_emb),
>
> 'U_history_emb': u_emb, \# stored as list
>
> 'U_dwell_mean': row.dwell_proxy,
>
> 'I_category': news_df.loc\[item_id, 'category'\],
>
> 'I_sentiment': news_df.loc\[item_id, 'sentiment'\],
>
> })
>
> \# Add negatives \...
>
> return pd.DataFrame(records)
> **⚠️ WARNING:** Dimensionality note: U_history_emb and I_entity_emb are high-dimensional. For DoWhy's tabular API, use PCA(n_components=32) on these embeddings before building the DataFrame. Store the full embeddings separately for the GCM and RL state vector.
### 6.4 Data Splits
-   Train (70%): Users U001--U700K (MIND-large). Causal model fit + RL training.

-   Validation (15%): Users U700K--U850K. RL hyperparameter tuning.

-   Test (15%): Users U850K--U1M. Final evaluation only — never used during training.

-   KuaiRand-Pure: Separate holdout for counterfactual fidelity validation (Iteration Cycle 4).


---


## 7. Phase 2: Causal Modeling with DoWhy
  **PHASE 2: Causal Modeling with DoWhy**

  **Timeline**                              Weeks 4--7                             dowhy v0.14 · networkx · pgmpy


### 7.1 Step 1 — Instantiate CausalModel
This is where all causal assumptions are made explicit. The graph is declared once and reused across all estimation calls. Every edge in the graph is a causal claim that will be tested.

> \# causal_model/dowhy_model.py
>
> import dowhy
>
> from dowhy import CausalModel
>
> import pandas as pd
>
> \# Load the SCM DataFrame from Phase 1
>
> df = pd.read_parquet('data/scm_train.parquet')
>
> \# PCA-reduced user embeddings are columns U_pca_0 \... U_pca_31
>
> common_causes = \[f'U_pca\_{i}' for i in range(32)\] + \['U_dwell_mean', 'I_category', 'I_sentiment'\]
>
> model = CausalModel(
>
> data=df,
>
> treatment='A', \# Binary treatment: was this item shown?
>
> outcome='Y_diversity', \# Causal outcome: diversity score post-recommendation
>
> graph=causal_graph_gml, \# GML string from Section 5.3
>
> common_causes=common_causes,
>
> instruments=None, \# No instrumental variables in this specification
>
> )
>
> \# Visualize for documentation / debugging
>
> model.view_model(layout='dot')

### 7.2 Step 2 — Identify the Causal Effect
identify_effect() implements Pearl's ID algorithm internally. It searches for a valid identification strategy given the graph. If none exists, it raises an IdentificationError — this is intentional and forces you to revise the graph.

> \# Step 2: Identification
>
> estimand = model.identify_effect(
>
> proceed_when_unidentifiable=False \# NEVER set True in production
>
> )
>
> print(estimand)
>
> \# Expected output:
>
> \# Estimand type: nonparametric-ate
>
> \# Estimand: E\[Y_diversity|do(A=1)\] - E\[Y_diversity|do(A=0)\]
>
> \# Backdoor variables: {U_pca_0..31, U_dwell_mean, I_category, I_sentiment}
> **⚠️ WARNING:** If identify_effect() returns an IV estimand or frontdoor estimand instead of backdoor, your graph has a misspecification. Common cause: you forgot to include a common cause of A and Y in the graph, so DoWhy cannot use backdoor and falls back to other strategies.
### 7.3 Step 3 — Estimate the Causal Effect
We use propensity score weighting (IPW) as the primary estimator. The propensity score P(A=1|U) is fit by a logistic regression that predicts which items the legacy RS would have shown to a given user. Inverse weighting re-balances the observational data to simulate an RCT.

> from sklearn.linear_model import LogisticRegression
>
> \# Primary estimator: IPW
>
> estimate_ipw = model.estimate_effect(
>
> estimand,
>
> method_name='backdoor.propensity_score_weighting',
>
> method_params={
>
> 'propensity_score_model': LogisticRegression(max_iter=1000, C=1.0),
>
> 'weighting_scheme': 'ips', \# Inverse Propensity Score
>
> 'min_ps_threshold': 0.01, \# Clip extreme weights
>
> 'max_ps_threshold': 0.99,
>
> },
>
> target_units='ate', \# Average Treatment Effect over all users
>
> )
>
> print(f'ATE (IPW): {estimate_ipw.value:.4f}')
>
> \# Positive ATE → showing an item causally increases diversity
>
> \# Secondary estimator: Linear regression (cross-check)
>
> estimate_lr = model.estimate_effect(
>
> estimand,
>
> method_name='backdoor.linear_regression',
>
> target_units='ate'
>
> )
>
> print(f'ATE (LR): {estimate_lr.value:.4f}')
>
> \# Both should agree in direction; large divergence → model misspecification

#### 7.3.1 Per-Item Causal Diversity Impact (CDI)
The ATE gives a single population-level estimate. For the RL reward signal, we need a per-(user, item) CDI score. We compute this by calling estimate_effect() with target_units='ite' (individual treatment effect) or by querying the fitted propensity model directly.

> \# Generate CDI for each (user, item) pair in the candidate pool
>
> def compute_cdi_batch(model, estimand, df_candidates) -> dict:
>
> cdi_map = {}
>
> estimate = model.estimate_effect(
>
> estimand,
>
> method_name='backdoor.propensity_score_weighting',
>
> target_units='ite', \# Individual Treatment Effects
>
> method_params={'propensity_score_model': LogisticRegression(max_iter=1000)}
>
> )
>
> \# estimate.cate_estimates contains per-unit effects
>
> for idx, row in df_candidates.iterrows():
>
> cdi_map\[(row.user_id, row.item_id)\] = estimate.cate_estimates\[idx\]
>
> return cdi_map

### 7.4 Step 4 — Refute the Estimate (Mandatory)
This step is non-negotiable. Four refutation tests are run. A project iteration that fails any of these tests does NOT proceed to Phase 3 — the graph or estimator must be fixed first.

> \# Refuter 1: Placebo Treatment
>
> \# Replace A with a random binary variable. CDI should collapse to \~0.
>
> refute_placebo = model.refute_estimate(
>
> estimand, estimate_ipw,
>
> method_name='placebo_treatment_refuter',
>
> placebo_type='permute',
>
> num_simulations=100
>
> )
>
> print(refute_placebo) \# Expected: New effect ≈ 0
>
> \# Refuter 2: Bootstrap
>
> \# CDI should be stable across 100 bootstrap samples.
>
> refute_boot = model.refute_estimate(
>
> estimand, estimate_ipw,
>
> method_name='bootstrap_refuter',
>
> num_simulations=100,
>
> sample_size=0.8
>
> )
>
> print(refute_boot) \# Expected: CI excludes 0
>
> \# Refuter 3: Data Subset
>
> \# Effect should hold on 80% of data.
>
> refute_subset = model.refute_estimate(
>
> estimand, estimate_ipw,
>
> method_name='data_subset_refuter',
>
> subset_fraction=0.8, num_simulations=20
>
> )
>
> \# Refuter 4: Add Unobserved Confounder (Sensitivity)
>
> \# How large would an unmeasured confounder need to be to nullify CDI?
>
> refute_confound = model.refute_estimate(
>
> estimand, estimate_ipw,
>
> method_name='add_unobserved_common_cause',
>
> confounders_effect_on_treatment='binary_flip',
>
> confounders_effect_on_outcome='linear',
>
> effect_strength_on_treatment=0.05,
>
> effect_strength_on_outcome=0.05
>
> )
>
> \# E-value should be > 2.0 for acceptable robustness


  **Refuter**         **What It Tests**                              **Pass Criterion**            **Fail Action**

  Placebo Treatment   Is CDI > 0 because of the treatment itself?   Placebo CDI ≈ 0 (p > 0.05)   Graph misspecified — re-check edges

  Bootstrap           Is CDI estimate statistically stable?          95% CI excludes 0             Increase training data or regularize propensity model

  Data Subset         Does CDI hold across data subsets?             Effect direction consistent   Check for data leakage or distribution shift

  Add Confounder      Robustness to unmeasured confounding           E-value > 2.0                Add proxy variables; expand U feature set


---


## 8. Phase 3: Counterfactual Engine (DoWhy GCM)
  **PHASE 3: Counterfactual Engine**

  **Timeline**                         Weeks 8--11                            dowhy.gcm · networkx · sklearn


### 8.1 Why GCM for Counterfactuals
The four-step API (Section 7) gives us the Average Treatment Effect at the population level. The RL agent needs something more precise: given THIS user's current state, what will their diversity score be if I show article X vs article Y? This requires individual-level counterfactual inference, which is the domain of DoWhy's GCM module.

The GCM approach models the data-generating process explicitly: each node's value is a function of its parents plus noise. Once fit, you can clamp any node to a specific value (intervention) and propagate through the graph to get the counterfactual outcome — exactly what the RL agent needs for reward computation during offline training.

### 8.2 Building & Fitting the GCM
> \# counterfactual/gcm_engine.py
>
> import networkx as nx
>
> from dowhy import gcm
>
> from dowhy.gcm import auto
>
> \# Build causal graph as NetworkX DiGraph
>
> causal_graph = nx.DiGraph(\[
>
> ('U_pca_0', 'A'), ('U_dwell_mean', 'A'),
>
> ('I_category', 'A'), ('I_category', 'Y_diversity'),
>
> ('I_sentiment','Y_diversity'),
>
> ('U_pca_0', 'Y_diversity'), ('U_dwell_mean','Y_diversity'),
>
> ('A', 'Y_diversity'),
>
> ('A', 'Y_click'),
>
> ('U_pca_0', 'Y_click'),
>
> \])
>
> \# Instantiate StructuralCausalModel
>
> scm = gcm.StructuralCausalModel(causal_graph)
>
> \# Auto-assign causal mechanisms (fits linear/nonlinear per node)
>
> auto.assign_causal_mechanisms(scm, df_train, override_models=True)
>
> \# This fits: LinearRegression or AdaBoostRegressor per node
>
> \# based on cross-validated R\^2. Inspect assignments:
>
> for node in causal_graph.nodes:
>
> print(node, type(scm.causal_mechanism(node)))
>
> \# Fit all mechanisms on training data
>
> gcm.fit(scm, df_train)
>
> print('GCM fitted successfully.')

### 8.3 Counterfactual Query API
This is the 'What-If' simulator. Given a user's observed features and a proposed new item, it returns the predicted diversity score under the hypothetical intervention.

> def predict_diversity_counterfactual(
>
> scm: gcm.StructuralCausalModel,
>
> user_row: pd.Series, \# Observed user state (U features)
>
> new_item_category: str, \# The candidate article's category
>
> new_item_sentiment: float
>
> ) -> float:
>
> \"\"\"
>
> Returns E\[Y_diversity | do(A=1, I_category=new_item_category),
>
> I_sentiment=new_item_sentiment,
>
> U=user_row\]
>
> \"\"\"
>
> interventions = {
>
> 'A': lambda n: np.ones(n), \# Force: item IS shown
>
> 'I_category': lambda n: np.full(n, category_to_int(new_item_category)),
>
> 'I_sentiment': lambda n: np.full(n, new_item_sentiment),
>
> }
>
> samples = gcm.counterfactual_samples(
>
> scm,
>
> target_node='Y_diversity',
>
> observed_data=pd.DataFrame(\[user_row\]),
>
> interventions=interventions,
>
> num_samples_from_conditional=50
>
> )
>
> return float(samples\['Y_diversity'\].mean())

### 8.4 Batch CDI Computation for RL Training
During RL training, the agent queries CDI for each candidate item before selecting. Pre-computing and caching CDI values for the training set dramatically speeds up the RL loop.

> \# Pre-compute CDI lookup table for training sessions
>
> from tqdm import tqdm
>
> import pickle
>
> cdi_cache = {}
>
> for session in tqdm(train_sessions):
>
> user_row = df_train\[df_train.user_id == session.user_id\].iloc\[0\]
>
> for item_id in session.candidate_pool:
>
> item = news_df.loc\[item_id\]
>
> cdi = predict_diversity_counterfactual(
>
> scm, user_row, item.category, item.sentiment
>
> )
>
> cdi_cache\[(session.user_id, item_id)\] = cdi
>
> with open('data/cdi_cache.pkl', 'wb') as f:
>
> pickle.dump(cdi_cache, f)
>
> print(f'Cached {len(cdi_cache)} CDI values.')
> **📌 NOTE:** Performance note: gcm.counterfactual_samples() can be slow for large candidate pools. Use the pre-computation above for training. At inference time, the trained PPO policy does not call the GCM — it uses the learned policy weights directly. GCM is a training-time tool.
---


## 9. Phase 4: Reinforcement Learning Agent (PPO)
  **PHASE 4: RL Policy Optimization**

  **Timeline**                          Weeks 12--15                           PyTorch · stable-baselines3 · gymnasium


### 9.1 Why PPO Over A2C
The original specification proposed A2C. We upgrade to PPO (Proximal Policy Optimization) for three reasons:

-   Clipped surrogate objective: PPO prevents destructive large policy updates that destabilize training — critical when the reward signal (CDI) is noisy from the GCM.

-   Better sample efficiency: PPO reuses each batch of experience multiple times (multiple epochs per rollout), reducing the number of environment interactions needed.

-   Off-the-shelf implementation: stable-baselines3 provides a production-quality PPO with minimal configuration. We focus engineering effort on the environment, not the optimizer.

### 9.2 MDP Formulation
  **Component**   **Symbol**                    **Definition**

  State Space     S = R\^(384+1)                s_t = concat(U_history_emb_t \[384-dim\], D_t \[scalar\]). Captures both user semantic context and current echo chamber severity.

  Action Space    A = {0..K-1}                  Discrete selection of one article from K=20 candidates in the impression. At inference, agent ranks K candidates; top-N shown to user.

  Reward          r_t = w·r_click + (1-w)·CDI   r_click from MIND log (0/1). CDI from cdi_cache. w=0.6 default (tuned in Cycle 3). Episode reward = sum of discounted r_t over session.

  Transition      P(s\_{t+1} | s_t, a_t)       If user clicks a_t: U_history_emb\_{t+1} = EMA(U_history_emb_t, I_title_emb\_{a_t}, alpha=0.1). D\_{t+1} recomputed. If no click: state unchanged.

  Episode         T = 10 steps                  One news session = 10 impression rounds. Each round: agent selects from K candidates; observe click; update state.

  Discount        gamma = 0.95                  Near-long-horizon: agent is incentivized to plan several steps ahead for diversity benefits that materialize gradually.


### 9.3 Custom Gymnasium Environment
> \# rl_agent/news_env.py
>
> import gymnasium as gym
>
> import numpy as np
>
> class NewsRecommendEnv(gym.Env):
>
> def \_\_init\_\_(self, sessions, news_df, cdi_cache, w=0.6, K=20):
>
> self.sessions = sessions \# List of MIND replay sessions
>
> self.news_df = news_df
>
> self.cdi_cache = cdi_cache
>
> self.w = w
>
> self.K = K \# Candidate pool size
>
> self.T = 10 \# Steps per episode
>
> \# Observation: 384-dim history emb + 1 diversity scalar
>
> self.observation_space = gym.spaces.Box(
>
> low=-1.0, high=1.0, shape=(385,), dtype=np.float32
>
> )
>
> self.action_space = gym.spaces.Discrete(K)
>
> def reset(self, seed=None):
>
> self.session = np.random.choice(self.sessions)
>
> self.step_idx = 0
>
> self.history_emb = self.session.initial_history_emb.copy()
>
> self.D = compute_session_diversity(self.history_emb, self.session)
>
> return self.\_obs(), {}
>
> def step(self, action: int):
>
> candidate = self.session.candidates\[self.step_idx\]\[action\]
>
> r_click = self.session.clicks\[self.step_idx\]\[action\] \# 0 or 1
>
> cdi = self.cdi_cache.get((self.session.user_id, candidate.item_id), 0.0)
>
> reward = self.w \* r_click + (1 - self.w) \* cdi
>
> if r_click: \# Update history only if clicked
>
> self.history_emb = ema_update(self.history_emb, candidate.title_emb)
>
> self.D = 1.0 - cosine_sim(self.history_emb, candidate.title_emb)
>
> self.step_idx += 1
>
> done = (self.step_idx >= self.T)
>
> return self.\_obs(), reward, done, False, {}
>
> def \_obs(self):
>
> return np.append(self.history_emb, self.D).astype(np.float32)

### 9.4 PPO Training
> \# rl_agent/train_ppo.py
>
> from stable_baselines3 import PPO
>
> from stable_baselines3.common.vec_env import SubprocVecEnv
>
> def make_env(sessions, news_df, cdi_cache, w):
>
> return lambda: NewsRecommendEnv(sessions, news_df, cdi_cache, w=w)
>
> \# Parallel environments for faster rollout collection
>
> n_envs = 8
>
> vec_env = SubprocVecEnv(\[make_env(train_sessions, news_df, cdi_cache, w=0.6)
>
> for \_ in range(n_envs)\])
>
> model = PPO(
>
> 'MlpPolicy',
>
> vec_env,
>
> n_steps=512, \# Rollout buffer size per env
>
> batch_size=64, \# Minibatch size for gradient updates
>
> n_epochs=10, \# PPO epochs per rollout
>
> gamma=0.95, \# Discount factor
>
> gae_lambda=0.95, \# GAE lambda for advantage estimation
>
> clip_range=0.2, \# PPO clipping epsilon
>
> ent_coef=0.01, \# Entropy coefficient (exploration)
>
> learning_rate=3e-4,
>
> policy_kwargs=dict(net_arch=\[256, 128\]), \# Actor-Critic shared layers
>
> verbose=1,
>
> tensorboard_log='./tb_logs/'
>
> )
>
> model.learn(total_timesteps=5_000_000, progress_bar=True)
>
> model.save('checkpoints/ppo_causal_rs_w06')

### 9.5 Reward Weight Ablation
Three agents are trained with different w values to establish the Pareto frontier between engagement and diversity:


  **w value**   **System Type**      **Expected Behavior**          **When to Use**

  w = 1.0       Pure CF Baseline     High NDCG, ILD→0               Baseline comparison only

  w = 0.7       Engagement-heavy     High NDCG, moderate ILD gain   Conservative deployment (engagement-sensitive platform)

  w = 0.6       Balanced (default)   Target operating point         Primary evaluation point

  w = 0.4       Diversity-heavy      High ILD, moderate NDCG drop   Responsible AI deployment with explicit diversity mandate

  w = 0.0       Pure Diversity       Max ILD, low NDCG              Ablation only — likely irrelevant content


---


## 10. Phase 5: Evaluation Framework
  **PHASE 5: Evaluation**

  **Timeline**              Weeks 16--18                           numpy · matplotlib · scipy


### 10.1 Offline Evaluation Protocol (Replay Method)
Since a live A/B test is not feasible, we use the Replay Method: the trained RL agent is inserted into historical sessions. At each step, if the agent's top-1 recommendation matches the item the user actually clicked in history, it counts as a Hit. Diversity metrics are computed over the agent's full recommended list regardless of historical match.

> \# eval/replay_evaluator.py
>
> def replay_evaluate(policy, test_sessions, news_df, cdi_cache, K=10):
>
> results = \[\]
>
> for session in test_sessions:
>
> env = NewsRecommendEnv(\[session\], news_df, cdi_cache, w=0.6)
>
> obs, \_ = env.reset()
>
> rec_lists = \[\]
>
> for step in range(env.T):
>
> action, \_ = policy.predict(obs, deterministic=True)
>
> \# Get full ranked list (not just top-1)
>
> q_values = policy.policy.evaluate_actions(
>
> obs\[None\], np.arange(env.K)
>
> )
>
> ranked = np.argsort(q_values)\[::-1\]\[:K\]
>
> rec_lists.append(ranked)
>
> obs, \_, done, \_, \_ = env.step(action)
>
> if done: break
>
> results.append(compute_session_metrics(session, rec_lists, news_df))
>
> return aggregate_metrics(results)

### 10.2 Metrics — Complete Definitions
#### 10.2.1 Accuracy Metrics
> \# NDCG@K — Normalized Discounted Cumulative Gain
>
> def ndcg_at_k(recommended: list, clicked: set, K: int) -> float:
>
> dcg = sum(1/np.log2(i+2) for i,item in enumerate(recommended\[:K\]) if item in clicked)
>
> idcg = sum(1/np.log2(i+2) for i in range(min(len(clicked), K)))
>
> return dcg / idcg if idcg > 0 else 0.0
>
> \# Precision@K
>
> def precision_at_k(recommended: list, clicked: set, K: int) -> float:
>
> return len(set(recommended\[:K\]) & clicked) / K

#### 10.2.2 Diversity Metrics
> \# Intra-List Diversity (ILD) — semantic spread within the recommended list
>
> def ild(recommended_embeddings: np.ndarray) -> float:
>
> N = len(recommended_embeddings)
>
> if N < 2: return 0.0
>
> total = 0.0
>
> for i in range(N):
>
> for j in range(i+1, N):
>
> sim = cosine_similarity(
>
> recommended_embeddings\[i:i+1\],
>
> recommended_embeddings\[j:j+1\]
>
> )\[0\]\[0\]
>
> total += (1 - sim)
>
> return total / (N \* (N-1) / 2)
>
> \# Homogeneity Score Trend — track across session steps
>
> def homogeneity_trend(session_history_embs, recommended_embs) -> list:
>
> return \[cosine_similarity(h\[None\], r\[None\])\[0\]\[0\]
>
> for h, r in zip(session_history_embs, recommended_embs)\]
>
> \# A healthy causal-rl system should show DECREASING trend over steps
>
> \# vs CF baseline which shows FLAT or INCREASING trend

### 10.3 Comparison Matrix
  **System**                 **NDCG@10**   **ILD**        **Homogeneity Δ**     **CDI Refuted?**

  Popularity Baseline        \~0.35        Low            +0.05 (worsening)     N/A

  CF (BM25 + cosine)         \~0.41        Low            \~0.00 (flat)         N/A

  RL w/o DoWhy (heuristic)   \~0.39        Medium         −0.03 (marginal)      Not tested

  DoWhy-RL (w=0.6) TARGET    > 0.39       High (+0.15)   −0.12 (significant)   Yes — all 4 pass


### 10.4 Statistical Significance
All metric comparisons are tested for statistical significance using a paired t-test across users (N > 1000 test users). Threshold: p < 0.01. Effect sizes reported using Cohen's d.

> from scipy import stats
>
> def significance_test(causal_rl_scores, baseline_scores, metric_name):
>
> t_stat, p_val = stats.ttest_rel(causal_rl_scores, baseline_scores)
>
> d = (np.mean(causal_rl_scores) - np.mean(baseline_scores)) / np.std(causal_rl_scores)
>
> print(f'{metric_name}: t={t_stat:.3f}, p={p_val:.4f}, Cohen_d={d:.3f}')
>
> return p_val < 0.01
>

---


## 11. Five Critical Iteration Cycles
The project is structured as five sequential gates. A gate must be passed — all criteria met, all refutation tests green — before work on the next cycle begins. This prevents compounding errors where a broken causal foundation silently corrupts downstream training.
> **📌 NOTE:** Philosophy: In causal inference, a wrong assumption confidently applied is worse than admitting ignorance. The iteration cycles enforce epistemic discipline — every cycle ends with a question: 'Should I trust this result enough to build on it?'
------------------------------------------------------ ---------------- ------------
  **CYCLE 1: Causal Graph Validity & Identifiability**   **Weeks 4--5**   **GATE 1**


#### Objective
Establish a correct, identifiable causal graph before any estimation or training. An unidentifiable graph makes all downstream estimates meaningless. This cycle validates the DAG both theoretically (Pearl's criteria) and empirically (data-driven independence tests).

#### Tasks
6.  Define tiered DAG in GML string (as specified in Section 5.3).

7.  Instantiate CausalModel with MIND-small SCM DataFrame.

8.  Call identify_effect(proceed_when_unidentifiable=False). Verify: backdoor criterion satisfied through U variables.

9.  Run d-separation tests: for each claimed conditional independence in the graph, verify against data using conditional independence tests (Fisher's Z or KCI kernel test via pgmpy).

10. Visualize the causal graph with model.view_model(). Review with domain logic: does every edge make intuitive sense?

11. Document every edge with a written causal claim (e.g., 'U_history_emb → A because the legacy RS personalizes based on history').

#### Diagnostic Criteria
  **Test**                          **Expected Result**                           **Fail Action**

  identify_effect() return type     nonparametric-ate via backdoor                Audit graph for unblocked backdoor paths; add missing common causes

  D-separation: A ⊥ Y | U (data)   p > 0.05 (independent given U)               Edge A→Y is not the sole path; add missing mediator nodes

  Graph acyclicity check            No cycles (DAG constraint)                    Networkx: nx.is_directed_acyclic_graph(g) == True

  Tier ordering respected           No edges from Tier N to Tier M where N > M   Remove temporally impossible edges; re-check domain logic

> **🔴 CRITICAL:** Critical Failure Mode: identify_effect() raises IdentificationError. This means there is an unblocked backdoor path from A to Y that cannot be closed by any conditioning set in the observed data. You must either add the missing confounder to the dataset or use an IV estimand. Do not proceed to Cycle 2 until this resolves.
#### Cycle 1 Pass Gate
-   identify_effect() succeeds with backdoor estimand.

-   All claimed d-separations validated at p > 0.05.

-   Graph acyclicity confirmed. Every edge has a documented causal rationale.


  **CYCLE 2: Causal Estimation Robustness**         **Weeks 6--8**   **GATE 2**


#### Objective
Ensure the Causal Diversity Impact (CDI) estimate is statistically valid, robust to data perturbation, and not an artefact of model choice. A CDI estimate that fails refutation tests will produce a corrupted reward signal that teaches the RL agent the wrong policy.

#### Tasks
12. Run primary IPW estimation. Check: ATE direction is positive (showing items causally increases diversity from baseline).

13. Run secondary linear regression estimation. Direction must agree; magnitude within 30% — else investigate propensity score positivity violation.

14. Run all four refutation tests (Section 7.4). Document results in refutation_report.json.

15. Inspect propensity score distributions: plot P(A=1|U) for treated and control groups. Overlap must be substantial — no extreme weights (all weights < 20 after clipping).

16. Compute per-item CDI for 1000 sample (user, item) pairs. Histogram should show bimodal distribution: low CDI for same-category items, high CDI for cross-category items. This is the sanity check.

17. If bootstrap CI includes 0: increase propensity model complexity (use GradientBoostingClassifier instead of LogisticRegression) and re-run.

#### Quantitative Targets
  **Refutation Test**           **Target**                    **Interpretation**

  Placebo CDI                   ≈ 0 (|val| < 0.01)         True CDI is driven by treatment, not spurious correlation

  Bootstrap 95% CI              Excludes 0                    CDI is statistically distinguishable from zero

  Data Subset                   Effect direction consistent   Result is not driven by outlier users

  E-value (hidden confounder)   > 2.0                        Unmeasured confounder would need to be strong (2x) to nullify CDI

  IPW vs LR ATE agreement       Within 30%                    Both estimators agree on direction and approximate magnitude

> **🔴 CRITICAL:** Critical Failure Mode: Bootstrap CI includes 0. This means the CDI signal is statistically noise — if you feed this into the RL reward, the agent will train on random signal. Fix: (1) Increase propensity model complexity, (2) increase training data, (3) revisit feature selection for U variables.
#### Cycle 2 Pass Gate
-   All four refutation tests pass. CDI > 0 (ATE) with bootstrap CI excluding 0.

-   CDI per-item histogram shows sensible bimodal structure.

-   Propensity score overlap confirmed visually. cdi_cache.pkl generated for all training sessions.


  **CYCLE 3: GCM Counterfactual Fidelity**          **Weeks 9--11**   **GATE 3**


#### Objective
Validate that the fitted GCM predicts individual-level diversity shifts accurately on data with known ground truth. If the GCM is a poor simulator, the RL agent's training-time CDI rewards will be inaccurate, and the policy will learn from a distorted version of reality.

#### Tasks
18. Fit GCM on MIND-small training split using gcm.auto.assign_causal_mechanisms().

19. Run gcm.evaluate_causal_model() — check per-node R² and KS-test for distributional fit.

20. On KuaiRand-Pure holdout: for each randomized impression, compare GCM predicted Y_diversity vs observed Y_diversity. Compute MAE and Pearson r.

21. Validate the 'Bridge' scenario (Use Case 3): Take 100 users with high homogeneity (Crypto readers). Simulate do(A=Fintech article). GCM must predict a measurable homogeneity reduction vs do(A=another Crypto article).

22. If MAE > 0.08: re-run auto.assign_causal_mechanisms() with override_models=True and expanded model list including GradientBoostingRegressor. Re-evaluate.

23. Inspect GCM mechanism assignments: nodes with low R² get manually specified mechanisms (e.g., SCMLinearModel or SCMPolynomialModel).

#### GCM Evaluation Code
> from dowhy.gcm import evaluate_causal_model, EvaluationResult
>
> results = evaluate_causal_model(
>
> scm,
>
> df_kuairand_holdout,
>
> compare_mechanism_baselines=True,
>
> evaluate_invertibility_assumptions=True,
>
> )
>
> print(results.summary())
>
> \# Check: Y_diversity node R\^2 > 0.6; KS p-value > 0.05 per node

#### Cycle 3 Pass Gate
-   GCM Y_diversity node R² > 0.60 on KuaiRand-Pure.

-   MAE < 0.08 between predicted and observed Y_diversity on randomized holdout.

-   Bridge scenario: GCM predicts ≥ 20% homogeneity reduction for cross-category intervention vs same-category baseline.


  **CYCLE 4: RL Policy Calibration & Pareto Analysis**   **Weeks 12--15**   **GATE 4**


#### Objective
Find the optimal composite reward weight w that delivers meaningful diversity gains without catastrophic accuracy degradation. Train and compare all five w-ablation agents. Confirm the trained policy's decisions are interpretable and aligned with causal intuition.

#### Tasks
24. Train five PPO agents (w ∈ {0.0, 0.4, 0.6, 0.7, 1.0}) for 5M timesteps each.

25. Evaluate all five on validation split. Compute NDCG@10 and ILD per agent.

26. Plot Pareto frontier: x-axis = NDCG@10, y-axis = ILD. Identify the 'knee' — the w value that maximizes ILD gain per unit NDCG cost.

27. Inspect 50 sample recommendation sessions from w=0.6 agent. Manually verify: does the agent's first recommendation for a Crypto-heavy user introduce a bridging topic (Fintech/Law) rather than another Crypto article?

28. Check cold-start behavior: for users with < 3 history items, diversity score is undefined. Verify state representation handles this (default to D=0.5 for cold-start users).

29. Monitor training stability: PPO loss curves should converge. Entropy should decrease gradually (not collapse) — indicates policy is learning, not memorizing.

#### Expected Pareto Frontier
  **w**     **NDCG@10 (est.)**   **ILD (est.)**   **Hom. Δ (est.)**   **Verdict**

  1.0       0.41 (CF level)      0.18 (low)       +0.00               Echo chamber

  0.7       0.40                 0.26             −0.04               Safe deploy

  0.6 ★     0.39                 0.33             −0.10               Target point

  0.4       0.37                 0.37             −0.14               Responsible AI

  0.0       0.29 (low)           0.41 (max)       −0.18               Irrelevant

> **🔴 CRITICAL:** Critical Failure Mode: At w=0.6, NDCG drops > 8% below CF baseline. This means the causal diversity signal is too noisy to serve as a useful reward — likely Cycle 2 was passed too leniently. Re-run Cycle 2 with stricter propensity model; regenerate cdi_cache.pkl; retrain.
#### Cycle 4 Pass Gate
-   Five agents trained; Pareto frontier plotted.

-   w=0.6 agent: NDCG within 5% of CF baseline, ILD > +0.12 absolute gain.

-   Manual session inspection confirms 'bridging' recommendation behavior.

-   PPO training curves converge. Policy checkpoint saved.


  **CYCLE 5: End-to-End Ablation, Interpretability & Final Report**   **Weeks 16--18**   **GATE 5**


#### Objective
Confirm the full DoWhy-RL pipeline outperforms all ablated variants on the test split. Generate interpretable explanations for recommendation decisions. Document final E-value for transparency. Produce the CLI demo and final report.

#### Tasks
30. Run all systems on held-out test split (never used before). Collect NDCG@10, Precision@10, ILD, Homogeneity Trend, Homogeneity Δ at session-end.

31. Statistical significance: paired t-test (DoWhy-RL vs CF baseline) for each metric. Report p-value and Cohen's d.

32. Ablation: run RL agent with CDI replaced by a simple ILD-based diversity reward (no DoWhy). Compare vs DoWhy-RL. If DoWhy-RL is significantly better on Homogeneity Δ, this validates the causal advantage over heuristic diversity.

33. Interpretability output: for each recommendation, produce a text explanation: 'Item X recommended because CDI=0.42 > threshold. Current homogeneity: 0.73. Predicted new homogeneity if clicked: 0.61.'

34. Final sensitivity analysis: report E-value from add_unobserved_common_cause. This is the 'honesty number' — published in the report to acknowledge remaining uncertainty.

35. Generate CLI demo output (see Section 12).

**Final Metric Table (Target)**


  **System**                 **NDCG@10**   **P@10**   **ILD**    **Hom. Δ**   **E-val**

  Popularity Baseline        0.35          0.30       0.18       +0.05        N/A

  CF Baseline                0.41          0.36       0.19       0.00         N/A

  RL (heuristic diversity)   0.39          0.34       0.27       −0.04        N/A

  DoWhy-RL w=0.6 ★           > 0.39       > 0.33    > 0.33    < −0.10     > 2.0

> **✅ TIP:** Cycle 5 Pass Gate: DoWhy-RL w=0.6 significantly outperforms CF baseline on ILD (p < 0.01) and Homogeneity Δ (p < 0.01) with NDCG degradation < 5%. All four DoWhy refutation tests pass on final test-set rerun. CLI demo generates interpretable output. E-value > 2.0 documented.
---


## 12. Repository Structure & CLI Demo
### 12.1 Repository Layout
> causal_rs/
>
> ├── README.md
>
> ├── requirements.txt
>
> ├── setup.py
>
> │
>
> ├── data_prep/
>
> │ ├── download_mind.sh \# Download MIND-small + MIND-large
>
> │ ├── parse_mind.py \# Parse news.tsv + behaviors.tsv
>
> │ ├── embed_articles.py \# SentenceTransformer title embeddings
>
> │ ├── build_scm_df.py \# Build (U,I,A,Y) SCM DataFrame
>
> │ ├── diversity_scorer.py \# compute_diversity_score(), ema_update()
>
> │ └── pca_reduce.py \# PCA(32) on user/item embeddings
>
> │
>
> ├── causal_model/
>
> │ ├── dag_definition.py \# causal_graph_gml string (the DAG)
>
> │ ├── dowhy_model.py \# CausalModel instantiation
>
> │ ├── identify_estimate.py \# identify_effect() + estimate_effect()
>
> │ ├── refute.py \# All four refutation tests
>
> │ └── cdi_batch.py \# Per-(user,item) CDI computation
>
> │
>
> ├── counterfactual/
>
> │ ├── gcm_engine.py \# StructuralCausalModel + fit()
>
> │ ├── gcm_query.py \# predict_diversity_counterfactual()
>
> │ ├── gcm_evaluate.py \# evaluate_causal_model() on KuaiRand
>
> │ └── precompute_cdi.py \# Generate cdi_cache.pkl
>
> │
>
> ├── rl_agent/
>
> │ ├── news_env.py \# Custom Gymnasium environment
>
> │ ├── train_ppo.py \# PPO training script (5 w-ablations)
>
> │ ├── evaluate_policy.py \# Replay evaluation + metrics
>
> │ └── checkpoints/ \# Saved PPO models
>
> │
>
> ├── eval/
>
> │ ├── metrics.py \# NDCG, Precision, ILD, Homogeneity
>
> │ ├── pareto_plot.py \# NDCG vs ILD Pareto frontier
>
> │ ├── significance.py \# Paired t-test, Cohen's d
>
> │ └── ablation_report.py \# Full comparison table generator
>
> │
>
> ├── cli.py \# Main demo entry point
>
> │
>
> ├── notebooks/
>
> │ ├── cycle_1_graph_validation.ipynb
>
> │ ├── cycle_2_estimation_refutation.ipynb
>
> │ ├── cycle_3_gcm_fidelity.ipynb
>
> │ ├── cycle_4_rl_pareto.ipynb
>
> │ └── cycle_5_ablation_final.ipynb
>
> │
>
> └── data/ \# gitignored; populated by data_prep/
>
> ├── scm_train.parquet
>
> ├── scm_val.parquet
>
> ├── scm_test.parquet
>
> └── cdi_cache.pkl

### 12.2 CLI Demo
> \# Usage:
>
> python cli.py \--user_id U123456 \--model_checkpoint checkpoints/ppo_causal_rs_w06
>
> \# Sample output:
>
> User U123456 | Current Homogeneity: 0.78 | Diversity Score: 0.22
>
> (User has been in a heavy Cryptocurrency echo chamber for 8 sessions)
>
> === BASELINE CF RECOMMENDATION ===
>
> 1. 'Bitcoin Surpasses \$90K Amid Institutional Buying' \[Crypto\] CDI: 0.04
>
> 2. 'Ethereum Merge One Year On: What Changed?' \[Crypto\] CDI: 0.03
>
> 3. 'Top DeFi Protocols By TVL in 2026' \[Crypto\] CDI: 0.05
>
> === CAUSAL-RL RECOMMENDATION (w=0.6) ===
>
> 1. 'SEC Chair Proposes New Crypto Disclosure Rules' \[Law\] CDI: 0.41
>
> → BRIDGE: Connects Crypto → Regulatory/Policy domain
>
> → Predicted Homogeneity after click: 0.64 (−0.14 reduction)
>
> 2. 'Fed Rate Decision: Impact on Risk Assets' \[Finance\] CDI: 0.35
>
> 3. 'How Fintech Startups Are Navigating Basel IV' \[Finance\] CDI: 0.33
>

---


## 13. Installation Guide
### 13.1 Requirements
  **Package**              **Version**     **Purpose**

  dowhy                    >= 0.14        Core causal inference: CausalModel, GCM, refutation

  torch                    >= 2.2         PPO policy network (Actor-Critic MLP)

  stable-baselines3        >= 2.3         PPO implementation with parallel environments

  gymnasium                >= 0.29        RL environment interface

  sentence-transformers    >= 2.7         all-MiniLM-L6-v2 for title embeddings (384-dim)

  networkx                 >= 3.2         DAG construction for GCM

  pgmpy                    >= 0.1.25      Independence tests for graph validation

  econml                   >= 0.15        Optional: DML estimator for Cycle 2 cross-check

  pandas / numpy / scipy   latest          Data manipulation and statistics

  scikit-learn             >= 1.4         Propensity model (LogisticRegression / GBC)

  matplotlib / seaborn     latest          Pareto plots, causal graph visualization


### 13.2 Setup Commands
> \# 1. Clone repository
>
> git clone https://github.com/Abhishek-M-29/causal_rs.git
>
> cd causal_rs
>
> \# 2. Create virtual environment (Python 3.10+ required)
>
> python -m venv venv
>
> source venv/bin/activate \# Windows: venv\\Scripts\\activate
>
> \# 3. Install dependencies
>
> pip install -r requirements.txt
>
> \# 4. Download MIND-small dataset
>
> bash data_prep/download_mind.sh small
>
> \# Downloads to data/raw/MIND-small/ (\~130MB)
>
> \# 5. Run Phase 1 data pipeline
>
> python data_prep/parse_mind.py \--split small
>
> python data_prep/embed_articles.py \--model all-MiniLM-L6-v2
>
> python data_prep/build_scm_df.py
>
> \# Output: data/scm_train.parquet, data/scm_val.parquet, data/scm_test.parquet
>
> \# 6. Run Jupyter notebooks in order
>
> jupyter lab notebooks/
>
> \# Start with: cycle_1_graph_validation.ipynb
>

---


## 14. Project Timeline — 20 Weeks
  **Weeks**   **Phase**   **Tasks**                                                                                         **Deliverable**                                       **Gate**

  1--2        1           Environment setup; MIND download; literature review finalization                                  Dev environment; annotated bibliography               ---

  3           1           Full data pipeline: parsing, embedding, SCM DataFrame, diversity scorer, PCA reduction            scm_train/val/test.parquet                            ---

  4--5        2           DAG definition; CausalModel; identify_effect(); d-separation tests; graph documentation           dag_definition.py; cycle_1 notebook PASS              GATE 1

  6--8        2           IPW + LR estimation; all 4 refutation tests; propensity overlap analysis; CDI batch computation   cdi_cache.pkl; refutation_report.json; cycle_2 PASS   GATE 2

  9--10       3           GCM fit; auto mechanism assignment; gcm.evaluate_causal_model()                                   Fitted GCM checkpoint; evaluation report              ---

  11          3           KuaiRand counterfactual fidelity test; Bridge scenario validation                                 gcm_fidelity_report.json; cycle_3 PASS                GATE 3

  12--13      4           Gymnasium env; 5 PPO ablation trainings (w values); TensorBoard monitoring                        5 policy checkpoints                                  ---

  14--15      4           Pareto frontier analysis; session inspection; cold-start fix; cycle_4 notebook                    pareto_plot.png; cycle_4 PASS                         GATE 4

  16--17      5           Test-split evaluation; paired t-tests; ablation table; interpretability output                    metrics_report.json; ablation_report.pdf              ---

  18          5           CLI demo; E-value documentation; cycle_5 notebook                                                 cli.py demo; cycle_5 PASS                             GATE 5

  19--20      —         Final report writing; GitHub cleanup; README; video demo recording                                Final report PDF + GitHub repo                        DONE


---


## 15. Risks, Mitigations & Ethical Considerations
### 15.1 Technical Risks
  **Risk**                                                          **Severity**   **Mitigation**                                                                       **Detection**

  identify_effect() fails — graph not identifiable                High           Add proxy variables for latent confounders; use IV estimand if available             IdentificationError raised by DoWhy

  CDI bootstrap CI includes 0 (noisy signal)                        High           Upgrade propensity model to GBC; increase MIND training split                        Cycle 2 bootstrap refutation fails

  GCM MAE > 0.08 (poor counterfactual fidelity)                    Medium         Override auto-mechanisms with GradientBoostingRegressor; add interaction features    Cycle 3 KuaiRand holdout test fails

  PPO training instability (loss explodes)                          Medium         Reduce learning rate to 1e-4; normalize reward signal to \[-1,1\]; clip CDI values   TensorBoard loss curves; entropy collapse

  Memory OOM on MIND-large                                          Low            Use chunked parquet reads; stream batches; use MIND-small for all development        OOM error during data pipeline

  Positivity violation — some (user, item) pairs never observed   Medium         Clip propensity scores to \[0.01, 0.99\]; use stabilized IPW weights                 Extreme weights > 100 in propensity distribution plot


### 15.2 Ethical Considerations
**Nudge vs Manipulation**

The system intervenes in a user's information diet without their explicit request. This is ethically justified because: (a) the intervention increases choice rather than reducing it, (b) relevance is a hard constraint — irrelevant content is never recommended, and (c) the system's causal effect is measured and reported, not hidden.

**Transparency**

The CLI demo exposes the CDI score and predicted homogeneity reduction for every recommendation. This allows audit of the system's decision-making. The E-value documents the limits of our causal knowledge.

**User Autonomy**

The w hyperparameter is designed to be user-configurable. A user who prefers pure engagement (no diversity nudge) can set w=1.0. The system must not silently impose diversity goals.

**Data Privacy**

All development uses the MIND and KuaiRand datasets, which are anonymized and publicly released for research. No real user data is collected or retained.


---


## 16. Conclusion & References
### 16.1 Conclusion
This document defines a complete, executable blueprint for a Causal-Informed Recommendation System that mitigates echo chambers in news consumption. The project's central contribution is the integration of DoWhy v0.14 as the causal backbone — replacing ad-hoc estimators with a principled four-step pipeline that enforces assumption transparency, automatic identifiability checking, and mandatory refutation before any result is trusted.

The five-phase architecture (Data → DoWhy Model/Identify → DoWhy Estimate/Refute → PPO RL → Evaluation) is tied together by five sequential Gate iterations that enforce epistemic discipline: no phase begins until its predecessor has passed all validation criteria. This fail-fast structure prevents the compounding of errors that has historically plagued causal inference applications in machine learning.

The project makes a concrete, verifiable claim: a DoWhy-informed RL policy, trained with a composite reward that includes a validated causal diversity signal, will outperform correlation-based baselines on diversity metrics (ILD, Homogeneity Reduction) while sacrificing less than 5% on accuracy metrics (NDCG@10) — and it can prove this claim with statistical significance and a documented sensitivity bound (E-value).

### 16.2 References
| # | Reference |
|---|---|
  1        Sharma, A. & Kiciman, E. (2020). DoWhy: An End-to-End Library for Causal Inference. arXiv:2011.04216.

  2        DoWhy Documentation v0.14. PyWhy Organization. https://www.pywhy.org/dowhy/v0.14/

  3        Zhang, Y. et al. (2022). Dynamic Causal Collaborative Filtering. RecSys 2022.

  4        Mureddu, M. et al. (2022). The Effect of People Recommenders on Echo Chambers and Polarization. ICWSM 2022.

  5        Chen, J. et al. (2021). Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in RS. KDD 2021.

  6        Woo, S. et al. (2025). LLM-Enhanced Reinforcement Learning for Diverse and Novel Recommendations. WSDM 2025.

  7        Cavenaghi, E. et al. (2024). Causal Discovery in Recommender Systems: Example and Discussion. ECIR 2024.

  8        An, F. et al. (2023). KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos. CIKM 2022.

  9        Wu, F. et al. (2020). MIND: A Large-scale Dataset for News Recommendation. ACL 2020.

  10       Pearl, J. (2009). Causality: Models, Reasoning and Inference (2nd ed.). Cambridge University Press.

  11       Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

  12       Ding, P. & VanderWeele, T. (2016). Sensitivity Analysis Without Assumptions. Epidemiology, 27(3).

--- End of Document ---

Abhishek M G · Acharya Institute of Technology · April 2026
