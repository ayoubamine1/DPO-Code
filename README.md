# DPO-Toxic-Reduction: Aligning LLMs for Safety

**Course:** Reinforcement Learning & Optimal Control (Fall 2025)

**Paper:** [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)

---

## 📌 Project Overview

This repository contains a practical implementation and experimentation of **Direct Preference Optimization (DPO)**. While standard RLHF (Reinforcement Learning from Human Feedback) relies on complex actor-critic pipelines (like PPO) to align models, DPO demonstrates that the language model itself implicitly represents the reward function.

We explore **Scenario A: Safety Alignment via Toxicity Reduction**. Instead of using standard dialogue datasets, we adapted the **IMDb Movie Review dataset** to simulate a safety alignment task, teaching a generic Language Model (DistilGPT-2) to suppress toxic/negative generation styles in favor of constructive/safe outputs without an explicit reward model.

---

## 🧪 The Experiment:

We hypothesize that DPO can effectively treat "Safety" as a preference optimization problem.

### The Setup

| Component | Description |
|-----------|-------------|
| **Dataset** | IMDb Movie Reviews (50k samples) |
| **Winner ($y_w$)** | Positive reviews (Proxy for "Safe/Constructive") |
| **Loser ($y_l$)** | Negative reviews (Proxy for "Toxic/Harmful") |
| **Policy ($\pi_\theta$)** | `distilgpt2` (Fine-tuned) |
| **Reference ($\pi_{ref}$)** | `distilgpt2` (Frozen) |
| **Oracle Judge** | `distilbert-base-uncased-finetuned-sst-2-english` (evaluation only) |

### The Methodology

We utilize the closed-form solution for the KL-constrained RL objective derived in the paper. Instead of training a separate reward model $r(s,a)$, we optimize the policy directly using the **DPO loss**:

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

Where $x$ is the prompt (state $s$) and $y$ is the completion (action $a$).

---

## 📂 Repository Structure

```
DPO Code/
├── data/
│   └── IMDB Dataset.csv          # IMDb movie reviews dataset
├── models/                        # Saved trained models
│   ├── dpo_toxic_reduced_model/  # DPO-trained model
│   └── ppo_toxic_reduced_model/  # PPO-trained model (baseline)
├── evaluation_plots/              # Generated visualizations
│   └── dpo_results_dashboard.png # Main results dashboard
│
├── config.py                      # Hyperparameters & configuration
├── dataset.py                     # IMDb preference pair dataset loader
├── dpo_utils.py                   # DPO loss function & utilities
├── ppo_utils.py                   # PPO loss function & utilities
├── train_dpo.py                   # DPO training pipeline
├── train_ppo.py                   # PPO training pipeline (baseline)
├── eval_utils.py                  # Evaluation metrics & plotting
├── main.py                        # Main entry point
│
├── pyproject.toml                 # Project dependencies (uv)
└── README.md                      # This file
```

---

## 🚀 Usage

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "DPO Code"

# Install dependencies using uv
uv sync
```

### Training the Models

```bash
# Train both DPO and PPO (full comparison)
uv run main.py

# Train DPO only
uv run main.py --dpo-only

# Train PPO only (baseline)
uv run main.py --ppo-only
```

### Analyzing Results

After training, evaluation plots are automatically generated in `evaluation_plots/`:

- **Learning Curve:** Safety score evolution over training steps
- **Pareto Frontier:** Safety vs. KL divergence trade-off
- **Temperature Robustness:** Win rates at different sampling temperatures

---

## 📊 Key Results

Based on the evaluation dashboard (`dpo_results_dashboard.png`):

### 1. **Safety Evolution Over Training**
- **DPO**: Steady improvement from ~0.74 → ~0.88 (120 steps)
- **PPO**: Stagnant/declining from ~0.73 → ~0.67 (120 steps)
- **Conclusion**: DPO achieves significantly higher safety scores with consistent improvement

### 2. **Pareto Frontier (Safety vs. KL Divergence)**
- **DPO**: Higher safety scores (0.78-0.89) at lower KL divergence (0.0-0.55)
- **PPO**: Lower safety scores (0.65-0.74) across similar KL range
- **Conclusion**: DPO is more efficient—achieves better safety with less model drift

### 3. **Temperature Robustness**
- **DPO Win Rate**: 0.70-0.82 across temperatures (0.25, 0.5, 0.75, 1.0)
- **Reference Baseline**: 0.50 (random chance)
- **Conclusion**: DPO maintains superior performance regardless of sampling temperature

---

## 🎯 Key Findings

1. **DPO outperforms PPO** in safety alignment task
   - Higher final safety scores (0.88 vs 0.67)
   - Consistent improvement trajectory vs. stagnation

2. **DPO is more efficient**
   - Achieves higher safety with lower KL divergence
   - Better trade-off between performance and model stability

3. **DPO is robust**
   - Maintains high win rates across different sampling temperatures
   - More reliable than PPO for safety-critical applications

4. **Simpler architecture**
   - No separate reward model needed
   - Direct optimization on preferences
   - Easier to implement and debug

---

## 🔬 Experimental Setup Details

**Hyperparameters**:
- **DPO**: β = 0.1, LR = 5e-5, Batch Size = 4, Max Length = 64
- **PPO**: Clip ε = 0.2, KL Coef = 0.1, Value Coef = 0.1, LR = 5e-5

**Training**:
- 500 samples from IMDb dataset
- Evaluation every 30 steps
- Same base model (DistilGPT-2) for fair comparison

**Evaluation Metrics**:
- Safety Score: Average sentiment score from DistilBERT-SST-2
- KL Divergence: Monte Carlo estimate of D_KL(π_θ || π_ref)
- Win Rate: Fraction of prompts where DPO > Reference at different temperatures



---

## 📚 References

1. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). **Direct Preference Optimization: Your Language Model is Secretly a Reward Model.** *arXiv preprint arXiv:2305.18290.*

2. Course Notes: Introduction to Reinforcement Learning (Fall 2025) - MDPs, Policy Gradients, and KL-Regularization.

---

## 👥 Contributors

- **Ayoub AMINE**
- **Antoine GILSON**
- **Osiris YETNA**

