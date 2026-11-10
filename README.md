# OncoEvo-RL: Evolution-Aware Deep Reinforcement Learning for Adaptive Prostate Cancer Therapy

## Overview

OncoEvo-RL is a local computational oncology research prototype that explores evolution-aware adaptive therapy strategies for prostate cancer using Deep Reinforcement Learning (DRL).

The project uses patient-specific PSA data, a calibrated tumour evolution simulator, and reinforcement learning to investigate hypothetical treatment ON/OFF strategies. It also compares the learned policy against continuous-treatment and PSA-threshold intermittent-treatment baselines.

Recorded PSA observations, calibrated estimates, and hypothetical simulation outputs are treated as separate types of information.

**Disclaimer:** This is an experimental computational research prototype, not a clinical treatment recommender. Its simulations and outputs must not be used to make real patient treatment decisions.

## Key Features

- Patient-specific calibration using historical PSA and treatment records.
- Two-population tumour simulation representing treatment-sensitive and treatment-resistant populations.
- Gymnasium-based reinforcement learning environment.
- Proximal Policy Optimization (PPO) using Stable-Baselines3.
- Treatment ON/OFF action selection.
- Comparison with continuous-treatment and PSA-threshold intermittent-treatment baselines.
- Local research dashboard using FastAPI and React/TypeScript.
- Evaluation across patient splits, random seeds, and sensitivity settings.

## Project Structure

```text
OncoEvo-RL/
│
├── app.py
├── run.py
├── config.yaml
├── requirements.txt
├── README.md
│
├── data/
│   └── Patient datasets and processed records
│
├── frontend/
│   └── React / TypeScript application
│
├── models/
│   └── Trained research model checkpoints
│
├── results/
│   └── Evaluation outputs and reports
│
├── src/
│   └── Simulation, training, and evaluation modules
│
└── tests/
    └── Python tests
```

**Note:** The `data/`, `models/`, and `results/` directories are excluded from this repository. They contain local research data, model checkpoints, and generated evaluation outputs. The original dataset must be obtained separately before running training or evaluation.

## Requirements

- Python 3.10 or Python 3.11
- Node.js LTS
- Internet connection for first-time setup

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd OncoEvo-RL
```

### 2. Launch the application

Run:

```bash
python run.py
```

The launcher:

- Installs missing runtime packages.
- Builds the frontend when needed.
- Starts the local application.
- Opens the application in your browser.

Access the application at:

http://127.0.0.1:8000

The application runs locally after the initial setup.

To stop the application, press `Ctrl+C`.

If the default port is occupied, use:

```bash
python run.py --port 8001
```

### 3. Run the original Streamlit interface

The original Streamlit interface is also available:

```bash
streamlit run app.py
```

## Research Commands

### Run Python tests

```bash
python -m pytest -q
```

### Evaluate on the validation split

```bash
python -m src.evaluate --split validation --output-dir results/full_validation
```

### Evaluate on the test split

```bash
python -m src.evaluate --split test --output-dir results/full_test
```

### Train a new PPO policy

```bash
python -m src.train --timesteps 100000 --absorbing-failure-cost --output models/ppo_research_v2.zip
```

Training generates a new model checkpoint and refuses to overwrite an existing checkpoint.

The application launcher does not train models.

### Frontend development and browser testing

Navigate to the frontend directory:

```bash
cd frontend
```

Start the development server:

```bash
npm run dev
```

Run browser tests:

```bash
npm run test:e2e
```

Install Playwright Chromium before running browser tests:

```bash
npx playwright install chromium
```

Python tests require the packages listed in `requirements.txt`.

## Model Evaluation and Limitations

The existing 4,096-step PPO policy showed constant treatment-OFF proposals and poor disease control in lightweight checks.

The research workflow supports full validation and test evaluation using the random seeds and sensitivity settings specified in `config.yaml`.

No clinical benefit has been established. Results are dependent on the simulator, calibration assumptions, and experimental settings.

## Research Sources

1. **Weber / Bruchovsky Patient Series**

   [Prostate Cancer Modeling Study](https://github.com/allison-weber/prostate_cancer_modeling_study)

   Raw records are preserved. Corrections and exclusions are audited in `data/processed/audit.json`.

2. **Zhang et al. (2017)**

   [Ecological competition and cancer treatment](https://www.nature.com/articles/s41467-017-01968-5)

   Provides inspiration for the ecological competition model. This project's two-population PSA proxy is a simplified adaptation.

3. **Zhang et al. (2022)**

   [Adaptive therapy using PSA thresholds](https://elifesciences.org/articles/76284)

   Provides the basis for the adaptive abiraterone PSA-threshold research baseline. This transferred baseline is a research rule, not a universal ADT protocol.

Latent resistance, treatment kinetics, and progression thresholds remain model assumptions.

## Disclaimer

OncoEvo-RL is developed exclusively for computational research and educational purposes. It is not a medical device, clinical decision-support system, or treatment recommendation tool. All treatment strategies and outcomes generated by the simulator are hypothetical and should not be interpreted as clinical evidence.