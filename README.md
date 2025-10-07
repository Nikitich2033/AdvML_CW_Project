# Advanced ML in Finance — Coursework

Short project exploring **reinforcement learning for trading**, with a focus on **Deep Q-Learning (DQN)** applied to equity price data. Includes data prep, environment design, a baseline rules strategy, and a DQN agent trained and evaluated on historical data.

---

## What’s inside

* **Jupyter notebooks**

  * `Deep_Q_Learning_Trading.ipynb` — main RL trading workflow (data load, environment, DQN training/eval).
  * `AdvML_CW_code.ipynb` — supporting analysis/utilities used during the coursework.
* **Data**

  * `INFY.csv` — sample daily OHLCV data used for experiments.
* **Reports & references**

  * `Advanced_ML_Coursework.pdf` and `COMP0162 - Project Instructions.pdf` — coursework brief and write-up.
  * Additional PDFs/images with notes and context.

---

## Quick start

> Python 3.10+ recommended.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 2) Install core dependencies
pip install -U pip wheel
pip install numpy pandas matplotlib scikit-learn gymnasium torch tqdm ta

# optional: for notebooks
pip install jupyter ipykernel
python -m ipykernel install --user --name advml-trading
```

Open the notebook:

```bash
jupyter notebook Deep_Q_Learning_Trading.ipynb
```

---

## Goal

Evaluate whether a **DQN agent** can learn profitable trading policies on daily stock data under realistic constraints (transaction costs, position limits), and compare against simple baselines (buy-and-hold, moving-average crossovers).

---

## Method overview

* **Environment**: custom Gym-style trading env with discrete actions (Buy/Hold/Sell), inventory and cash tracking, transaction cost model, and episodic resets.
* **State**: windowed technical features (returns, moving averages, RSI/volatility via `ta`), position flags, and normalized price context.
* **Agent**: DQN with experience replay, ε-greedy exploration, and target network; MSE loss on TD targets; gradient clipping.
* **Training**: walk-forward or time-split (train/val/test) to reduce look-ahead; early stopping on validation reward.
* **Metrics**: total and risk-adjusted return, max drawdown, hit ratio, trade count, turnover, and strategy stability.

---

## Reproducibility

* Set random seeds in the notebook to make training runs comparable.
* Hyperparameters (learning rate, γ, ε schedule, replay size, batch size) are surfaced at the top of the notebook.
* Use the included `INFY.csv` to replicate baseline results; swap in your own OHLCV CSV with the same columns to re-run.

---

## Results (illustrative)

* Baselines establish a performance floor (buy-and-hold vs rules).
* DQN aims to improve **risk-adjusted** performance through dynamic exposure; outcomes depend strongly on feature set, cost assumptions, and exploration schedule.

> Charts/tables are produced at the end of `Deep_Q_Learning_Trading.ipynb` (equity curves, drawdowns, summary metrics).

---

## Repository structure

```
AdvML_CW_Project/
├─ Deep_Q_Learning_Trading.ipynb      # main RL notebook
├─ AdvML_CW_code.ipynb                # supporting analysis / utilities
├─ INFY.csv                           # sample market data
├─ Advanced_ML_Coursework.pdf         # coursework report
├─ COMP0162 - Project Instructions.pdf# brief / rubric
├─ (other PDFs/images)                 # references & figures
└─ README.md
```

---

## Limitations & next steps

* Single-asset, daily bars; extend to multi-asset or intraday.
* Transaction costs and slippage are simplified; calibrate to venue.
* Consider **Double-DQN / Dueling-DQN**, prioritized replay, and distributional RL.
* Add **hyperparameter sweeps** and **k-fold time splits** for robustness.
* Compare to **policy-gradient** baselines (e.g., PPO) on the same environment.

---

## Minimal `requirements.txt` (optional)

```txt
numpy
pandas
matplotlib
scikit-learn
gymnasium
torch
tqdm
ta
jupyter
ipykernel
```

---

## License

Academic/educational use. If you plan to productionize any components, conduct independent testing and due diligence.
