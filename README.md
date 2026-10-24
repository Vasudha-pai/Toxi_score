# Pathway-specific toxicity prediction (Tox21)

Predicting **which biological pathway** a compound perturbs, rather than a single
toxic/non-toxic flag. Twelve endpoints: seven nuclear receptor (NR) assays and
five stress response (SR) assays.

Built as a learning project. Every script is meant to be read, not just run —
the comments explain *why* each choice was made, including the ones that cost
you accuracy on paper but buy you honesty.

---

## Running it

```bash
conda create -n toxpred python=3.11
conda activate toxpred
conda install -c conda-forge rdkit scikit-learn pandas numpy pyarrow
pip install lightgbm chembl_structure_pipeline shap

cd src
python s00_download.py      # fetch Tox21 (~8k compounds, 12 assays)
python s01_standardize.py   # clean structures, dedupe, merge labels
python s02_featurize.py     # Morgan fingerprints + RDKit descriptors
python s03_split.py         # Bemis-Murcko scaffold split
python s04_baseline.py --features both --model rf
```

Steps 0–3 take a few minutes total. Step 4 takes roughly 10–20 minutes
depending on cores.

---

## What came out

Data after cleaning: **7,514 unique compounds**, down from 8,014 raw rows.

Rejected during standardization:

| reason | count |
|---|---|
| mixture (multiple fragments after salt stripping) | 73 |
| inorganic (no carbon) | 50 |
| too small (<3 heavy atoms) | 33 |
| unparseable SMILES | 8 |
| too large (>100 heavy atoms) | 7 |

Plus **329 duplicate rows merged**, during which **127 label conflicts** were
found — the same structure recorded as both active and inactive in the same
assay. Those are set to missing rather than resolved by guesswork. If you skip
deduplication, whichever duplicate pandas happens to keep decides the label for
you, silently.

### Baseline results (scaffold split, test set)

| features | model | mean ROC-AUC | mean PR-AUC |
|---|---|---|---|
| Morgan + descriptors | RF | **0.802** | **0.382** |
| Morgan + descriptors | LightGBM | 0.780 | 0.381 |
| descriptors only | LightGBM | 0.782 | 0.369 |
| Morgan only | RF | 0.742 | 0.332 |
| Morgan only | LightGBM | 0.721 | 0.302 |

Random-guess PR-AUC is 0.105 (the mean active rate), so the best config is a
**3.6× lift** over random.

Per-assay, best config:

| assay | test actives | ROC-AUC | PR-AUC | random PR-AUC |
|---|---|---|---|---|
| NR-AR | 14 | 0.642 | 0.031 | 0.021 |
| NR-AR-LBD | 12 | 0.866 | 0.235 | 0.020 |
| NR-AhR | 110 | 0.840 | 0.623 | 0.195 |
| NR-Aromatase | 55 | 0.819 | 0.529 | 0.115 |
| NR-ER | 68 | 0.767 | 0.410 | 0.130 |
| NR-ER-LBD | 22 | 0.740 | 0.150 | 0.035 |
| NR-PPAR-gamma | 31 | 0.845 | 0.319 | 0.057 |
| SR-ARE | 119 | 0.791 | 0.590 | 0.246 |
| SR-ATAD5 | 33 | 0.854 | 0.385 | 0.052 |
| SR-HSE | 51 | 0.834 | 0.426 | 0.087 |
| SR-MMP | 84 | 0.849 | 0.526 | 0.188 |
| SR-p53 | 69 | 0.772 | 0.354 | 0.114 |

### Three things in that table worth sitting with

**1. Descriptors beat fingerprints here, and that is surprising.**
Morgan alone gets 0.332 PR-AUC; 214 physicochemical descriptors alone get
0.369. Substructure matching is the usual default in cheminformatics, but
several of these assays are driven by bulk properties — lipophilicity,
membrane permeability, general reactivity — not by one privileged scaffold.
SR-MMP (mitochondrial membrane potential) is the clearest case: disrupting a
membrane potential is a physicochemical act. Combining both blocks helps, but
less than you would expect, because the two blocks overlap more than they look
like they do.

**2. ROC-AUC and PR-AUC tell different stories, and PR-AUC is the honest one.**
NR-AR-LBD gets ROC-AUC 0.866, which reads as a good model. Its PR-AUC is 0.235.
At a 2% active rate, ranking most inactives below most actives is easy; putting
actives at the *top* is the hard part, and that is what you need when you can
only follow up on the top 100 compounds. This is the same failure you hit on
the IL-6 model — respectable ROC-AUC, poor precision on actives. It was never a
bug in that model; it is what ROC-AUC does on imbalanced data.

**3. NR-AR is not a model, it is noise.**
PR-AUC 0.031 against a 0.021 baseline, with 14 test actives. Do not report this
number as if it means something. With that few positives, the confidence
interval swallows the estimate whole. Either merge NR-AR with NR-AR-LBD (same
receptor, different readout), or drop it and say why. Reporting a mean across
all 12 assays that includes this one is how benchmark papers get quietly wrong.

---

## Design decisions, and what they cost

**Missing labels stay missing.** Roughly a third of the label matrix is empty.
Some distributions (TDC's, notably) fill those with 0, turning "never tested"
into "inactive". That inflates results and teaches the model a fiction. Here,
each assay trains only on compounds actually tested in it.

**Scaffold split, not random split.** Bemis-Murcko groups are kept whole, so
test compounds have cores the model has never seen. This costs roughly
0.03–0.08 AUC versus a random split. That is not the model getting worse — it
is the random-split number having been wrong. 3,974 scaffold groups across
7,514 compounds tells you how much analogue redundancy is in this dataset.

**No `AddHs()` anywhere.** Featurization must be byte-identical at training and
prediction time. Adding explicit hydrogens at one and not the other shifts the
fingerprint bits silently — the model still runs, AUC still looks plausible,
precision quietly collapses.

**No threshold is chosen.** The models output ranked probabilities. Picking an
operating point depends on how many compounds you can afford to follow up on,
which is a decision about your resources, not a property of the model.

---

## Where to take it next

Roughly in order of effort, each one teaches something different:

**Make the evaluation trustworthy first.** Everything above is a single
scaffold split. Repeat over 3–5 different scaffold splits and report mean ± sd.
If the spread is ±0.05, a paper claiming a 0.01 improvement over baseline has
told you nothing. This is the cheapest and most valuable next step.

**Applicability domain.** You already have Tanimoto AD code from IL-6. Drop it
in: compute max similarity from each test compound to the training set, bin by
similarity, and plot PR-AUC per bin. You will almost certainly find performance
falls off a cliff below ~0.3 Tanimoto. That plot is the single most useful thing
you can hand someone who wants to actually use the model.

**Multitask learning — the real question.** A shared-trunk neural net with 12
output heads lets the data-rich assays (SR-ARE, NR-AhR) inform the data-poor
ones (NR-PPAR-gamma, NR-AR-LBD). Measure the per-assay gain over the single-task
baselines in this README. The hypothesis is that assays sharing biology help
each other, so the gain should be uneven and concentrated in related pathways —
if it is, that is a genuinely interesting result, not a benchmark number.

**SHAP for structural alerts.** Run SHAP on the descriptor models, per assay.
Check whether the top features recover known alerts: phenolic/stilbene-like
features for NR-ER, electrophiles and Michael acceptors for SR-ARE, lipophilic
cations for SR-MMP. If they do, the model learned chemistry. If it learned
"molecular weight > 400", it learned the dataset. This connects directly to
your HSP90 XAI work.

**Chemprop D-MPNN as the graph baseline.** `pip install chemprop`. Learned graph
representations versus fixed fingerprints is the standard comparison, and on
Tox21 the margin is usually smaller than people expect.

**The composite score.** Aggregate the 12 predictions into NR-burden and
SR-burden scores, then test whether those predict an external endpoint — LD50
(`LD50_Zhu` from TDC), DILI, or hERG — better than a model trained directly on
that endpoint. That is a real "biological-component-specific toxicity score"
and it is the piece that would make this more than a reproduction.

**REINVENT4 tie-in.** You have the generative setup already. Wrap the trained
multitask model as a scoring component and run multi-objective optimization:
potency up, NR/SR liability down. Directly reusable in the HSP90 work.

---

## Layout

```
toxpred/
├── README.md
├── data/
│   ├── raw/tox21.csv                  # 8,014 rows as downloaded
│   └── processed/
│       ├── tox21_clean.csv            # 7,514 standardized, deduped
│       ├── tox21_split.csv            # + scaffold split column
│       ├── features_morgan.parquet    # 2,048 bits
│       └── features_descriptors.parquet  # 214 descriptors
├── results/
│   ├── baseline_*.csv / .json         # per-assay metrics per config
│   └── split_summary.json
└── src/
    ├── common.py            # paths, task names, seed
    ├── s00_download.py
    ├── s01_standardize.py   # the step that decides if anything means anything
    ├── s02_featurize.py
    ├── s03_split.py
    └── s04_baseline.py
```

## Data notes

MoleculeNet version of Tox21 (~8k compounds). The original Tox21 Challenge set
(12,060 train / 647 held-out test) lives at
<https://tripod.nih.gov/tox21/challenge/data.jsp> and differs in both
composition and evaluation protocol — numbers from the two are **not**
comparable, so do not claim to beat a Challenge leaderboard using this version.
