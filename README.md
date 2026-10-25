# Drug review sentiment with manual LSTM cells

Advanced Natural Language Processing, 2026/2027.

**Authors:** Sara Baalla, Zouhair Eouichak, Malak Bartal, Aymane Youssi.

Classify English patient reviews as negative, neutral or positive. Vanilla LSTM, CIFG and WMC recurrent cells use `torch.nn.Linear`, sigmoid, tanh and elementwise operations. The studied cells do not use a built-in LSTM layer.

## Measured results

Every run uses the same stratified subsets: 12,000 training, 2,000 validation and 4,000 cleaned test reviews. Each of nine runs trains for four epochs with seed 42. Configuration and checkpoint selection use validation macro-F1 only.

| Architecture | Parameters | LR | Batch | Validation F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 801,219 | 0.003 | 128 | 0.5050 | 63.27% | 0.5052 |
| CIFG | 792,963 | 0.003 | 128 | **0.5246** | **63.78%** | **0.5093** |
| WMC | 813,507 | 0.001 | 128 | 0.4934 | 62.75% | 0.4858 |
| Always positive | 0 | - | - | - | 66.40% | 0.2660 |

CIFG is the primary model selected on validation. The majority baseline has higher accuracy but substantially lower macro-F1. Single-seed results, limited optimization data, rating-derived labels and exact-text deduplication limit generalization claims.

## Installation on Windows / VS Code

Recorded environment: Python 3.12.14, pandas 3.0.1, PyTorch 2.14.1+cpu. Other environments may produce slightly different results. Use a standard Python installation for a standalone environment.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data sources

Original source: [UCI Drugs.com, dataset 462](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com), [DOI](https://doi.org/10.24432/C5SK5S).

The experiment used [this GitHub TSV mirror](https://github.com/semnan-university-ai/Drug-Review-Dataset) after UCI access problems. Its README mentions Druglib.com, but the selected files are **drugsComTrain_raw.tsv** and **drugsComTest_raw.tsv**, with the seven-column Drugs.com schema. Byte identity with a fresh UCI archive has not been independently verified. The exact file hashes are in `resultats/synthese.json`.

Raw files are not committed. Review the original dataset terms before reuse. Download with:

```powershell
.\.venv\Scripts\python.exe telecharger_donnees.py
```

Alternatively put the two TSV files under `data/` with the names above.

## Reproduce the final experiment

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe etape_03_nettoyer.py
.\.venv\Scripts\python.exe etape_04_separer.py
.\.venv\Scripts\python.exe etape_05_vocabulaire.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe etape_10_experiences.py
```

The three settings are LR/batch `0.001/128`, `0.003/128`, `0.001/64`. This compares factors individually rather than a complete factorial grid. Training uses Adam defaults apart from LR, class-weighted cross entropy, gradient clipping at norm 1.0, four epochs, seed 42 and four CPU threads. There is no dropout. Embeddings and hidden states have dimension 64, with a 128-token sequence limit.

The 12,000-entry vocabulary fits the full 95,443-review training pool. Optimization uses only a fixed 12,000-review subset. Validation/test text never participates in vocabulary fitting. Test preparation removes conflicting labels, exact normalized duplicates and any text present in the full training or validation pool, then selects 4,000 examples. Near-duplicates and correlations by drug may remain.

Stage 10 skips completed runs with matching arguments and available checkpoints. After changing code or source data, use a fresh checkout or first move the old `resultats/*lr*.json` and corresponding `.pt` files to a separate archive directory. Do not mix changed inputs with cached runs.

## Inference

```powershell
.\.venv\Scripts\python.exe etape_11_predire.py "This medicine helped me and I feel better." --architecture cifg
```

Requires `resultats/meilleur_cifg.pt`, included in the submission ZIP or produced by stage 10. Alternatives: `vanilla`, `wmc`. The model probabilities are not calibrated confidence or medical recommendations. Only load trusted checkpoints.

## Numbered learning stages

| File | Purpose |
|---|---|
| `etape_01_explorer.py` | Inspect a CSV/TSV and missing values |
| `etape_02_categories.py` | Create rating-derived labels and count classes |
| `etape_03_nettoyer.py` | Decode HTML, normalize texts, remove duplicates and label conflicts |
| `etape_04_separer.py` | Stratified 85/15 split with seed 42 |
| `etape_05_vocabulaire.py` | Training vocabulary and token encodings |
| `etape_06_embedding.py` | Demonstrate an untrained embedding |
| `etape_07_modele_vanilla.py` | Construct the manual Vanilla LSTM |
| `etape_08_entrainer.py` | Educational Vanilla trainer: default 3 epochs on the full training pool |
| `etape_09_variantes.py` | CIFG and WMC cells |
| `etape_10_experiences.py` | Final nine-run comparison: 4 epochs, fixed subsets, final test |
| `etape_11_predire.py` | Inference with a selected trained model |

Stages 6 and 7 are demonstrations, not training. Stage 8 is separate from the final study. Its `--verification` option is a tiny code check, never a final performance result. All reported final numbers come from stage 10.

## Outputs

- `resultats/synthese.json`: runtime, source hashes, split counts, histories, test matrices, baseline.
- `resultats/comparaison.csv`: final model comparison.
- `resultats/*lr*.json`: nine experiment histories.
- `resultats/predictions_*.csv`: actual and predicted class IDs in `test_experience.tsv` order.
- `resultats/manifeste_splits.json`: ordered hashes and labels of subset texts.
- `data/vocabulaire.json`: fixed vocabulary and category mapping.
- `resultats/meilleur_*.pt`: selected checkpoints, in the submission ZIP rather than Git.
- `resultats/*_experience.tsv`: experimental subsets, in the ZIP rather than Git.
- `documents/`: five-page report and PowerPoint with English speaker notes.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Four passing tests check equations against explicit matrix products, numerical gradients for all cells, finite end-to-end gradients, PAD invariance and parameter counts.

## References

- Course brief: `Guidelines_RNN_2627.pdf`, appendix equations and comparison requirements.
- Kallumadi, S. and Grasser, F. (2018), UCI Drugs.com dataset, DOI `10.24432/C5SK5S`.
- CIFG reading in the brief: https://arxiv.org/abs/1503.04069
- WMC reading in the brief: https://www.sciencedirect.com/science/article/pii/S0893608021003439
- PyTorch: https://docs.pytorch.org/docs/stable/nn.html
