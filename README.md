# MovieLens Sequential Recommendation

This course project predicts users' future movie interactions and presents Top-10 recommendations. It compares four PyTorch models, movie side-feature fusion, GAN-based augmentation, and a validation-selected optimization.

## Data and evaluation

Place the course-provided MovieLens-Small files `ratings.csv`, `movies.csv`, and `tags.csv` next to the notebook. The dataset contains 610 users, 9,724 rated movies and 100,836 ratings. All observed ratings are treated as interactions, not necessarily positive preferences.

For each user, the last five interactions form the test set, the preceding five form validation, and earlier interactions form training. Item rating statistics, popularity used by the GAN, and GAN training profiles are computed from training data only. Train negatives exclude known training interactions; an unobserved negative can be a future positive. Evaluation uses five held-out interactions and 100 distinct randomly sampled movies outside all observed interactions. The same fixed candidates are shared by models.

This per-user split is not a global time split: another user's training interaction may occur after a given user's target. The same held-out interactions were examined during earlier development, so this test set is not a previously untouched external benchmark. Sampled AUC pools user-item pairs; macro AUC separately compares items within each user and is reported as a complementary measure. MAP@10 and NDCG@10 evaluate ranking within each candidate set.

## Test results

| Model | Pooled AUC | MAP@10 | NDCG@10 |
|---|---:|---:|---:|
| Popularity | 0.8426 | 0.3393 | 0.4742 |
| NCF | 0.8388 | 0.3097 | 0.4508 |
| 1D-CNN | 0.8133 | 0.2598 | 0.3941 |
| GRU | 0.8310 | 0.2767 | 0.4133 |
| SASRec | 0.8473 | 0.3087 | 0.4484 |
| SASRec + GAN | 0.8482 | 0.3187 | 0.4541 |
| SASRec + side features | 0.8687 | 0.3776 | 0.5163 |
| Optimized single model | 0.8661 | 0.5149 | 0.6402 |
| Three CE models | 0.8660 | 0.5240 | 0.6448 |
| Five-model ensemble | 0.8841 | 0.5281 | 0.6499 |

The final choice is the **five-model ensemble**, selected using validation independently of test results. Each model uses history length 40, embedding dimension 64, 200 training negatives, and weight decay 1e-05. Three models use cross-entropy, two use balanced BCE; the CE mean has weight 0.8 and the BCE mean has weight 0.2. Validation experiments and feature ablations are saved in `out/validation_sweep.json`, while the second-round comparisons are in `out/二轮验证对照.json`. The leave-one-out full-catalog NDCG@10 is 0.0452 for the selected model and 0.0185 for popularity. This full-ranking result and the sampled-candidate result use different protocols and cannot be compared numerically.

The three individual CE seeds have test NDCG@10 values 0.6402, 0.6304, 0.6377, with mean 0.6361 and population standard deviation 0.0042. This is descriptive, not a confidence interval. The final ensemble requires five models and five inference passes, compared with three for the previous ensemble.

In a separate validation rerun, the three-CE and five-model ensembles had pooled AUC 0.8900 and 0.9076; 105-candidate NDCG@10 was 0.7285 and 0.7314. With another 1,000 negative items, NDCG@10 was 0.3598 and 0.3594. These candidate sets share validation positives; they are not independent datasets. The full-ranking NDCG difference was 0.0002, with a user-bootstrap 95% interval [-0.0031, 0.0034], so a reliable ranking improvement has not been established.

## Reproduce

Use Python 3.12 and install the listed dependencies, including a PyTorch build compatible with the local CUDA driver if GPU acceleration is desired:

```bash
pip install -r requirements.txt
jupyter notebook 推荐系统.ipynb
```

Run all notebook cells in order from the project directory. The notebook writes figures and JSON summaries to `out/`. The original environment used PyTorch 2.12.0 + CUDA 12.6 and an NVIDIA RTX 4060 Laptop GPU. Seeds are fixed, but exact reproducibility across hardware and library versions is not guaranteed. The notebook includes a Top-10 example excluding the selected user's observed movies.

The validation-only sweep is recorded separately from the final notebook run. All five models are trained in the notebook; `out/selection.json` records the chosen configuration. An ensemble's macro AUC is computed from combined scores, not averaged from individual AUCs. In the second-round full-catalog validation, user bootstrap intervals for the NDCG difference include zero; the ranking gain is therefore uncertain.

In the submission archive, the `实验脚本/` folder contains the validation-only sweep, balanced BCE comparison, GAN threshold comparison, and their notebook source. From the extracted project directory, run `python 实验脚本/sweep.py`, then `python 实验脚本/loss_trial.py` and `python 实验脚本/gan_trial.py` to reproduce these experiments. They replace the corresponding validation JSON files in `out/`.

The second-round scripts are under `实验脚本/二轮/`. Run `python 实验脚本/二轮/try_models.py`, then `python 实验脚本/二轮/check_full.py`, `python 实验脚本/二轮/repeat_baseline.py`, `python 实验脚本/二轮/compare_finalists.py`, and `python 实验脚本/二轮/confidence.py`. They write checkpoints and validation results inside that folder and can take considerably longer than a single notebook run.

## Models

The baseline encoders are NCF, 1D-CNN, GRU, and a simplified SASRec. The side model combines ID embeddings, genres, and training-only rating statistics. A GAN generates synthetic rating profiles for sparse users; selected interactions are shuffled and appended to training histories, so their order is synthetic rather than real chronology. Only the GAN experiment uses the augmented training set. The optimized models use CE or BCE loss and a bidirectional query position.

## References

- He et al., Neural Collaborative Filtering, WWW 2017.
- Hidasi et al., Session-based Recommendations with Recurrent Neural Networks, ICLR 2016.
- Kang and McAuley, Self-Attentive Sequential Recommendation, ICDM 2018.
- Goodfellow et al., Generative Adversarial Nets, NIPS 2014.
- Harper and Konstan, The MovieLens Datasets: History and Context, ACM TIIS 2015.

License: MIT; see `LICENSE`.
