# MovieLens Sequential Recommendation

A PyTorch course project for predicting movie interactions and producing Top-10 recommendations from user rating histories. The notebook contains data processing, model definitions, training, evaluation, experimental results, and an example recommendation list.

## Contents

- `推荐系统.ipynb`: complete, executed experiment notebook.
- `requirements.txt`: Python dependencies.
- `LICENSE`: project license.

## Reproduce the experiment

Use Python 3.12 and install the dependencies in `requirements.txt`. Install a PyTorch build compatible with your CUDA driver if using a GPU. Put `ratings.csv`, `movies.csv`, and `tags.csv` from the course-provided MovieLens-Small dataset next to the notebook, then run all cells in order:

```bash
pip install -r requirements.txt
jupyter notebook 推荐系统.ipynb
```

The original run used an NVIDIA RTX 4060 Laptop GPU and PyTorch 2.12.0 with CUDA 12.6. Training five ensemble models takes substantially longer than training one model. The notebook creates its own `out/` directory. Random seeds are fixed; results can vary across software and hardware.

## Models and evaluation

Four baselines are compared: NCF, 1D-CNN, GRU, and simplified SASRec. Further experiments evaluate a GAN for data augmentation, fusion of movie genres and training-only rating statistics, and an attention model using a final query position. The final ensemble combines three cross-entropy models and two balanced binary-cross-entropy models, with group weights of 0.8 and 0.2 selected on validation data.

For each user, the last five interactions are held out for testing and the preceding five for validation. All earlier interactions are training data. Every observed rating is treated as an interaction, not necessarily a positive preference. Evaluation ranks five held-out interactions with 100 fixed, distinct sampled negatives per user. Training statistics and GAN profiles use training data only. The split is per user, rather than a global time split. Earlier experiments had already inspected the same test targets, so this test set is not a new independent benchmark.

## Results

| Model | Pooled AUC | MAP@10 | NDCG@10 |
|---|---:|---:|---:|
| Popularity | 0.8426 | 0.3393 | 0.4742 |
| NCF | 0.8388 | 0.3097 | 0.4508 |
| 1D-CNN | 0.8133 | 0.2598 | 0.3941 |
| GRU | 0.8310 | 0.2767 | 0.4133 |
| Simplified SASRec | 0.8473 | 0.3087 | 0.4484 |
| SASRec + GAN | 0.8482 | 0.3187 | 0.4541 |
| SASRec + side features | 0.8687 | 0.3776 | 0.5163 |
| Optimized single model | 0.8661 | 0.5149 | 0.6402 |
| Five-model ensemble | 0.8841 | 0.5281 | 0.6499 |

For full-catalog leave-one-out ranking, NDCG@10 was 0.0452 for the final model and 0.0185 for popularity. These scores use a different protocol from the sampled-candidate results. A separate validation comparison found a small full-ranking difference whose user-bootstrap interval included zero; the ranking improvement is uncertain.

## References

- He et al., Neural Collaborative Filtering, WWW 2017.
- Hidasi et al., Session-based Recommendations with Recurrent Neural Networks, ICLR 2016.
- Kang and McAuley, Self-Attentive Sequential Recommendation, ICDM 2018.
- Goodfellow et al., Generative Adversarial Nets, NIPS 2014.
- Harper and Konstan, The MovieLens Datasets: History and Context, ACM TIIS 2015.

License: MIT. See `LICENSE`.
