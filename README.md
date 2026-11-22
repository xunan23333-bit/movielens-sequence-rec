# Sequential Movie Recommendation

This PyTorch notebook compares a GRU and single-layer query attention on the same history input and candidate sets. It also evaluates history decay, movie features, and GAN augmentation. The selected query attention model produces ten recommendations for each of 610 users.

## Run

Use Python 3.12 with the packages in requirements.txt. Place the provided ratings.csv and movies.csv next to 推荐系统.ipynb. Open the notebook and run all cells from a clean kernel. The notebook uses CUDA when available. It saves the run log and metrics in out/results.json and recommendations in out/top10.csv.

## Evaluation

For each user the last five ratings are test targets and the preceding five are validation targets. Each sampled evaluation compares five targets and 100 fixed negatives. AUC pools user-candidate scores. Full-catalog NDCG uses the same five targets, filtering other known items. Earlier runs had inspected these test interactions.

The matched comparison uses 40 interactions, 64 dimensions, sampled cross-entropy and no side features. The chosen attention setting is **40 interactions, decay 0.95, side features: both**. Values below are means across seeds 42 and 7.

| Model | AUC | MAP@10 | NDCG@10 | Full NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| GRU | 0.8841 | 0.4476 | 0.5752 | 0.0491 |
| Query attention, matched settings | 0.8934 | 0.4750 | 0.6026 | 0.0513 |
| Query attention, selected settings | 0.8717 | 0.5034 | 0.6284 | 0.0613 |

GAN augmentation and an equal-count popularity control share a backbone and match the added interactions and ratings per user. Both controls add 8650 interactions. See the notebook for the short- and long-history validation results.

Data credit: [GroupLens MovieLens data and usage terms](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html), [Harper and Konstan, 2015](https://doi.org/10.1145/2827872).

Code license: MIT. See LICENSE.
