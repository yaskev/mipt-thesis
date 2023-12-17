## Datasets
### Original data for most experiments:
100k prices. 100 paths per price, 365 steps
datasets/train/prices_mc_with_ci.csv

10k prices. 100 paths per price, 365 steps
datasets/test/prices_mc_with_ci.csv

### More accurate data for training for intrinsic value:
5k prices. 20k paths per price, 365 steps
prices_mc_20000_paths_5000.csv

1k prices. 20k per price, 365 steps
prices_mc_20k_test_paths_1000.csv

### Even more accurate data for estimating greeks with acceptable accuracy
100 prices. 40k paths per price, 365 steps
No explicit training required
datasets/train/greeks_mc_with_ci.csv


## Models
### Trained positive network using basic dataset from part 1
trained_positive.sav

### Trained convex network using basic dataset from part 1
trained_convex.sav

### Trained positive network using accurate data from part 2
pos_no_preproc_20k_paths_17_12.sav
MSE: 1.17e-2
Training chart: positive_network/charts/pos_no_preproc_20k_paths_17_12.jpg

### Trained (and overfitted) net using accurate data from part 2 and preprocessing with intrinsic value subtraction and log
pos_intr_log_20k_paths_overfit_17_12.sav
MSE train: 2.81e-03, val MSE: 1.45e-02
Training chart: positive_network/charts/pos_intr_log_20k_paths_overfit_17_12.jpg
This is a usual FF network with 4 linear layers