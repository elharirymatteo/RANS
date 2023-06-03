import numpy as np
import pandas as pd


def success_rate_from_distances(distances, threshold=0.02):
    """Compute the success rate from the distances to the target.
    Args:
        distances (np.ndarray): Array of distances to the target for N episodes.
        precision (float): Distance at which the target is considered reached.
    Returns:
        float: Success rate.
    """
    distances_df = pd.DataFrame(distances, columns = [f'Ep_{i}' for i in range(distances.shape[1])])
    # get a boolean dataframe where True means that the distance is less than the threshold
    less_than_thr_df = distances_df.lt(threshold)
    threshold_2 = threshold / 2
    less_than_thr2_df = distances_df.lt(threshold_2)

    # get the index of the first True value for each episode and fill with -1 if there is no True value
    first_less_than_thr_idxs = less_than_thr_df.idxmax().where(less_than_thr_df.any(), -1)
    first_less_than_thr2_idxs = less_than_thr2_df.idxmax().where(less_than_thr2_df.any(), -1)

    #less_than_thr_consistent = less_than_thr_df[less_than_thr_df.index.isin(range(first_less_than_thr_idxs, distances.shape[0]))].all(axis=0)
    print(first_less_than_thr_idxs)
    print(first_less_than_thr2_idxs)

    less_than_margin_df = distances_df.lt(threshold * 4)
    # Check if values are all True after the specific index for each column
    all_true_after_index = pd.DataFrame(index=less_than_margin_df.columns)
    all_true_after_index['all_true'] = less_than_margin_df.apply(lambda column: column.loc[first_less_than_thr_idxs[column.name]:].all(), axis=0)
    success_and_stay_rate = all_true_after_index.value_counts(normalize=True)
    print(all_true_after_index)

    success_rate_thr = (first_less_than_thr_idxs > -1).mean() * 100
    success_rate_thr2 = (first_less_than_thr2_idxs > -1).mean() * 100
    print(f'Success rate with threshold {threshold}: {success_rate_thr}')
    print(f'Success rate with threshold {threshold_2}: {success_rate_thr2}')
    print(f'Success rate and stay with margin {threshold * 4}: {success_and_stay_rate[True] * 100}')
    print(success_and_stay_rate)

    return 0