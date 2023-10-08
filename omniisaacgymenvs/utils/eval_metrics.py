__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"


import pandas as pd
import numpy as np


def build_distance_dataframe(distances: np.ndarray, threshold: float) -> list:
    distances_df = pd.DataFrame(
        distances, columns=[f"Ep_{i}" for i in range(distances.shape[1])]
    )
    # get a boolean dataframe where True means that the distance is less than the threshold
    less_than_thr_df = distances_df.lt(threshold)
    threshold_2 = threshold / 2
    less_than_thr2_df = distances_df.lt(threshold_2)

    # get the index of the first True value for each episode and fill with -1 if there is no True value
    first_less_than_thr_idxs = less_than_thr_df.idxmax().where(
        less_than_thr_df.any(), -1
    )
    first_less_than_thr2_idxs = less_than_thr2_df.idxmax().where(
        less_than_thr2_df.any(), -1
    )

    margin = threshold * 7.5
    less_than_margin_df = distances_df.lt(margin)
    return less_than_margin_df, first_less_than_thr_idxs, first_less_than_thr2_idxs


def check_stay(
    less_than_margin_df: pd.DataFrame,
    first_less_than_thr_idxs: pd.DataFrame,
    first_less_than_thr2_idxs: pd.DataFrame,
) -> list:
    all_true_after_index = pd.DataFrame(index=less_than_margin_df.columns)
    all_true_after_index["all_true"] = less_than_margin_df.apply(
        lambda column: column.loc[first_less_than_thr_idxs[column.name] :].all(), axis=0
    )
    success_and_stay_rate = all_true_after_index.value_counts(normalize=True)
    success_and_stay_rate = (
        success_and_stay_rate[True] if True in success_and_stay_rate.index else 0
    )

    success_rate_thr = (first_less_than_thr_idxs > -1).mean() * 100
    success_rate_thr2 = (first_less_than_thr2_idxs > -1).mean() * 100
    return success_rate_thr, success_rate_thr2, success_and_stay_rate


def print_success(
    success_rate_thr: float,
    success_rate_thr2: float,
    success_and_stay_rate: float,
    threshold: float,
    print_intermediate: bool = False,
) -> None:
    if print_intermediate:
        print(f"Success rate with threshold {threshold}: {success_rate_thr}")
        print(f"Success rate with threshold {threshold/2}: {success_rate_thr2}")
        print(
            f"Success rate and stay with margin {threshold*7.5}: {success_and_stay_rate * 100}"
        )


def get_GoToPose_success_rate_new(
    ep_data: dict, threshold: float = 0.02, print_intermediate: bool = False
) -> dict:
    """Compute the success rate from the distances to the target.

    Args:
        distances (np.ndarray): Array of distances to the target for N episodes.
        precision (float): Distance at which the target is considered reached.
    Returns:
        float: Success rate."""

    distances = np.linalg.norm(ep_data["obs"][:, :, 6:8], axis=2)
    dist = distances
    avg_p005 = np.mean([dist < 0.05])
    avg_p002 = np.mean([dist < 0.02])
    avg_p001 = np.mean([dist < 0.01])
    heading = np.abs(np.arctan2(ep_data["obs"][:, :, -1], ep_data["obs"][:, :, -2]))
    avg_h005 = np.mean([heading < np.pi * 5 / 180])
    avg_h002 = np.mean([heading < np.pi * 2 / 180])
    avg_h001 = np.mean([heading < np.pi * 1 / 180])
    print(
        "percentage of time spent under (5cm, 2cm, 1cm):",
        avg_p005 * 100,
        avg_p002 * 100,
        avg_p001 * 100,
    )
    print(
        "percentage of time spent under (5deg, 2deg, 1deg):",
        avg_h005 * 100,
        avg_h002 * 100,
        avg_h001 * 100,
    )
    success_rate_df = pd.DataFrame(
        {
            "PT5": [avg_p005],
            "PT2": [avg_p002],
            "PT1": [avg_p001],
            "OT5": [avg_h005],
            "OT2": [avg_h002],
            "OT1": [avg_h001],
        }
    )

    return {"pose": success_rate_df}


def get_GoToXY_success_rate(
    ep_data: dict, threshold: float = 0.02, print_intermediate: bool = False
) -> dict:
    """Compute the success rate from the distances to the target.

    Args:
        distances (np.ndarray): Array of distances to the target for N episodes.
        precision (float): Distance at which the target is considered reached.
    Returns:
        float: Success rate."""

    distances = np.linalg.norm(ep_data["obs"][:, :, 6:8], axis=2)
    (
        less_than_margin_df,
        first_less_than_thr_idxs,
        first_less_than_thr2_idxs,
    ) = build_distance_dataframe(distances, threshold)
    success_rate_thr, success_rate_thr2, success_and_stay_rate = check_stay(
        less_than_margin_df, first_less_than_thr_idxs, first_less_than_thr2_idxs
    )
    print_success(
        success_rate_thr,
        success_rate_thr2,
        success_and_stay_rate,
        threshold,
        print_intermediate,
    )
    success_rate_df = pd.DataFrame(
        {
            f"success_rate_{threshold}_m": [success_rate_thr],
            f"success_rate_{threshold/2}_m": [success_rate_thr2],
            f"success_and_stay_within_{threshold*7.5}_m": [success_and_stay_rate * 100],
        }
    )

    return {"position": success_rate_df}


def get_GoToPose_success_rate(
    ep_data: dict,
    position_threshold: float = 0.02,
    heading_threshold: float = 0.087,
    print_intermediate: bool = False,
) -> dict:
    """Compute the success rate from the distances to the target.

    Args:
        distances (np.ndarray): Array of distances to the target for N episodes.
        precision (float): Distance at which the target is considered reached.
    Returns:
        float: Success rate."""

    position_distances = np.linalg.norm(ep_data["obs"][:, :, 6:8], axis=2)
    heading_distances = np.abs(
        np.arctan2(ep_data["obs"][:, :, 9], ep_data["obs"][:, :, 8])
    )

    (
        less_than_margin_df,
        first_less_than_thr_idxs,
        first_less_than_thr2_idxs,
    ) = build_distance_dataframe(position_distances, position_threshold)
    success_rate_thr, success_rate_thr2, success_and_stay_rate = check_stay(
        less_than_margin_df, first_less_than_thr_idxs, first_less_than_thr2_idxs
    )
    print_success(
        success_rate_thr,
        success_rate_thr2,
        success_and_stay_rate,
        position_threshold,
        print_intermediate,
    )

    position_success_rate_df = pd.DataFrame(
        {
            f"success_rate_{position_threshold}_m": [success_rate_thr],
            f"success_rate_{position_threshold/2}_m": [success_rate_thr2],
            f"success_and_stay_within_{position_threshold*7.5}_m": [
                success_and_stay_rate * 100
            ],
        }
    )

    (
        less_than_margin_df,
        first_less_than_thr_idxs,
        first_less_than_thr2_idxs,
    ) = build_distance_dataframe(heading_distances, heading_threshold)
    success_rate_thr, success_rate_thr2, success_and_stay_rate = check_stay(
        less_than_margin_df, first_less_than_thr_idxs, first_less_than_thr2_idxs
    )
    print_success(
        success_rate_thr,
        success_rate_thr2,
        success_and_stay_rate,
        heading_threshold,
        print_intermediate,
    )

    heading_success_rate_df = pd.DataFrame(
        {
            f"success_rate_{heading_threshold}_rad": [success_rate_thr],
            f"success_rate_{heading_threshold/2}_rad": [success_rate_thr2],
            f"success_and_stay_within_{heading_threshold*7.5}_rad": [
                success_and_stay_rate * 100
            ],
        }
    )

    return {"position": position_success_rate_df, "heading": heading_success_rate_df}


def get_TrackXYVelocity_success_rate(
    ep_data: dict, threshold: float = 0.15, print_intermediate: bool = False
) -> dict:
    """Compute the success rate from the distances to the target.

    Args:
        distances (np.ndarray): Array of distances to the target for N episodes.
        precision (float): Distance at which the target is considered reached.
    Returns:
        float: Success rate."""

    distances = np.linalg.norm(ep_data["obs"][:, :, 6:8], axis=2)
    (
        less_than_margin_df,
        first_less_than_thr_idxs,
        first_less_than_thr2_idxs,
    ) = build_distance_dataframe(distances, threshold)
    success_rate_thr, success_rate_thr2, success_and_stay_rate = check_stay(
        less_than_margin_df, first_less_than_thr_idxs, first_less_than_thr2_idxs
    )
    print_success(
        success_rate_thr,
        success_rate_thr2,
        success_and_stay_rate,
        threshold,
        print_intermediate,
    )

    success_rate_df = pd.DataFrame(
        {
            f"success_rate_{threshold}_m/s": [success_rate_thr],
            f"success_rate_{threshold/2}_m/s": [success_rate_thr2],
            f"success_and_stay_within_{threshold*7.5}_m/s": [
                success_and_stay_rate * 100
            ],
        }
    )

    return {"xy_velocity": success_rate_df}


def get_TrackXYOVelocity_success_rate(
    ep_data: dict,
    xy_threshold: float = 0.15,
    omega_threshold: float = 0.3,
    print_intermediate: bool = False,
) -> float:
    """Compute the success rate from the distances to the target.

    Args:
        distances (np.ndarray): Array of distances to the target for N episodes.
        precision (float): Distance at which the target is considered reached.
    Returns:
        float: Success rate."""

    xy_distances = np.linalg.norm(ep_data["obs"][:, :, 6:8], axis=2)
    omega_distances = np.abs(ep_data["obs"][:, :, 8])

    (
        less_than_margin_df,
        first_less_than_thr_idxs,
        first_less_than_thr2_idxs,
    ) = build_distance_dataframe(xy_distances, xy_threshold)
    success_rate_thr, success_rate_thr2, success_and_stay_rate = check_stay(
        less_than_margin_df, first_less_than_thr_idxs, first_less_than_thr2_idxs
    )
    print_success(
        success_rate_thr,
        success_rate_thr2,
        success_and_stay_rate,
        xy_threshold,
        print_intermediate,
    )

    xy_success_rate_df = pd.DataFrame(
        {
            f"success_rate_{xy_threshold}_m/s": [success_rate_thr],
            f"success_rate_{xy_threshold/2}_m/s": [success_rate_thr2],
            f"success_and_stay_within_{xy_threshold*7.5}_m/s": [
                success_and_stay_rate * 100
            ],
        }
    )

    (
        less_than_margin_df,
        first_less_than_thr_idxs,
        first_less_than_thr2_idxs,
    ) = build_distance_dataframe(omega_distances, omega_threshold)
    success_rate_thr, success_rate_thr2, success_and_stay_rate = check_stay(
        less_than_margin_df, first_less_than_thr_idxs, first_less_than_thr2_idxs
    )
    print_success(
        success_rate_thr,
        success_rate_thr2,
        success_and_stay_rate,
        omega_threshold,
        print_intermediate,
    )

    omega_success_rate_df = pd.DataFrame(
        {
            f"success_rate_{omega_threshold}_rad/s": [success_rate_thr],
            f"success_rate_{omega_threshold/2}_rad/s": [success_rate_thr2],
            f"success_and_stay_within_{omega_threshold*7.5}_rad/s": [
                success_and_stay_rate * 100
            ],
        }
    )

    return {"xy_velocity": xy_success_rate_df, "omega_velocity": omega_success_rate_df}


def get_success_rate_table(success_rate_df: pd.DataFrame) -> None:
    print(
        success_rate_df.to_latex(
            index=False,
            formatters={"name": str.upper},
            float_format="{:.1f}".format,
            bold_rows=True,
            caption="Success rate for each experiment.",
            label="tab:success_rate",
        )
    )
