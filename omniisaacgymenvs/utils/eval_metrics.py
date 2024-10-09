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


def compute_average_linear_velocity(ep_data: dict) -> float:
    """Compute the average linear velocity of the agent.

    Args:
        ep_data (dict): Dictionary containing the data of an episode.
    Returns:
        float: Average linear velocity of the agent."""
    return np.mean(np.linalg.norm(ep_data["obs"][:, :, 2:4], axis=2))


def compute_average_angular_velocity(ep_data: dict) -> float:
    """Compute the average angular velocity of the agent.

    Args:
        ep_data (dict): Dictionary containing the data of an episode.
    Returns:
        float: Average angular velocity of the agent."""
    return np.mean(np.abs(ep_data["obs"][:, :, 4]))


def compute_average_action_count(ep_data: dict) -> float:
    """Compute the average number of actions taken by the agent.

    Args:
        ep_data (dict): Dictionary containing the data of an episode.
    Returns:
        float: Average number of actions taken by the agent."""
    return np.mean(np.sum(ep_data["act"] != 0, axis=2))


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


def get_GoToPose_success_rate(
    ep_data: dict, print_intermediate: bool = False
) -> dict:
    """Compute the success rate from the distances to the target. 

    Args:
        distances (np.ndarray): Array of distances to the target for N episodes.
        precision (float): Distance at which the target is considered reached.
        p005, p002, p001 (float): (5, 2, 1) cm precision, for position.
        h005, h002, h001 (float): (5, 2, 1) degrees precision, for heading.
        PT, OT (float): Position and orientation time spent under the precision.
    Returns:
        success_rate_df (pd.DataFrame): Success rate for each experiment, using the metrics above.
    """

    distances = np.linalg.norm(ep_data["obs"][:, :, 6:8], axis=2)
    dist = distances
    avg_p005 = np.mean([dist < 0.05])
    avg_p002 = np.mean([dist < 0.02])
    avg_p001 = np.mean([dist < 0.01])
    heading = np.abs(np.arctan2(ep_data["obs"][:, :, -1], ep_data["obs"][:, :, -2]))
    avg_h005 = np.mean([heading < np.pi * 5 / 180])
    avg_h002 = np.mean([heading < np.pi * 2 / 180])
    avg_h001 = np.mean([heading < np.pi * 1 / 180])

    if print_intermediate:
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


def get_TrackXYVelocity_success_rate(ep_data: dict, print_intermediate: bool = False) -> dict:
    """Compute the success rate from the velocity errors.

    Args:
        ep_data (dict): Dictionary containing episode data with keys "obs".
        print_intermediate (bool): Whether to print intermediate results.
        
    Returns:
        success_rate_df (pd.DataFrame): Success rate for each experiment, using the metrics above.
    """
    
    # Extract velocity errors from the episode data
    velocity_errors = np.linalg.norm(ep_data["obs"][:, :, 6:8], axis=2)
    
    # Calculate the percentage of time the velocity error is below the thresholds
    avg_ve1 = np.mean(velocity_errors < 0.1)
    avg_ve05 = np.mean(velocity_errors < 0.05)
    avg_ve02 = np.mean(velocity_errors < 0.02)
    
    if print_intermediate:
        print(
            "percentage of time velocity error is under (10cm/s, 5cm/s, 2cm/s):",
            avg_ve1 * 100,
            avg_ve05 * 100,
            avg_ve02 * 100,
        )
    success_rate_df = pd.DataFrame(
        {
            "VE1": [avg_ve1],
            "VE05": [avg_ve05],
            "VE02": [avg_ve02],
        }
    )
    return {"velocity_tracking": success_rate_df}

def get_GoThroughPoseSequence_success_rate(ep_data: dict, print_intermediate: bool = False) -> dict:
    """Compute the success rate from the position and orientation errors.

    Args:
        ep_data (dict): Dictionary containing episode data with keys "obs".
        print_intermediate (bool): Whether to print intermediate results.
        
    Returns:
        success_rate_df (pd.DataFrame): Success rate for each experiment, using the metrics above.
    """
    
    # Extract position and orientation errors from the episode data
    position_errors = np.linalg.norm(ep_data["obs"][:, :, 6:8], axis=2)
    orientation_errors = np.abs(np.arctan2(ep_data["obs"][:, :, -1], ep_data["obs"][:, :, -2]))
    
    # Calculate the percentage of time the position error is below the thresholds
    avg_p1 = np.mean(position_errors < 0.1)
    avg_p05 = np.mean(position_errors < 0.05)
    avg_p02 = np.mean(position_errors < 0.02)
    
    # Calculate the percentage of time the orientation error is below the thresholds
    avg_o5 = np.mean(orientation_errors < np.pi * 5 / 180)
    avg_o2 = np.mean(orientation_errors < np.pi * 2 / 180)
    avg_o1 = np.mean(orientation_errors < np.pi * 1 / 180)
    
    if print_intermediate:
        print(
            "percentage of time position error is under (10cm, 5cm, 2cm):",
            avg_p1 * 100,
            avg_p05 * 100,
            avg_p02 * 100,
        )
        print(
            "percentage of time orientation error is under (5deg, 2deg, 1deg):",
            avg_o5 * 100,
            avg_o2 * 100,
            avg_o1 * 100,
        )
    
    success_rate_df = pd.DataFrame(
        {
            "P1": [avg_p1],
            "P05": [avg_p05],
            "P02": [avg_p02],
            "O5": [avg_o5],
            "O2": [avg_o2],
            "O1": [avg_o1],
        }
    )
    
    return {"trajectory_tracking": success_rate_df}

# Example usage
ep_data_example = {
    "act": [], 
    "obs": np.random.rand(10, 20, 8),  # Random example data
    "rews": []
}

result = get_GoThroughPoseSequence_success_rate(ep_data_example, print_intermediate=True)
print(result)



def get_GoToXY_success_rate(ep_data: dict, print_intermediate: bool = False) -> dict:
    """Compute the success rate from the position errors.

    Args:
        ep_data (dict): Dictionary containing episode data with keys "obs".
        print_intermediate (bool): Whether to print intermediate results.
        
    Returns:
        success_rate_df (pd.DataFrame): Success rate for each experiment, using the metrics above.
    """
    
    # Extract position errors from the episode data
    position_errors = np.linalg.norm(ep_data["obs"][:, :, 6:8], axis=2)
    
    # Calculate the percentage of time the position error is below the thresholds
    avg_p1 = np.mean(position_errors < 0.1)
    avg_p05 = np.mean(position_errors < 0.05)
    avg_p02 = np.mean(position_errors < 0.02)
    
    if print_intermediate:
        print(
            "percentage of time position error is under (10cm, 5cm, 2cm):",
            avg_p1 * 100,
            avg_p05 * 100,
            avg_p02 * 100,
        )
    
    success_rate_df = pd.DataFrame(
        {
            "P1": [avg_p1],
            "P05": [avg_p05],
            "P02": [avg_p02],
        }
    )
    
    return {"position_control": success_rate_df}

# Example usage
ep_data_example = {
    "act": [], 
    "obs": np.random.rand(10, 20, 8),  # Random example data
    "rews": []
}

result = get_GoToXY_success_rate(ep_data_example, print_intermediate=True)
print(result)



def get_GoToPose_results(
    ep_data: dict,
    position_threshold: float = 0.02,
    heading_threshold: float = 0.087,
    print_intermediate: bool = False,
) -> None:
    SR = get_GoToPose_success_rate(ep_data, print_intermediate=False)
    alv = compute_average_linear_velocity(ep_data)
    aav = compute_average_angular_velocity(ep_data)
    aac = compute_average_action_count(ep_data) / 8

    ordered_metrics_keys = [
        "PA1",
        "PA2",
        "PSA",
        "OA1",
        "OA2",
        "OSA",
        "ALV",
        "AAV",
        "AAC",
        "PT5",
        "PT2",
        "PT1",
        "OT5",
        "OT2",
        "OT1",
    ]

    ordered_metrics_descriptions = [
        "Position reached below 0.02 m of the target",
        "Position reached below 0.01 m of the target",
        "Position success and stay within 0.15 m",
        "Orientation reached below 0.087 rad of the target",
        "Orientation reached below 0.0435 rad of the target",
        "Orientation success and stay within 0.6525 rad",
        "Average linear velocity",
        "Average angular velocity",
        "Average action count",
        "Percentage of time spent within 0.05 m of the target",
        "Percentage of time spent within 0.02 m of the target",
        "Percentage of time spent within 0.01 m of the target",
        "Percentage of time spent within 0.05 rad of the target",
        "Percentage of time spent within 0.02 rad of the target",
        "Percentage of time spent within 0.01 rad of the target",
    ]

    ordered_metrics_units = [
        "%",
        "%",
        "%",
        "%",
        "%",
        "%",
        "m/s",
        "rad/s",
        "N",
        "%",
        "%",
        "%",
        "%",
        "%",
        "%",
    ]

    ordered_metrics_multipliers = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        100,
        100,
        100,
        100,
        100,
        100,
    ]

    metrics = np.array(
        [
            alv,  # ALV
            aav,  # AAV
            aac,  # AAC
            SR["pose"]["PT5"][0],  # PT5
            SR["pose"]["PT2"][0],  # PT2
            SR["pose"]["PT1"][0],  # PT1
            SR["pose"]["OT5"][0],  # OT5
            SR["pose"]["OT2"][0],  # OT2
            SR["pose"]["OT1"][0],  # OT1
        ]
    )

    # Print the metrics line by line
    print(f"Metrics acquired using a sample of {ep_data['act'].shape[1]}:")
    # for i, (metric, unit, mult, desc) in enumerate(
    #     zip(
    #         ordered_metrics_keys,
    #         ordered_metrics_units,
    #         ordered_metrics_multipliers,
    #         ordered_metrics_descriptions,
    #     )
    # ):
    #     print(f"  + {metric}: {metrics[i]*mult:.2f}{unit}. {desc}.")
    return


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
