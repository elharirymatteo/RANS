from argparse import ArgumentParser
from omniisaacgymenvs.utils.eval_metrics import (
    get_GoToPose_success_rate_new,
    get_GoToPose_success_rate,
    compute_average_action_count,
    compute_average_angular_velocity,
    compute_average_linear_velocity,
)
import pandas as pd
import numpy as np
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--folder_path", type=str, default=None)
    parser.add_argument("--save_metrics", action="store_true")
    parser.add_argument("--use_xyzw", action="store_true")
    parser.add_argument("--use_wxyz", action="store_true")
    parser.add_argument("--display_metrics", action="store_true")
    return parser.parse_args()


args = parse_args()

if args.use_xyzw and args.use_wxyz:
    raise ValueError("Cannot use both xyzw and wxyz")

if not args.use_xyzw and not args.use_wxyz:
    raise ValueError("Must use either xyzw or wxyz")

folder_path = args.folder_path
save_metrics = args.save_metrics

files = os.listdir(folder_path)
csvs = [f for f in files if f.endswith(".csv")]

eps_data = {}
eps_data["obs"] = []
eps_data["act"] = []

obss = []
acts = []

for csv in csvs:
    df = pd.read_csv(os.path.join(folder_path, csv))

    # Replicate an observation buffer
    obs = np.zeros((df.shape[0], 10))

    # Position
    x = df["x_position"].to_numpy()
    y = df["y_position"].to_numpy()
    tx = df["x_position_target"].to_numpy()
    ty = df["y_position_target"].to_numpy()

    # Velocities
    vx = df["x_linear_velocity"].to_numpy()
    vy = df["y_linear_velocity"].to_numpy()
    vrz = df["z_angular_velocity"].to_numpy()

    # Heading
    if args.use_xyzw:
        quat = np.column_stack(
            [
                df["x_quaternion"],
                df["y_quaternion"],
                df["z_quaternion"],
                df["w_quaternion"],
            ]
        )
    elif args.use_wxyz:
        quat = np.column_stack(
            [
                df["w_quaternion"],
                df["x_quaternion"],
                df["y_quaternion"],
                df["z_quaternion"],
            ]
        )
    else:
        raise ValueError("Must use either xyzw or wxyz")

    th = df["heading_target"].to_numpy()
    siny_cosp = 2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
    cosy_cosp = 1 - 2 * (quat[:, 2] * quat[:, 2] + quat[:, 3] * quat[:, 3])
    orient_z = np.arctan2(siny_cosp, cosy_cosp)
    heading_error = np.arctan2(np.sin(th - orient_z), np.cos(th - orient_z))
    obs[:, 0] = np.cos(orient_z)
    obs[:, 1] = np.sin(orient_z)
    obs[:, 2] = vx
    obs[:, 3] = vy
    obs[:, 4] = vrz
    obs[:, 5] = 1
    obs[:, 6] = tx - x
    obs[:, 7] = ty - y
    obs[:, 8] = np.cos(heading_error)
    obs[:, 9] = np.sin(heading_error)

    act = np.column_stack(
        [
            df["t_0"].to_numpy(),
            df["t_1"].to_numpy(),
            df["t_2"].to_numpy(),
            df["t_3"].to_numpy(),
            df["t_4"].to_numpy(),
            df["t_5"].to_numpy(),
            df["t_6"].to_numpy(),
            df["t_7"].to_numpy(),
        ]
    )

    acts.append([act])
    obss.append([obs])

eps_data["act"] = np.concatenate(acts, axis=0)
eps_data["obs"] = np.concatenate(obss, axis=0)

new_SR = get_GoToPose_success_rate_new(eps_data, print_intermediate=False)
old_SR = get_GoToPose_success_rate(eps_data, print_intermediate=False)
alv = compute_average_linear_velocity(eps_data)
aav = compute_average_angular_velocity(eps_data)
aac = compute_average_action_count(eps_data) / 8

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
        old_SR["position"]["success_rate_0.02_m"][0],  # PA1
        old_SR["position"]["success_rate_0.01_m"][0],  # PA2
        old_SR["position"]["success_and_stay_within_0.15_m"][0],  # PSA
        old_SR["heading"]["success_rate_0.087_rad"][0],  # OA1
        old_SR["heading"]["success_rate_0.0435_rad"][0],  # OA2
        old_SR["heading"]["success_and_stay_within_0.6525_rad"][0],  # OSA
        alv,  # ALV
        aav,  # AAV
        aac,  # AAC
        new_SR["pose"]["PT5"][0],  # PT5
        new_SR["pose"]["PT2"][0],  # PT2
        new_SR["pose"]["PT1"][0],  # PT1
        new_SR["pose"]["OT5"][0],  # OT5
        new_SR["pose"]["OT2"][0],  # OT2
        new_SR["pose"]["OT1"][0],  # OT1
    ]
)

np.save(os.path.join(folder_path, "aggregated_results.npy"), metrics)

# Print the metrics line by line
print(f"Metrics acquired using a sample of {eps_data['act'].shape[0]}:")
if args.display_metrics:
    for i, (metric, unit, mult, desc) in enumerate(
        zip(
            ordered_metrics_keys,
            ordered_metrics_units,
            ordered_metrics_multipliers,
            ordered_metrics_descriptions,
        )
    ):
        print(f"  + {metric}: {metrics[i]*mult:.2f}{unit}. {desc}.")
