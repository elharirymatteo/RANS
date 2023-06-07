import torch

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)

class Core:
    def __init__(self, num_envs, device):
        self._num_envs = num_envs
        self._device = device
    
        self._dim_orientation: 2 # theta heading in the world frame (cos(theta), sin(theta)) [0:2]
        self._dim_velocity: 2 # velocity in the world (x_dot, y_dot) [2:4]
        self._dim_omega: 1 # rotation velocity (theta_dot) [4]
        self._dim_task_label: 1 # label of the task to be executed (int) [5]
        self._dim_task_data: 4 # data to be used to fullfil the task (floats) [6:10]

        self._num_observations = 10
        self._obs_buffer = torch.zeros((self._num_envs, self._num_observations), device=self._device, dtype=torch.float32)
        self._task_label = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._task_data = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
    
    def update_observation_tensor(self, current_state: dict):
        self._obs_buffer[:, 0:2] = current_state["orientation"]
        self._obs_buffer[:, 2:4] = current_state["linear_velocity"]
        self._obs_buffer[:, 4] = current_state["angular_velocity"]
        self._obs_buffer[:, 5] = self._task_label
        self._obs_buffer[:, 6:10] = self._task_data
        return self._obs_buffer

class TaskDict:
    def __init__(self):
        self.gotoxy = 0
        self.gotopose = 1
        self.trackxyvel = 2
        self.trackxyovel = 3
        self.trackxyvelheading = 4

def parse_data_dict(dataclass, data, ask_for_validation=False):
    unknown_keys = []
    for key in data.keys():
        if key in dataclass.__dict__.keys():
            dataclass.__setattr__(key, data[key])
        else:
            unknown_keys.append(key)
    try:
        dataclass.__post_init__()
    except:
        pass

    print("Parsed configuration parameters:")
    for key in dataclass.__dict__:
        print("     + "+key+":"+str(dataclass.__getattribute__(key)))
    if unknown_keys:
        print("The following keys were given but do not match any parameters:")
        for i, key in enumerate(unknown_keys):
            print("     + "+str(i)+" : "+key)
    if ask_for_validation:
        lock = input("Press enter to validate.")
    return dataclass