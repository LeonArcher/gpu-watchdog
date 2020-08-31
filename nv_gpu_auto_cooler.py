import time
import dataclasses
import subprocess
from typing import Optional


def set_auto_fan(gpu_id: int) -> str:
    return (f'DISPLAY=:0 XAUTHORITY=/run/user/121/gdm/Xauthority '
            f'nvidia-settings -a "[gpu:{gpu_id}]/GPUFanControlState=0"')


def set_manual_fan(gpu_id: int, speed=100) -> str:
    return (f'DISPLAY=:0 XAUTHORITY=/run/user/121/gdm/Xauthority '
            f'nvidia-settings -a "[gpu:{gpu_id}]/GPUFanControlState=1" ; '
            f'DISPLAY=:0 XAUTHORITY=/run/user/121/gdm/Xauthority '
            f'nvidia-settings -a "[fan:{gpu_id}]/GPUTargetFanSpeed={speed}"')


def set_power_limit(gpu_id: int, limit: int) -> str:
    return f'nvidia-smi -pl {limit} -i {gpu_id}'


def get_temperature(gpu_id: int) -> str:
    return (f'nvidia-smi --query-gpu=temperature.gpu '
            f'--format=csv,noheader -i {gpu_id}')


def get_number_of_gpus() -> str:
    return 'nvidia-smi --query-gpu=name --format=csv,noheader | wc -l'


@dataclasses.dataclass
class CoolingLevel:
    fan_speed: Optional[int] = None
    power_limit: Optional[int] = None

    def invoke(self, gpu_id: int):
        commands = []

        if self.fan_speed is None:
            commands.append(set_auto_fan(gpu_id))
        else:
            commands.append(set_manual_fan(gpu_id, self.fan_speed))

        if self.power_limit is not None:
            commands.append(set_power_limit(gpu_id, self.power_limit))

        to_execute = ' ; '.join(commands)
        subprocess.run(
            to_execute,
            shell=True,
            check=True,
            universal_newlines=True,
            stdout=subprocess.PIPE,
        )


COOLING_ACTIONS = [
    CoolingLevel(fan_speed=None, power_limit=250),
    CoolingLevel(fan_speed=60, power_limit=250),
    CoolingLevel(fan_speed=80, power_limit=250),
    CoolingLevel(fan_speed=90, power_limit=250),
    CoolingLevel(fan_speed=90, power_limit=175),
    CoolingLevel(fan_speed=100, power_limit=175),
    CoolingLevel(fan_speed=100, power_limit=150),
]


@dataclasses.dataclass
class GPUMonitor:
    gpu_id: int
    temp_lower_boundary: int = 75
    temp_upper_boundary: int = 80
    current_level: int = 0
    verbose: bool = False

    def __post_init__(self):
        COOLING_ACTIONS[self.current_level].invoke(self.gpu_id)

    def _get_temperature(self) -> int:
        command = get_temperature(self.gpu_id)
        t = subprocess.run(
            command,
            shell=True,
            check=True,
            universal_newlines=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        return int(t)

    def check_overheat(self):
        if (
                self._get_temperature() > self.temp_upper_boundary
                and self.current_level < len(COOLING_ACTIONS) - 1
        ):
            if self.verbose:
                print(f'Overheat detected: GPU {self.gpu_id}. Invoking action: '
                      f'{COOLING_ACTIONS[self.current_level + 1]}')

            self.current_level += 1
            COOLING_ACTIONS[self.current_level].invoke(self.gpu_id)

    def check_idle(self):
        if (
                self._get_temperature() < self.temp_lower_boundary
                and self.current_level > 0
        ):
            if self.verbose:
                print(f'Idle detected: GPU {self.gpu_id}. Invoking action: '
                      f'{COOLING_ACTIONS[self.current_level - 1]}')

            self.current_level -= 1
            COOLING_ACTIONS[self.current_level].invoke(self.gpu_id)


if __name__ == '__main__':
    n_gpu = subprocess.run(
        get_number_of_gpus(),
        shell=True,
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    n_gpu = int(n_gpu)

    monitors = [GPUMonitor(gpu_id=i, verbose=True) for i in range(n_gpu)]
    while True:
        for m in monitors:
            m.check_idle()
            m.check_overheat()

        time.sleep(5)
