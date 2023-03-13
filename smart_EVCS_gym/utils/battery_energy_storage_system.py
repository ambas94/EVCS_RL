# from dataclasses import dataclass
from numpy import ndarray


class BatteryEnergyStorageSystem:
    def __init__(self, max_capacity: int, current_capacity: float, max_charging_power: int, max_discharging_power: int,
                 charging_efficiency: float = 0.95, discharging_efficiency: float = 0.95,
                 depth_of_discharge: float = 0.15):
        self.max_capacity: int = max_capacity
        self.current_capacity: float = current_capacity
        self.charging_efficiency: float = charging_efficiency
        self.discharging_efficiency: float = discharging_efficiency
        self.max_charging_power: int = max_charging_power
        self.max_discharging_power: int = max_discharging_power
        self.depth_of_discharge: float = depth_of_discharge

    def charge(self, available_power, battery_action, time_interval):
        capacity_available_to_charge = 1 - self.current_capacity
        # Todo: Feat: Add penalty for negative action when trying to charge battery
        if battery_action < 0:
            battery_penalty = -battery_action
        else:
            battery_penalty = 0
            # Todo: Feat: Add battery_penalty = battery_action or different penalising strategy

        if capacity_available_to_charge > 0:
            power_available_for_charge = (capacity_available_to_charge * self.max_capacity) / time_interval
            max_charging_power = min([self.max_charging_power, power_available_for_charge])
            charging_power = battery_action * max_charging_power

            remaining_available_power = available_power - charging_power

            if remaining_available_power < 0:
                charging_power = battery_action * available_power
                remaining_available_power = 0

            self.current_capacity = self.current_capacity + (charging_power * time_interval) / self.max_capacity

            return remaining_available_power, battery_penalty
        else:
            return available_power, battery_penalty

    def discharge(self, power_demand, battery_action, time_interval):
        capacity_available_to_discharge = self.current_capacity - self.depth_of_discharge
        # Todo: Feat: Add penalty for positive action when trying to discharge battery
        if battery_action > 0:
            battery_penalty = battery_action
        else:
            battery_penalty = 0
            # Todo: Feat: Add battery_penalty = battery_action or different penalising strategy

        if capacity_available_to_discharge > 0:
            power_available_for_discharge = (capacity_available_to_discharge * self.max_capacity) / time_interval
            max_discharging_power = min([self.max_discharging_power, power_available_for_discharge])
            discharging_power = battery_action * max_discharging_power

            remaining_demand = power_demand + discharging_power

            if remaining_demand < 0:
                discharging_power = battery_action * power_demand
                remaining_demand = 0

            self.current_capacity = self.current_capacity + (discharging_power * time_interval) / self.max_capacity

            return remaining_demand, battery_penalty
        else:
            return power_demand, battery_penalty

    def get_state_of_charge(self):
        return self.current_capacity
