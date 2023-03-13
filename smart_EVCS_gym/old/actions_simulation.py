import numpy as np
import time


def calculate_charging_or_discharging_power(max_charging_power, action):
    return action * max_charging_power


def calculate_next_vehicle_state_of_charge(power_value, self, arrival, hour, soc):
    if hour in arrival:
        soc[hour] = soc[hour] + power_value / self.EV_PARAMETERS['CAPACITY']
    else:
        soc[hour] = soc[hour - 1] + power_value / self.EV_PARAMETERS['CAPACITY']
        # soc[charger, timestep] = soc[charger, timestep - 1] + (charging_power[charger] * time_interval) / \
        #                          self.EV_PARAMETERS['CAPACITY']


def calculate_max_charging_power(self, arrival, hour, soc):
    # max_charging_energy = min([self.EV_PARAMETERS['MAX CHARGING POWER'],
    #                           soc[charger, timestep] * self.EV_PARAMETERS['CAPACITY'] / time_interval])
    if hour in arrival:
        remaining_uncharged_capacity = 1 - soc[hour]
        power_left_to_charge = remaining_uncharged_capacity * self.EV_PARAMETERS['CAPACITY']
    else:
        remaining_uncharged_capacity = 1 - soc[hour - 1]
        power_left_to_charge = remaining_uncharged_capacity * self.EV_PARAMETERS['CAPACITY']

    max_charging_energy = min([self.EV_PARAMETERS['MAX CHARGING POWER'], power_left_to_charge])
    return max_charging_energy


def charge_vehicle(self, action, arrival, hour, vehicle_soc):
    max_charging_power = calculate_max_charging_power(self, arrival, hour, vehicle_soc)
    charging_power = calculate_charging_or_discharging_power(max_charging_power, action)
    calculate_next_vehicle_state_of_charge(charging_power, self, arrival, hour, vehicle_soc)
    return charging_power


def calculate_max_discharging_power(self, arrival, hour, soc):
    if hour in arrival:
        power_left_to_discharge = soc[hour] * self.EV_PARAMETERS['CAPACITY']
    else:
        power_left_to_discharge = soc[hour - 1] * self.EV_PARAMETERS['CAPACITY']

    max_discharging_energy = min([self.EV_PARAMETERS['MAX DISCHARGING POWER'], power_left_to_discharge])
    return max_discharging_energy


def discharge_vehicle(self, action, arrival, hour, vehicle_soc):
    max_discharging_power = calculate_max_discharging_power(self, arrival, hour, vehicle_soc)
    discharging_power = calculate_charging_or_discharging_power(max_discharging_power, action)
    calculate_next_vehicle_state_of_charge(discharging_power, self, arrival, hour, vehicle_soc)
    return discharging_power


def charge_or_discharge_vehicle(self, action, arrival, hour, vehicle_soc):
    if action >= 0:
        charger_power_value = charge_vehicle(self, action, arrival, hour, vehicle_soc)
    else:
        charger_power_value = discharge_vehicle(self, action, arrival, hour, vehicle_soc)

    return charger_power_value


def calculate_available_renewable_energy(renewable, consumed):
    available_energy = renewable - consumed
    return max([0, available_energy])


def calculate_grid_energy(total_power, available_renewable_energy):
    grid_energy = total_power - available_renewable_energy
    return max([grid_energy, 0])


def calculate_grid_energy_cost(grid_energy, price):
    cost = grid_energy * price
    return cost


def calculate_insufficiently_charged_penalty_per_vehicle(vehicle, soc, hour):
    uncharged_capacity = 1 - soc[vehicle, hour + 1]
    penalty = (uncharged_capacity * 2) ** 2
    return penalty


def calculate_insufficiently_charged_penalty(departing_vehicles, soc, hour):
    penalties_per_departing_vehicle = []
    for vehicle in range(len(departing_vehicles)):
        penalty = calculate_insufficiently_charged_penalty_per_vehicle(departing_vehicles[vehicle], soc, hour)
        penalties_per_departing_vehicle.append(penalty)

    return sum(penalties_per_departing_vehicle)


def calculate_total_cost(grid_cost, total_penalty):
    return grid_cost + total_penalty


def simulate_central_management_system(self, actions):
    # hour = self.timestep
    # timestep = self.timestep
    # time_interval = 1
    hour = self.timestep
    consumed = self.solar_radiation['Consumed']
    renewable = self.solar_radiation['Available renewable']
    charger_occupancy = self.initial_simulation_values['Charger occupancy']
    arrivals = self.initial_simulation_values['Arrivals']

    departing_vehicles = self.departing_vehicles
    soc = self.ev_state_of_charge

    charger_power_values = np.zeros(self.NUMBER_OF_CHARGERS)

    for charger in range(self.NUMBER_OF_CHARGERS):
        # to-do later (maybe): -1=Charger reserved -> lasts for max 15 minutes, 1=Occupied, 0=Empty
        if charger_occupancy[charger, hour] == 1:
            charger_power_values[charger] = charge_or_discharge_vehicle(self, actions[charger], arrivals[charger],
                                                                        hour, soc[charger])
        else:
            charger_power_values[charger] = 0

    total_charging_power = sum(charger_power_values)

    available_renewable_energy = calculate_available_renewable_energy(renewable[0, hour], consumed[0, hour])
    grid_energy = calculate_grid_energy(total_charging_power, available_renewable_energy)

    grid_energy_cost = calculate_grid_energy_cost(grid_energy, self.solar_radiation["Price"][0, hour])
    insufficiently_charged_vehicles_penalty = calculate_insufficiently_charged_penalty(departing_vehicles, soc, hour)

    total_cost = calculate_total_cost(grid_energy_cost, insufficiently_charged_vehicles_penalty)

    return {
        'Total cost': total_cost,
        'Grid energy': grid_energy,
        'Utilized renewable energy': available_renewable_energy,
        'Insufficiently charged vehicles penalty': insufficiently_charged_vehicles_penalty,
        'EV state of charge': soc
    }
