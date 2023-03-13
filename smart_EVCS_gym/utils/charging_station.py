import time

from numpy import random, zeros
from scipy.io import loadmat, savemat

from smart_nanogrid_gym.utils.charger import Charger
from smart_nanogrid_gym.utils.config import data_files_directory_path
from smart_nanogrid_gym.utils.electric_vehicle import ElectricVehicle


class ChargingStation:
    def __init__(self, number_of_chargers, time_interval):
        self.NUMBER_OF_CHARGERS = number_of_chargers
        array_columns = int(25 / time_interval)
        self.chargers = [Charger() for _ in range(self.NUMBER_OF_CHARGERS)]
        self.vehicle_state_of_charge = zeros([self.NUMBER_OF_CHARGERS, array_columns])
        self.charger_occupancy = zeros([self.NUMBER_OF_CHARGERS, array_columns])
        self.arrivals = []
        self.departures = []
        self.departing_vehicles = []
        self.departure_times = []
        self.vehicle_state_of_charge_at_current_timestep = []

        self.electric_vehicle_info = ElectricVehicle(battery_capacity=40, current_capacity=0, charging_efficiency=0.95,
                                                     discharging_efficiency=0.95, max_charging_power=22,
                                                     max_discharging_power=22)

    def simulate(self, current_timestep, time_interval):
        self.find_departing_vehicles(current_timestep, time_interval)
        self.calculate_departure_times(current_timestep)
        self.extract_current_state_of_charge_per_vehicle(current_timestep)

        return self.departure_times, self.vehicle_state_of_charge_at_current_timestep

    def find_departing_vehicles(self, timestep, time_interval):
        if timestep >= (24 / time_interval):
            return []

        self.departing_vehicles.clear()
        for charger in range(self.NUMBER_OF_CHARGERS):
            charger_occupied = self.check_charger_occupancy(self.charger_occupancy[charger, timestep])
            vehicle_departing = self.check_is_vehicle_departing(self.departures[charger], timestep)

            if charger_occupied and vehicle_departing:
                self.departing_vehicles.append(charger)

    def check_charger_occupancy(self, charger_occupancy):
        if charger_occupancy == 1:
            return True
        else:
            return False

    def check_is_vehicle_departing(self, vehicle_departure, timestep):
        if timestep + 1 in vehicle_departure:
            return True
        else:
            return False

    def calculate_departure_times(self, timestep):
        self.departure_times.clear()
        for charger in range(self.NUMBER_OF_CHARGERS):
            charger_occupied = self.check_charger_occupancy(self.charger_occupancy[charger, timestep])

            if charger_occupied:
                departure_time = self.calculate_next_departure_time(self.departures[charger], timestep)
                self.departure_times.append(departure_time)
            else:
                self.departure_times.append(0)

    def calculate_next_departure_time(self, charger_departures, timestep):
        for vehicle in range(len(charger_departures)):
            if timestep <= charger_departures[vehicle]:
                return charger_departures[vehicle] - timestep
        return []

    def extract_current_state_of_charge_per_vehicle(self, timestep):
        self.vehicle_state_of_charge_at_current_timestep.clear()
        for charger in range(self.NUMBER_OF_CHARGERS):
            self.vehicle_state_of_charge_at_current_timestep.append(self.vehicle_state_of_charge[charger, timestep])

    def load_initial_values(self):
        self.clear_initialisation_variables()

        initial_values = loadmat(data_files_directory_path + '\\initial_values.mat')

        arrival_times = initial_values['Arrivals']
        departure_times = initial_values['Departures']

        self.vehicle_state_of_charge = initial_values['SOC']
        self.charger_occupancy = initial_values['Charger_occupancy']

        for charger in range(self.NUMBER_OF_CHARGERS):
            if arrival_times.shape == (1, self.NUMBER_OF_CHARGERS):
                arrivals = arrival_times[0][charger][0]
                departures = departure_times[0][charger][0]
            elif arrival_times.shape == (self.NUMBER_OF_CHARGERS, 3):
                arrivals = arrival_times[charger]
                departures = departure_times[charger]
            else:
                raise Exception("Initial values loaded from initial_values.mat have wrong shape.")

            self.arrivals.append(arrivals.tolist())
            self.departures.append(departures.tolist())
            self.chargers[charger].vehicle_arrivals = self.arrivals[charger]
            self.chargers[charger].vehicle_state_of_charge = self.vehicle_state_of_charge[charger, :]
            self.chargers[charger].occupancy = initial_values['Charger_occupancy'][charger, :]

    def clear_initialisation_variables(self):
        try:
            self.arrivals.clear()
            self.departures.clear()
            self.charger_occupancy.fill(0)
            self.vehicle_state_of_charge.fill(0)
            return True
        except ValueError:
            return False

    def generate_new_initial_values(self, time_interval):
        initial_variables_cleared = self.clear_initialisation_variables()
        initial_vehicle_presence_generated = self.generate_initial_vehicle_presence(initial_variables_cleared, time_interval)

        generated_initial_values = {
            'SOC': self.vehicle_state_of_charge,
            'Arrivals': self.arrivals,
            'Departures': self.departures,
            'Charger_occupancy': self.charger_occupancy
        } if initial_vehicle_presence_generated else {}

        savemat(data_files_directory_path + '\\initial_values.mat', generated_initial_values)

    def generate_initial_vehicle_presence(self, initial_variables_cleared, time_interval):
        if initial_variables_cleared:
            for charger in range(self.NUMBER_OF_CHARGERS):
                self.generate_initial_vehicle_presence_per_charger(charger, time_interval)
            return True
        return False

    def generate_initial_vehicle_presence_per_charger(self, charger, time_interval):
        vehicle_arrivals = []
        vehicle_departures = []

        vehicle_present = False
        current_departure_time = 0

        total_timesteps = int(24 / time_interval)
        for timestep in range(total_timesteps):
            if not vehicle_present:
                arrival = round(random.rand() - 0.1)
                if arrival == 1 and timestep < total_timesteps:
                    vehicle_present = True

                    self.generate_random_arrival_vehicle_state_of_charge(charger, timestep)
                    vehicle_arrivals.append(timestep)

                    current_departure_time = self.generate_random_vehicle_departure_time(timestep, time_interval, total_timesteps)
                    vehicle_departures.append(current_departure_time)

            if vehicle_present and timestep < current_departure_time:
                self.charger_occupancy[charger, timestep] = 1
                self.chargers[charger].occupancy[timestep] = 1
            else:
                vehicle_present = False
                self.charger_occupancy[charger, timestep] = 0
                self.chargers[charger].occupancy[timestep] = 0

        self.arrivals.append(vehicle_arrivals)
        self.departures.append(vehicle_departures)
        self.chargers[charger].vehicle_arrivals.extend(vehicle_arrivals)

    def generate_random_arrival_vehicle_state_of_charge(self, charger, timestep):
        random_integer = random.randint(10, 90)
        self.vehicle_state_of_charge[charger, timestep] = random_integer / 100
        self.chargers[charger].vehicle_state_of_charge[timestep] = self.vehicle_state_of_charge[charger, timestep]

    def generate_random_vehicle_departure_time(self, timestep, time_interval, total_timesteps):
        max_charging_time = timestep + int(10 / time_interval)
        max_departing_time = total_timesteps + int(1 / time_interval)
        upper_limit = min(max_charging_time, max_departing_time)
        low = timestep + int(4 / time_interval)
        high = int(upper_limit)
        if low >= high:
            return int(low)
        return random.randint(low, high)

    def simulate_vehicle_charging(self, actions, current_timestep, time_interval):
        charger_power_values = zeros(self.NUMBER_OF_CHARGERS)

        for index, charger in enumerate(self.chargers):
            action = actions[index]
            # to-do later (maybe): -1=Charger reserved -> lasts for max 15 minutes, 1=Occupied, 0=Empty
            if charger.occupancy[current_timestep] == 1 and action != 0:
                charger_power_values[index] = charger.charge_or_discharge_vehicle(action, current_timestep, time_interval)
            else:
                charger_power_values[index] = 0

        total_discharging_power = charger_power_values[charger_power_values < 0].sum()
        total_charging_power = charger_power_values[charger_power_values > 0].sum()

        return total_charging_power, total_discharging_power
