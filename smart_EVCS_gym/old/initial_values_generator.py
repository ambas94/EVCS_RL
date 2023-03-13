import numpy as np
from numpy import random
from scipy.io import loadmat, savemat


def generate_new_values(file_directory_path, number_of_chargers):
    arrivals = []
    departures = []

    vehicle_state_of_charge = np.zeros([number_of_chargers, 25])
    charger_occupancy = np.zeros([number_of_chargers, 25])

    # initial state stochastic creation
    for charger in range(number_of_chargers):
        is_vehicle_present = 0
        total_occupancy_timesteps_per_charger = 0
        arrival = 0

        vehicle_arrivals = []
        vehicle_departures = []

        for hour in range(24):
            if is_vehicle_present == 0:
                arrival = round(random.rand()-0.1)
                if arrival == 1 and hour <= 20:
                    random_integer = random.randint(20, 50)
                    vehicle_state_of_charge[charger, hour] = random_integer / 100

                    total_occupancy_timesteps_per_charger = total_occupancy_timesteps_per_charger+1

                    vehicle_arrivals.append(hour)
                    upper_limit = min(hour + 10, 25)
                    vehicle_departures.append(random.randint(hour+4, int(upper_limit)))

            if arrival == 1 and total_occupancy_timesteps_per_charger > 0:
                if hour < vehicle_departures[total_occupancy_timesteps_per_charger - 1]:
                    is_vehicle_present = 1
                    charger_occupancy[charger, hour] = 1
                else:
                    is_vehicle_present = 0
                    charger_occupancy[charger, hour] = 0
            else:
                is_vehicle_present = 0
                charger_occupancy[charger, hour] = 0

        arrivals.append(vehicle_arrivals)
        departures.append(vehicle_departures)

    # information vector creator
    total_vehicles_charging = np.zeros([24])
    for hour in range(24):
        total_vehicles_charging[hour] = np.sum(charger_occupancy[:, hour])

    generated_initial_values = {
        'SOC': vehicle_state_of_charge,
        'Arrivals': arrivals,
        'Departures': departures,
        'Total vehicles charging': total_vehicles_charging,
        'Charger occupancy': charger_occupancy
    }

    savemat(file_directory_path + '\\initial_values.mat', generated_initial_values)

    return generated_initial_values


def load_initial_values(file_directory_path, number_of_chargers):
    initial_values = loadmat(file_directory_path + '\\initial_values.mat')

    arrival_times = initial_values['Arrivals']
    departure_times = initial_values['Departures']

    reformatted_arrivals = []
    reformatted_departures = []

    for charger in range(number_of_chargers):
        if arrival_times.shape == (1, 10):
            arrivals = arrival_times[0][charger][0]
            departures = departure_times[0][charger][0]
        elif arrival_times.shape == (10, 3):
            arrivals = arrival_times[charger]
            departures = departure_times[charger]
        else:
            raise Exception("Initial values loaded from initial_values.mat have wrong shape.")

        reformatted_arrivals.append(arrivals.tolist())
        reformatted_departures.append(departures.tolist())

    return {
        'SOC': initial_values['SOC'],
        'Arrivals': reformatted_arrivals,
        'Departures': reformatted_departures,
        'Total vehicles charging': initial_values['Total vehicles charging'],
        'Charger occupancy': initial_values['Charger occupancy']
    }
