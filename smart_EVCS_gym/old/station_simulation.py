import numpy as np


def check_charger_occupancy(charger_occupancy):
    if charger_occupancy == 1:
        return True
    else:
        return False


def check_is_vehicle_departing(vehicle_departure, hour):
    if hour + 1 in vehicle_departure:
        return True
    else:
        return False


def find_departing_vehicles(number_of_chargers, hour, departures, charger_occupancy):
    if hour >= 24:
        return []

    departing_vehicles = []
    for charger in range(number_of_chargers):
        charger_occupied = check_charger_occupancy(charger_occupancy[charger, hour])
        vehicle_departing = check_is_vehicle_departing(departures[charger], hour)

        if charger_occupied and vehicle_departing:
            departing_vehicles.append(charger)

    return departing_vehicles


def calculate_next_departure_time(charger_departures, hour):
    for vehicle in range(len(charger_departures)):
        if hour <= charger_departures[vehicle]:
            return charger_departures[vehicle] - hour
    return []


def calculate_departure_times(number_of_chargers, hour, departures, charger_occupancy):
    departure_times = []
    for charger in range(number_of_chargers):
        charger_occupied = check_charger_occupancy(charger_occupancy[charger, hour])

        if charger_occupied:
            departure_time = calculate_next_departure_time(departures[charger], hour)
            departure_times.append(departure_time)
        else:
            departure_times.append(0)

    return departure_times


def calculate_state_of_charge_for_each_vehicle(number_of_chargers, vehicle_soc, hour):
    vehicles_state_of_charge = []
    for charger in range(number_of_chargers):
        vehicles_state_of_charge.append(vehicle_soc[charger, hour])
    return vehicles_state_of_charge


def simulate_ev_charging_station(self):
    vehicle_soc = self.ev_state_of_charge
    arrivals = self.initial_simulation_values['Arrivals']
    departures = self.initial_simulation_values['Departures']
    charger_occupancy = self.initial_simulation_values['Charger occupancy']
    number_of_chargers = self.NUMBER_OF_CHARGERS
    day = self.day
    hour = self.timestep

    departing_vehicles = find_departing_vehicles(number_of_chargers, hour, departures, charger_occupancy)

    departure_times = calculate_departure_times(number_of_chargers, hour, departures, charger_occupancy)

    vehicles_state_of_charge = calculate_state_of_charge_for_each_vehicle(number_of_chargers, vehicle_soc, hour)

    return departing_vehicles, departure_times, vehicles_state_of_charge
