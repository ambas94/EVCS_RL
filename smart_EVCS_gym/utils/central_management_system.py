from numpy import zeros, array, concatenate, random

from smart_nanogrid_gym.utils.battery_energy_storage_system import BatteryEnergyStorageSystem


class CentralManagementSystem:
    def __init__(self, battery_system_available_in_model, pv_system_available_in_model, vehicle_to_everything):
        # Add , building_demand, building_in_nanogrid as init arguments
        self.total_cost = 0
        self.grid_energy_cost = 0
        self.battery_system = self.initialise_battery_system(battery_system_available_in_model)
        self.pv_system_available = pv_system_available_in_model
        self.vehicle_to_everything = vehicle_to_everything
        # self.building_demand = self.initialise_building_demand(building_in_nanogrid)
        # self.building_exclusive_demand = self.initialise_building_exclusive_demand(building_demand)

    def initialise_battery_system(self, battery_system_available_in_model):
        if battery_system_available_in_model:
            return BatteryEnergyStorageSystem(80, 0.5, 44, 44, 0.95, 0.95, 0.15)
        else:
            return None

    def initialise_building_demand(self, building_in_nanogrid):
        if building_in_nanogrid:
            demand = array([(random.rand() * 10) for _ in range(24)])
            return demand
        else:
            return None

    def initialise_building_exclusive_demand(self, building_demand):
        if building_demand:
            demand = array([round(random.rand() - 0.1) for _ in range(24)])
            return demand
        else:
            return None

    def simulate(self, timestep, total_charging_power, total_discharging_power, solar_power, energy_price,
                 departing_vehicles, soc, battery_action, time_interval):
        available_solar_power = self.get_available_solar_power_at_current_timestep(solar_power, timestep)

        total_power = total_charging_power + total_discharging_power
        grid_power, battery_penalty = self.calculate_grid_power(total_power, available_solar_power, battery_action, time_interval)
        grid_energy = grid_power * time_interval

        self.calculate_grid_energy_cost(grid_energy, energy_price[0, timestep])
        insufficiently_charged_vehicles_penalty = self.calculate_insufficiently_charged_penalty(departing_vehicles, soc,
                                                                                                timestep)
        # Todo: Feat: Add Penaliser class to include all penalty calculations
        #       or Add Accountant class to include all economy calculations, including cost, rewards and penalties
        total_penalty = insufficiently_charged_vehicles_penalty + battery_penalty
        self.calculate_total_cost(total_penalty)

        if self.battery_system:
            battery_soc = self.battery_system.current_capacity
        else:
            battery_soc = 0

        return {
            'Total cost': self.total_cost,
            'Grid power': grid_power,
            'Grid energy': grid_energy,
            'Utilized solar energy': available_solar_power,
            'Insufficiently charged vehicles penalty': insufficiently_charged_vehicles_penalty,
            'Battery state of charge': battery_soc,
            'Grid energy cost': self.grid_energy_cost
        }

    def get_available_solar_power_at_current_timestep(self, solar_power, current_timestep):
        if self.pv_system_available:
            available_solar_power = solar_power[0, current_timestep]
        else:
            available_solar_power = 0
        return available_solar_power

    def calculate_grid_power(self, power_demand, available_solar_power, battery_action, time_interval):
        # if building_in_nanogrid:
        #     remaining_energy_demand = building_demand + total_power - available_renewable_energy
        remaining_power_demand = power_demand - available_solar_power

        if remaining_power_demand == 0:
            return 0, 0
        elif remaining_power_demand > 0:
            power_from_grid, battery_penalty = self.calculate_amount_of_power_supplied_from_grid(remaining_power_demand, battery_action, time_interval)
            return power_from_grid, battery_penalty
        else:
            available_power = available_solar_power - power_demand
            power_to_grid, battery_penalty = self.calculate_amount_of_power_supplied_to_grid(available_power, battery_action, time_interval)
            return power_to_grid, battery_penalty

    def calculate_amount_of_power_supplied_from_grid(self, power_demand, battery_action, time_interval):
        if self.battery_system and battery_action != 0:
            remaining_power_demand, battery_penalty = self.battery_system.discharge(power_demand, battery_action, time_interval)
            return remaining_power_demand, battery_penalty
        else:
            return power_demand, 0

    def calculate_amount_of_power_supplied_to_grid(self, available_power, battery_action, time_interval):
        if self.battery_system and battery_action != 0:
            remaining_available_power, battery_penalty = self.battery_system.charge(available_power, battery_action, time_interval)
        else:
            remaining_available_power = available_power
            battery_penalty = 0

        if self.vehicle_to_everything:
            return -remaining_available_power, battery_penalty
        else:
            # Todo: Feat: Add penalty for wasted energy/power -> wasted_power = (+||-???)remaining_available_power
            return 0, battery_penalty

    def calculate_grid_energy_cost(self, grid_energy, price):
        self.grid_energy_cost = grid_energy * price

    def calculate_insufficiently_charged_penalty(self, departing_vehicles, soc, timestep):
        penalties_per_departing_vehicle = []
        for vehicle in range(len(departing_vehicles)):
            penalty = self.calculate_insufficiently_charged_penalty_per_vehicle(departing_vehicles[vehicle], soc, timestep)
            penalties_per_departing_vehicle.append(penalty)

        return sum(penalties_per_departing_vehicle)

    def calculate_insufficiently_charged_penalty_per_vehicle(self, vehicle, soc, timestep):
        uncharged_capacity = 1 - soc[vehicle, timestep - 1]
        penalty = (uncharged_capacity * 2) ** 2
        return penalty

    def calculate_total_cost(self, total_penalty):
        self.total_cost = self.grid_energy_cost + total_penalty

    def get_energy_price(self, current_price_model, experiment_length_in_days, time_interval):
        grid_tariff_high = 0.028
        grid_tariff_low = 0.013333333
        energy_tariff_high = 0.148933333
        energy_tariff_low = 0.087613333
        res_incentive = 0.014
        high_tariff = grid_tariff_high + energy_tariff_high + res_incentive
        low_tariff = grid_tariff_low + energy_tariff_low + res_incentive

        price_day = self.get_price_day(current_price_model, time_interval, low_tariff=low_tariff, high_tariff=high_tariff)
        price = zeros((experiment_length_in_days, 2 * 24))
        for day in range(0, experiment_length_in_days):
            price[day, :] = price_day
        return price

    def get_price_day(self, current_price_model, time_interval, low_tariff=0.0, high_tariff=0.0):
        price_day = []
        if current_price_model == 0:
            tariffs = []
            for i in range(int(24 / time_interval)):
                if i < 7 / time_interval or i > 19 / time_interval:
                    tariffs.append(low_tariff)
                else:
                    tariffs.append(high_tariff)

            day_tariffs = array(tariffs)
            price_day = array([low_tariff, low_tariff, low_tariff, low_tariff, low_tariff, low_tariff, low_tariff,
                               high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff,
                               high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff,
                               high_tariff, low_tariff, low_tariff, low_tariff, low_tariff])
        if current_price_model == 1:
            price_day = array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
            # low high
        elif current_price_model == 2:
            price_day = array([0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.08, 0.06,
                               0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05])
            # dynamic
        elif current_price_model == 3:
            price_day = array([0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080,
                               0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070])
            # dynamic
        elif current_price_model == 4:
            price_day = array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1])
            # dynamic
        elif current_price_model == 5:
            price_day[1, :] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
            price_day[2, :] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.08, 0.06, 0.05,
                               0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05]
            price_day[3, :] = [0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080,
                               0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070]
            price_day[4, :] = [0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1]

        price_day = concatenate([price_day, price_day], axis=0)
        return price_day
