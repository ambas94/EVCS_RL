import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from scipy.io import savemat
import time

from smart_nanogrid_gym.utils.central_management_system import CentralManagementSystem
from smart_nanogrid_gym.utils.charging_station import ChargingStation
from smart_nanogrid_gym.utils.pv_system_manager import PVSystemManager
from ..utils.config import data_files_directory_path


class SmartNanogridEnv(gym.Env):
    def __init__(self, price_model=0, pv_system_available_in_model=True, battery_system_available_in_model=True,
                 vehicle_to_everything=False):
        # Add building_in_nanogrid=False, building_demand=False as init arguments
        self.NUMBER_OF_CHARGERS = 8
        self.NUMBER_OF_DAYS_TO_PREDICT = 1
        # TODO: Add method for setting time_interval by keyword argument from ['1h', '2h'...-> '?h'; '15min'...->'?min']
        self.TIME_INTERVAL = 1
        self.NUMBER_OF_HOURS_AHEAD = 3
        self.CURRENT_PRICE_MODEL = price_model
        self.PV_SYSTEM_AVAILABLE_IN_MODEL = pv_system_available_in_model
        self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL = battery_system_available_in_model
        self.VEHICLE_TO_EVERYTHING = vehicle_to_everything
        # self.BUILDING_IN_NANOGRID = building_in_nanogrid

        self.charging_station = ChargingStation(self.NUMBER_OF_CHARGERS, self.TIME_INTERVAL)
        # self.central_management_system = CentralManagementSystem(battery_system_available_in_model,
        #                                                          building_demand, building_in_nanogrid)
        self.central_management_system = CentralManagementSystem(self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL,
                                                                 self.PV_SYSTEM_AVAILABLE_IN_MODEL,
                                                                 self.VEHICLE_TO_EVERYTHING)
        if self.PV_SYSTEM_AVAILABLE_IN_MODEL:
            self.pv_system_manager = PVSystemManager(self.NUMBER_OF_DAYS_TO_PREDICT, self.TIME_INTERVAL)

        self.timestep = None
        self.info = None

        self.solar_radiation = None
        self.available_solar_energy = None
        self.energy_price = None
        self.grid_energy_per_timestep, self.solar_energy_utilization_per_timestep = None, None
        self.total_cost_per_timestep, self.penalty_per_timestep = None, None
        self.battery_per_timestep, self.grid_energy_cost_per_timestep = None, None
        self.grid_power_per_timestep = None

        self.simulated_single_day = False

        amount_of_observed_variables = 1 + int(self.PV_SYSTEM_AVAILABLE_IN_MODEL)
        number_of_observed_charger_values = 2

        amount_of_charger_predictions = self.NUMBER_OF_CHARGERS * number_of_observed_charger_values
        amount_of_states = amount_of_observed_variables + (self.NUMBER_OF_HOURS_AHEAD * amount_of_observed_variables)

        self.total_amount_of_states = amount_of_states + amount_of_charger_predictions + int(self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL)

        spaces_low = np.array(np.zeros(self.total_amount_of_states), dtype=np.float32)
        spaces_high = np.array(np.ones(self.total_amount_of_states), dtype=np.float32)

        if self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL:
            if self.VEHICLE_TO_EVERYTHING:
                actions_low = np.array(np.ones(self.NUMBER_OF_CHARGERS + 1), dtype=np.float32) * (-1)
            else:
                actions_low = np.array(np.zeros(self.NUMBER_OF_CHARGERS), dtype=np.float32)
                actions_low = np.insert(actions_low, self.NUMBER_OF_CHARGERS, -1)
            actions_high = np.array(np.ones(self.NUMBER_OF_CHARGERS + 1), dtype=np.float32)

            self.action_space = spaces.Box(low=actions_low, high=actions_high, shape=(self.NUMBER_OF_CHARGERS + 1,),
                                           dtype=np.float32)
        else:
            if self.VEHICLE_TO_EVERYTHING:
                actions_low = -1
            else:
                actions_low = 0
            actions_high = 1
            self.action_space = spaces.Box(low=actions_low, high=actions_high, shape=(self.NUMBER_OF_CHARGERS,),
                                           dtype=np.float32)

        self.observation_space = spaces.Box(low=spaces_low, high=spaces_high, dtype=np.float32)

        # Todo: Add look-ahead action_space for looking at agents planned actions to see will departing vehicles be
        #       charged enough based on current action, and penalize wrong future actions

    def step(self, actions):
        charger_actions = actions[0:self.NUMBER_OF_CHARGERS]

        if self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL:
            battery_action = actions[-1]
        else:
            battery_action = 0
        [total_charging_power, total_discharging_power] = self.charging_station.simulate_vehicle_charging(charger_actions,
                                                                                                          self.timestep,
                                                                                                          self.TIME_INTERVAL)
        if self.PV_SYSTEM_AVAILABLE_IN_MODEL:
            available_solar_power = self.pv_system_manager.get_available_solar_produced_power(self.TIME_INTERVAL)
        else:
            available_solar_power = 0

        results = self.central_management_system.simulate(self.timestep, total_charging_power, total_discharging_power,
                                                          available_solar_power, self.energy_price,
                                                          self.charging_station.departing_vehicles,
                                                          self.charging_station.vehicle_state_of_charge,
                                                          battery_action, self.TIME_INTERVAL)

        self.total_cost_per_timestep.append(results['Total cost'])
        self.grid_power_per_timestep.append(results['Grid power'])
        self.grid_energy_per_timestep.append(results['Grid energy'])
        self.solar_energy_utilization_per_timestep.append(results['Utilized solar energy'])
        self.penalty_per_timestep.append(results['Insufficiently charged vehicles penalty'])
        self.battery_per_timestep.append(results['Battery state of charge'])
        self.grid_energy_cost_per_timestep.append(results['Total cost'])

        observations = self.__get_observations()
        self.timestep = self.timestep + 1

        self.simulated_single_day = self.__check_is_single_day_simulated()
        if self.simulated_single_day:
            self.timestep = 0
            self.__save_prediction_results()

        reward = -results['Total cost']
        self.info = {}

        return observations, reward, self.simulated_single_day, self.info

    def __get_observations(self):
        [departure_times, vehicles_state_of_charge] = self.charging_station.simulate(self.timestep, self.TIME_INTERVAL)
        if self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL:
            battery_soc = self.central_management_system.battery_system.get_state_of_charge()
        else:
            battery_soc = 0.0

        min_timesteps_ahead = self.timestep + 1
        max_timesteps_ahead = min_timesteps_ahead + self.NUMBER_OF_HOURS_AHEAD

        max_price = self.energy_price.max()
        if self.PV_SYSTEM_AVAILABLE_IN_MODEL:
            max_radiation = self.solar_radiation.max()
            normalized_disturbances_observation_at_current_timestep = np.array([
                self.solar_radiation[0, self.timestep] / max_radiation,
                self.energy_price[0, self.timestep] / max_price
            ])

            normalized_predictions = np.concatenate((
                np.array([self.solar_radiation[0, min_timesteps_ahead:max_timesteps_ahead] / max_radiation]),
                np.array([self.energy_price[0, min_timesteps_ahead:max_timesteps_ahead] / max_price])),
                axis=None
            )
        else:
            normalized_disturbances_observation_at_current_timestep = np.array([
                self.energy_price[0, self.timestep] / max_price
            ])

            normalized_predictions = np.array([self.energy_price[0, min_timesteps_ahead:max_timesteps_ahead] / max_price])

        if self.BATTERY_SYSTEM_AVAILABLE_IN_MODEL:
            normalized_states = np.concatenate((
                np.array(vehicles_state_of_charge),
                np.array(departure_times) / 24,
                np.array(battery_soc)),
                axis=None
            )
        else:
            normalized_states = np.concatenate((
                np.array(vehicles_state_of_charge),
                np.array(departure_times) / 24),
                axis=None
            )

        observations = np.concatenate((
            normalized_disturbances_observation_at_current_timestep,
            normalized_predictions,
            normalized_states),
            axis=None, dtype=np.float32
        )

        return observations

    def __check_is_single_day_simulated(self):
        if self.timestep == (24 / self.TIME_INTERVAL):
            return True
        else:
            return False

    def __save_prediction_results(self):
        if self.PV_SYSTEM_AVAILABLE_IN_MODEL:
            available_solar_energy = self.available_solar_energy
        else:
            available_solar_energy = []

        prediction_results = {
            'SOC': self.charging_station.vehicle_state_of_charge,
            'Grid_power': self.grid_power_per_timestep,
            'Grid_energy': self.grid_energy_per_timestep,
            'Utilized_solar_energy': self.solar_energy_utilization_per_timestep,
            'Penalties': self.penalty_per_timestep,
            'Available_solar_energy': available_solar_energy,
            'Total_cost': self.total_cost_per_timestep,
            'Battery_state_of_charge': self.battery_per_timestep,
            'Grid_energy_cost': self.grid_energy_cost_per_timestep
        }
        savemat(data_files_directory_path + '\\prediction_results.mat', {'Prediction_results': prediction_results})

    def reset(self, generate_new_initial_values=True):
        self.timestep = 0
        self.simulated_single_day = False
        self.total_cost_per_timestep = []
        self.grid_power_per_timestep = []
        self.grid_energy_per_timestep = []
        self.solar_energy_utilization_per_timestep = []
        self.penalty_per_timestep = []
        self.battery_per_timestep = []
        self.grid_energy_cost_per_timestep = []

        if self.PV_SYSTEM_AVAILABLE_IN_MODEL:
            self.solar_radiation = self.pv_system_manager.get_solar_radiation()
            self.available_solar_energy = self.pv_system_manager.get_available_solar_energy()

        self.energy_price = self.central_management_system.get_energy_price(self.CURRENT_PRICE_MODEL,
                                                                            self.NUMBER_OF_DAYS_TO_PREDICT,
                                                                            self.TIME_INTERVAL)
        self.__load_initial_simulation_values(generate_new_initial_values)

        return self.__get_observations()

    def __load_initial_simulation_values(self, generate_new_initial_values):
        if generate_new_initial_values:
            self.charging_station.generate_new_initial_values(self.TIME_INTERVAL)
        else:
            self.charging_station.load_initial_values()

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]
        pass

    def close(self):
        # return 0
        pass
