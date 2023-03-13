import numpy as np
from numpy import mean, reshape, zeros
from scipy.io import loadmat

from smart_nanogrid_gym.utils.pv_system import PVSystem
from smart_nanogrid_gym.utils.config import data_files_directory_path


class PVSystemManager:
    def __init__(self, number_of_days_to_predict, time_interval):
        self.PREDICTION_DAY_PADDING = 1
        total_timesteps = int(24 / time_interval)
        self.padded_total_timesteps = total_timesteps * 2
        self.padded_number_of_prediction_days = number_of_days_to_predict + self.PREDICTION_DAY_PADDING
        padded_experiment_length = total_timesteps * self.padded_number_of_prediction_days

        self.pv_system = PVSystem(length=2.279, width=1.134, depth=20, total_dimensions=2.279*1.134*20, efficiency=0.21)
        self.solar_irradiance = self.load_solar_irradiance_per_timestep(padded_experiment_length, time_interval)
        self.solar_irradiance_2 = self.reshape_solar_irradiance_per_days_of_experiment(number_of_days_to_predict)
        self.available_solar_energy = self.calculate_available_solar_energy()

    def load_solar_irradiance_per_timestep(self, padded_experiment_length, time_interval):
        solar_irradiance_forecast = self.load_raw_irradiance_data_from_mat_file('solar_irradiance.mat')
        solar_irradiance = self.calculate_solar_irradiance_mean(solar_irradiance_forecast, padded_experiment_length,
                                                                time_interval)
        return solar_irradiance

    def load_raw_irradiance_data_from_mat_file(self, irradiance_data_filename):
        irradiance_data = loadmat(data_files_directory_path + irradiance_data_filename)
        return irradiance_data['irradiance']

    def calculate_solar_irradiance_mean(self, irradiance_forecast, padded_experiment_length, time_interval):
        timestep_in_minutes = 60 * time_interval
        solar_irradiance = zeros([1, padded_experiment_length])
        experiment_length_in_minutes = timestep_in_minutes * padded_experiment_length

        count = 0
        for interval in range(0, experiment_length_in_minutes, timestep_in_minutes):
            next_interval = interval + timestep_in_minutes
            solar_irradiance[0, count] = (mean(irradiance_forecast[interval: next_interval]))
            count = count + 1
        return solar_irradiance

    def reshape_solar_irradiance_per_days_of_experiment(self, number_of_days_to_predict):
        if number_of_days_to_predict == 1:
            reshaped_solar_irradiance = reshape(
                self.solar_irradiance,
                (number_of_days_to_predict, self.padded_total_timesteps)
            )
        else:
            reshaped_solar_irradiance = np.reshape(self.solar_irradiance.flatten(),
                                                   (self.padded_number_of_prediction_days, -1))
            repeated_solar_irradiance = np.repeat(reshaped_solar_irradiance, 2, axis=0)
            mask = np.ones(repeated_solar_irradiance.shape, dtype=bool)
            mask[[0, -1]] = False
            solar_irradiance_with_repeated_middle = repeated_solar_irradiance[mask]

            reshaped_solar_irradiance = reshape(
                solar_irradiance_with_repeated_middle,
                (number_of_days_to_predict, self.padded_total_timesteps)
            )

        return reshaped_solar_irradiance

    def calculate_available_solar_energy(self):
        scaling_pv = self.calculate_pv_scaling_coefficient()
        scaling_sol = 1.5
        return self.solar_irradiance * scaling_pv * scaling_sol

    def calculate_pv_scaling_coefficient(self):
        return self.pv_system.total_dimensions * self.pv_system.efficiency / 1000

    def get_available_solar_energy(self):
        return self.available_solar_energy

    def get_solar_radiation(self):
        return self.solar_irradiance_2

    def get_available_solar_produced_power(self, time_interval):
        return self.available_solar_energy / time_interval
