from numpy import mean, reshape, zeros
from scipy.io import loadmat

from smart_nanogrid_gym.utils.pv_system import PVSystem
from smart_nanogrid_gym.utils.config import data_files_directory_path


class PVSystemManager:
    def __init__(self, experiment_length_in_days, number_of_days_ahead_for_prediction):
        experiment_length_in_hours = 24 * (experiment_length_in_days + number_of_days_ahead_for_prediction)

        self.pv_system = PVSystem(length=2.279, width=1.134, depth=20, total_dimensions=2.279*1.134*20, efficiency=0.21)
        self.solar_irradiance = self.load_solar_irradiance_per_hour(experiment_length_in_hours)
        self.solar_radiation = self.calculate_available_solar_radiation(experiment_length_in_days)
        self.available_solar_energy = self.calculate_available_solar_energy()

    def load_solar_irradiance_per_hour(self, experiment_length_in_hours, timestep_in_minutes=60):
        solar_irradiance_forecast = self.load_raw_irradiance_data_from_mat_file('solar_irradiance.mat')
        solar_irradiance = self.calculate_solar_irradiance_mean_per_timestep(solar_irradiance_forecast,
                                                                             timestep_in_minutes,
                                                                             experiment_length_in_hours)
        return solar_irradiance

    def load_raw_irradiance_data_from_mat_file(self, irradiance_data_filename):
        irradiance_data = loadmat(data_files_directory_path + irradiance_data_filename)
        return irradiance_data['irradiance']

    def calculate_solar_irradiance_mean_per_timestep(self, irradiance_forecast, timestep_in_minutes,
                                                     experiment_length_in_hours):
        solar_irradiance = zeros([experiment_length_in_hours, 1])
        experiment_length_in_minutes = timestep_in_minutes * experiment_length_in_hours

        count = 0
        for time_interval in range(0, experiment_length_in_minutes, timestep_in_minutes):
            next_time_interval = time_interval + timestep_in_minutes
            solar_irradiance[count, 0] = (mean(irradiance_forecast[time_interval: next_time_interval]))
            count = count + 1
        return solar_irradiance

    def calculate_available_solar_radiation(self, experiment_length_in_days, timestep_in_minutes=60):
        experiment_day_length_in_timesteps = int(60 / timestep_in_minutes) * 24

        reshaped_solar_irradiance = reshape(
            self.solar_irradiance,
            (experiment_length_in_days, experiment_day_length_in_timesteps * 2)
        )

        return reshaped_solar_irradiance

    def calculate_available_solar_energy(self):
        scaling_pv = self.calculate_pv_scaling_coefficient()
        scaling_sol = 1.5
        return self.solar_irradiance * scaling_pv * scaling_sol

    def calculate_pv_scaling_coefficient(self):
        return self.pv_system.total_dimensions * self.pv_system.efficiency / 1000

    def get_solar_energy(self):
        return {
            'Available solar energy': self.available_solar_energy,
            'Solar radiation': self.solar_radiation
        }
