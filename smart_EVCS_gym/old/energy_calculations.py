import numpy as np
import scipy.io


def get_price_day(current_price_model):
    # high_tariff = 0.028 + 0.148933
    # low_tariff = 0.013333 + 0.087613
    price_day = []
    # if current_price_model == 0:
    #     price_day = np.array([low_tariff, low_tariff, low_tariff, low_tariff, low_tariff, low_tariff, low_tariff,
    #                           high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff,
    #                           high_tariff, high_tariff, high_tariff, high_tariff, high_tariff, high_tariff,
    #                           low_tariff, low_tariff, low_tariff, low_tariff])
    if current_price_model == 1:
        price_day = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                              0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    elif current_price_model == 2:
        price_day = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.08, 0.06,
                              0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05])
    elif current_price_model == 3:
        price_day = np.array([0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080,
                              0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070])
    elif current_price_model == 4:
        price_day = np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                              0.1, 0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1])
    elif current_price_model == 5:
        price_day[1, :] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                           0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
        price_day[2, :] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.08, 0.06, 0.05,
                           0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05]
        price_day[3, :] = [0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080,
                           0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070]
        price_day[4, :] = [0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                           0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1]

    price_day = np.concatenate([price_day, price_day], axis=0)
    return price_day


def get_energy_price(price_day, experiment_length_in_days):
    price = np.zeros((experiment_length_in_days, 2 * 24))
    for day in range(0, experiment_length_in_days):
        price[day, :] = price_day
    return price


def calculate_solar_irradiance_mean_per_timestep(timestep_in_minutes, experiment_length_in_hours, file_directory_path):
    solar_irradiance_forecast = load_irradiance_data(file_directory_path, 'solar_irradiance.mat')

    solar_irradiance = np.zeros([experiment_length_in_hours, 1])
    experiment_length_in_minutes = timestep_in_minutes * experiment_length_in_hours

    count = 0
    for time_interval in range(0, experiment_length_in_minutes, timestep_in_minutes):
        next_time_interval = time_interval + timestep_in_minutes
        solar_irradiance[count, 0] = (np.mean(solar_irradiance_forecast[time_interval: next_time_interval]))
        count = count + 1
    return solar_irradiance


def calculate_pv_scaling_coefficient(pv_system_total_dimensions, pv_system_efficiency):
    return pv_system_total_dimensions * pv_system_efficiency / 1000


def calculate_available_solar_energy(solar_irradiance, pv_system_total_dimensions, pv_system_efficiency):
    scaling_pv = calculate_pv_scaling_coefficient(pv_system_total_dimensions, pv_system_efficiency)
    scaling_sol = 1.5
    return solar_irradiance * scaling_pv * scaling_sol


def calculate_available_solar_radiation(solar_irradiance, experiment_length_in_days, timestep_in_minutes):
    experiment_day_length_in_timesteps = int(60 / timestep_in_minutes) * 24

    reshaped_solar_irradiance = np.reshape(
        solar_irradiance,
        (experiment_length_in_days, experiment_day_length_in_timesteps * 2)
    )

    return reshaped_solar_irradiance


def load_irradiance_data(file_directory_path, irradiance_data_filename):
    irradiance_data = scipy.io.loadmat(file_directory_path + irradiance_data_filename)
    return irradiance_data['irradiance']


def get_energy(experiment_length_in_days, current_price_model, pv_system_available, file_directory_path,
               pv_system_total_dimensions, pv_system_efficiency, number_of_days_ahead_for_prediction):
    timestep_in_minutes = 60
    experiment_length_in_hours = 24 * (experiment_length_in_days + number_of_days_ahead_for_prediction)

    solar_irradiance = calculate_solar_irradiance_mean_per_timestep(timestep_in_minutes, experiment_length_in_hours,
                                                                    file_directory_path)

    solar_radiation = calculate_available_solar_radiation(solar_irradiance, experiment_length_in_days,
                                                          timestep_in_minutes)

    if pv_system_available:
        available_solar_energy = calculate_available_solar_energy(solar_irradiance, pv_system_total_dimensions,
                                                                  pv_system_efficiency)
    else:
        available_solar_energy = np.zeros(np.shape(solar_irradiance))

    price_day = get_price_day(current_price_model)
    energy_price = get_energy_price(price_day, experiment_length_in_days)
    consumed = np.zeros(np.shape(available_solar_energy))

    return {
        'Consumed': consumed,
        'Available renewable': available_solar_energy,
        'Price': energy_price,
        'Solar radiation': solar_radiation
    }
