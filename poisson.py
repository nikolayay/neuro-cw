import random as rnd
from load import load_data
import numpy as np
from functools import reduce

from matplotlib import pyplot as plt


def get_spike_train(rate, big_t, tau_ref):

    if 1 <= rate*tau_ref:
        print("firing rate not possible given refractory period f/p")
        return []

    exp_rate = rate/(1-tau_ref*rate)

    spike_train = []

    t = rnd.expovariate(exp_rate)

    while t < big_t:
        spike_train.append(t)
        t += tau_ref+rnd.expovariate(exp_rate)

    return spike_train


# SI UNITS
Hz = 1.0
sec = 1.0
ms = 0.001

# VARIABLES
rate = 15.0 * Hz
tau_ref = 5 * ms

#
big_t = 5 * sec


def chunk_spikes_by_interval(spike_train, trial_len, interval):
    """
    ? spike train is represented as a list of timestamps when spikes occur
    ? interval is in ms
    1. new count every interval length
    """

    samples = int(trial_len / interval)

    k = 0
    counts = []
    for i in range(samples):
        count = 0
        while (k < len(spike_train) and spike_train[k] < interval * (i + 1)):
            count += 1
            k += 1
        counts.append(count)

    return counts


def get_fano_factor(spike_train, trial_len, intervals):
    """
    1. for each interval:
        1.1) chunk spike train into intervals
        1.2) count spikes in each interval
        1.3) apply the formula per interval
    2. output average
    """
    fanos = []

    for interval in intervals:
        spike_counts = chunk_spikes_by_interval(
            spike_train, trial_len, interval)
        var = np.var(spike_counts)
        mean = np.mean(spike_counts)
        fano = var / mean
        fanos.append(fano)

    return np.mean(fanos)


def get_coef_of_var(spike_train):
    """
    1. work out time differences between counts across the whole train
    2. apply the formula
    """

    differences = np.diff(spike_train)

    return np.std(differences) / np.mean(differences)


def spikes_to_spike_train(spikes, sample_rate):
    count = 0
    spike_train = []
    for i in range(len(spikes)):
        if (spikes[i] == 1):
            spike_train.append(sample_rate * i)

    return spike_train


def count_spikes_in_range(spikes, start, step):

    count = 0
    current_index = start
    current_step = 0

    while (current_index < len(spikes) and current_step < step):
        if (spikes[current_index] == 1):
            count += 1

        current_index += 1
        current_step += 1

    return count / (step + 1)


def plot_autocorrelogram(spikes, steps):
    """
    1. go 50 each way
    """
    xs = []

    steps = int(steps)
    print(len(spikes))

    for i in range(len(spikes)):
        correlation = np.correlate(
            spikes[i:steps], spikes[i:steps], mode="same")
        print(np.mean(correlation), correlation)
        xs.append(np.mean(correlation))

    return xs


def get_STA(spikes, stimulus, window_size):
    """
    1. For every spike, add the stimulus values surrounding the spike into the spike-triggered average array. 
        For example, a spike at 10.12s, the stimulus at 10.02s goes into the -0.10s bin, 
        the stimulus at 10.03s goes into the -0.09s bin.
        Do this for a given time before and after the spike. Repeat for all spikes.
    2. Once all the values are added into the array, divide by number of spikes to get spike-triggered average.
    """
    arr = np.zeros(50)
    spike_counter = np.sum(spikes)

    for i in range(50, len(spikes)):
        if (spikes[i] == 1):
            for t in range(50):
                arr[t] += stimulus[i - t]

    return list(map(lambda x: x / spike_counter, arr))


def plot_STA(spike_train):
    return None


# ! QUESTION 1
# ? calculating the Fano factor and Coefficient of variation for simulated spike train data.
q1_rate = 35 * Hz
q1_tau_ref = 5 * ms  # refactory period
q1_trial_len = 1000 * sec

q1_spike_train = get_spike_train(q1_rate, q1_trial_len, q1_tau_ref)

q1_intervals = [10 * ms, 50 * ms, 100 * ms]
q1_Fano = get_fano_factor(q1_spike_train, q1_trial_len, q1_intervals)
q1_Coef_of_var = get_coef_of_var(q1_spike_train)

# ! QUESTION 1 - ANSWERS
print(f'ANSWERS FOR QUESTION 1')
# print("WITH SPIKE TRAIN: %s\n" % (spike_train))
print(f'FANO FACTOR: {q1_Fano}')
print(f'COEFFICIENT OF VARIATION: {q1_Coef_of_var}\n')


# ! QUESTION 2
# ? calculating the Fano factor and Coefficient of variation for real spike train data.
q2_spikes = load_data("rho.dat", int)
q2_rate = 500 * Hz
q2_tau_ref = 2 * ms
q2_trial_len = 20 * 60 * sec

q2_spike_train = spikes_to_spike_train(q2_spikes, q2_tau_ref)

# same as q1
q2_intervals = [10 * ms, 50 * ms, 100 * ms]
q2_Fano = get_fano_factor(q2_spike_train, q2_trial_len, q2_intervals)
q2_Coef_of_var = get_coef_of_var(q2_spike_train)


# ! QUESTION 2 - ANSWERS
print(f'ANSWERS FOR QUESTION 2')
# print("WITH SPIKE TRAIN: %s\n" % (spike_train))
print(f'FANO FACTOR: {q2_Fano}')
print(f'COEFFICIENT OF VARIATION: {q2_Coef_of_var}\n')


# # ! QUESTION 3
# # ? for same spike train in the rho vector, calculate and plot the autocorrelogram over a range.
q3_spikes = load_data("rho.dat", int)
q3_interval = 100 * ms
q3_tau_ref = 2 * ms

q3_steps = q3_interval / q3_tau_ref

print(q3_spikes)




# # ! QUESTION 4
# # ? calculate and plot the spike triggered average (STA) over a window
q4_window_size = 100 * ms
q4_stimulus = load_data("stim.dat", float)
q4_spikes = load_data("rho.dat", int)


q4_STA = get_STA(q4_spikes, q4_stimulus, q4_window_size)

plt.plot(np.linspace(-100, 0, len(q4_STA)), q4_STA[::-1])
plt.bar(np.linspace(-100, 0, len(q4_STA)), q4_STA[::-1])

plt.show()

# spike_train = get_spike_train(rate, big_t, tau_ref)
# print(len(spike_train)/big_t)
# print(spike_train)
