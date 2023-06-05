from math import *
import random
import numpy as np
import matplotlib.pyplot as plt
import analysis as ana


def travel_time():
    """
    Returns a value for a travel time.
    """
    return t1


def revisit_time():
    return t2


def feeding_rate(time):
    return exp(-time)


def is_leave_mvt(f_rate, opt_leaving_rate):
    if f_rate > opt_leaving_rate:
        return True
    else:
        return False


def is_leave_revisits(time, f_rate, minimal_time, optimal_leaving_rate):
    if time > minimal_time:
        if f_rate > optimal_leaving_rate:
            return True
        else:
            return False
    else:
        return False


def is_revisit(revisit_probability):
    p = random.uniform(0, 1)
    if p < revisit_probability:
        return True
    else:
        return False


def sim_mvt(n_time_steps, t1, t2, optimal_leaving_rate, revisit_probability):
    t = 0
    list_of_durations = []
    feeding_rate_list = []
    while t < n_time_steps:
        travel = travel_time()
        t = t + travel  # travel to next patch
        duration_of_visit = 0
        f_rate = feeding_rate(duration_of_visit)
        feeding_rate_list = feeding_rate_list + [0 for i in range(travel)]
        while is_leave_mvt(f_rate, optimal_leaving_rate) and t < n_time_steps:
            duration_of_visit += 1
            f_rate = feeding_rate(duration_of_visit)
            feeding_rate_list.append(f_rate)
        t = t+duration_of_visit
        list_of_durations.append(duration_of_visit)
        while is_revisit(revisit_probability) and t < n_time_steps:
            travel = revisit_time()
            t = t+travel
            feeding_rate_list = feeding_rate_list + [0 for i in range(travel)]
    return list_of_durations, feeding_rate_list, np.sum(feeding_rate_list)/t


def plot_f_rate(f_rate_list):
    plt.plot(range(len(f_rate_list)), f_rate_list)
    plt.show()


def opt_mvt(convergence_limit, t1, t2, p_revisit):
    current_rate = 1
    previous_rate = 0
    while abs(previous_rate - current_rate) > convergence_limit:
        avg_feeding_rate = sim_mvt(10000, t1, t2, current_rate, p_revisit)[2]
        previous_rate = current_rate
        current_rate = avg_feeding_rate
        print(current_rate)
    return current_rate


list_of_visits_mvt, feeding_rate_list_mvt, avg_feeding_rate_mvt = sim_mvt(1000, 20, 15, 0.4, 0.1)
opt_mvt(0.0000001, 10, 10, 0.1)
