from math import *
import random
import numpy as np
import matplotlib.pyplot as plt
import analysis as ana


def travel_time(t1):
    """
    Returns a value for a travel time (agent leaves patch to a different patch).
    For now, not stochastic and just returns the input parameter.
    """
    return t1


def revisit_time(t2):
    """
    Returns a value for a revisit time (agent leaves patch and comes back to it.
    For now, not stochastic and just returns the input parameter.
    """
    return t2


def feeding_rate(time, time_constant):
    """
    Returns feeding rate after spending time in a patch.
    For now, just returns an exponential.
    """
    return exp(-time/time_constant)


def is_leave(f_rate, opt_leaving_rate, time, minimal_time, error_rate=0):
    """
    Decision rule for leaving a patch, with an error_rate.
    Will return True when feeding rate is below optimal once time has exceeded minimal_time.
    """
    p = random.uniform(0, 1)
    if p > error_rate:
        if time > minimal_time:
            if f_rate < opt_leaving_rate:
                return True
            else:
                return False
        else:
            return False


def is_revisit(revisit_probability):
    """
    Determine if a transit is a revisit.
    Will return True with probability revisit_probability.
    """
    p = random.uniform(0, 1)
    if p < revisit_probability:
        return True
    else:
        return False


def sim_mvt(n_time_steps, t1, t2, t_min, optimal_leaving_rate, revisit_probability, time_constant):
    """
    Simulation for an MVT agent leaving patches when feeding rate reaches optimal_leaving_rate,
    and adding a certain revisit probability.

    """
    t = 0
    list_of_durations = []
    feeding_rate_list = -1 * np.ones(n_time_steps + 1)
    while t < n_time_steps:

        # Travel to the next patch
        travel = travel_time(t1)
        t += travel  # travel to next patch
        # Add 0 feeding_rate for the whole travel time
        feeding_rate_list[t - travel:t] = [0 for _ in range(travel - abs(min(0, n_time_steps - t + 1)))]

        # Initialize visit and feeding rate
        duration_of_visit = 0
        f_rate = feeding_rate(duration_of_visit, time_constant)  # update feeding rate

        # While the agent stays in the patch
        while t + duration_of_visit < n_time_steps and not is_leave(f_rate, optimal_leaving_rate, duration_of_visit, t_min):
            f_rate = feeding_rate(duration_of_visit, time_constant)  # update feeding rate
            feeding_rate_list[t + duration_of_visit] = f_rate  # add it to the list
            duration_of_visit += 1  # update duration of visit

        # Update global time counter
        t += duration_of_visit
        # Update list of visits
        list_of_durations.append(duration_of_visit)

        # Visit is over, now will there be a revisit?
        # If there is a revisit
        while t < n_time_steps and is_revisit(revisit_probability):
            # Travel to same patch
            travel = revisit_time(t2)
            t += travel

            # Add it to feeding rate list. Weird shit is because travel < travel when t reaches n_time_steps
            feeding_rate_list[t - travel:t] = [0 for _ in range(travel - abs(min(0, n_time_steps - t + 1)))]

            # Revisit it
            duration_of_revisit = 0
            while t + duration_of_revisit < n_time_steps and not is_leave(f_rate, optimal_leaving_rate, duration_of_revisit, t_min):
                duration_of_revisit += 1
                f_rate = feeding_rate(duration_of_visit + duration_of_revisit, time_constant)
                feeding_rate_list[t + duration_of_revisit] = f_rate  # add it to the list

    return list_of_durations, feeding_rate_list, np.mean(feeding_rate_list), np.mean(list_of_durations)


def plot_f_rate(f_rate_list):
    plt.plot(range(len(f_rate_list)), f_rate_list)
    plt.show()


def opt_mvt(condition_name, nb_of_points, nb_of_time_steps, max_leaving_rate, parameter_list, is_plot=True, is_print=True):
    """
    Will compute the average feeding rate of the environment for a range of thresholds to leave a patch.
    """
    avg_feeding_rate_list = []
    avg_visit_length_list = []
    t1, t2, t_min, p_rev, constant = parameter_list
    leaving_rate_list = np.linspace(0.0001, max_leaving_rate, nb_of_points)

    # Run simulations for all leaving_rates
    for leaving_rate in leaving_rate_list:
        _, _, avg_feeding_rate, avg_visit_length = sim_mvt(nb_of_time_steps, t1, t2, t_min, leaving_rate, p_rev, constant)
        avg_feeding_rate_list.append(avg_feeding_rate)
        avg_visit_length_list.append(avg_visit_length)

    # From there, compute the leaving rate where the maximum is reached
    opt_index = np.argmax(avg_feeding_rate_list)
    opt_leaving_rate = leaving_rate_list[opt_index]
    opt_avg_feeding_rate = avg_feeding_rate_list[opt_index]
    opt_avg_visit_length = avg_visit_length_list[opt_index]

    if is_plot:
        fig, ax1 = plt.subplots()
        fig.suptitle("Nb_of_time_steps: "+str(nb_of_time_steps)+" for condition "+condition_name+", tau ="+str(constant))
        ax1.set_xlabel("Leaving rate f*")
        ax1.set_ylabel("Average feeding rate <f>")
        ax1.plot(leaving_rate_list, avg_feeding_rate_list, color="orange")
        ax1.tick_params(axis='y', labelcolor="orange")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Average visit length")
        ax2.plot(leaving_rate_list, avg_visit_length_list, color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")

        # Vertical line to indicate our computed optimal
        ax1.axvline(opt_leaving_rate, color="blue", linestyle="dotted")
        # Write the corresponding visit length
        ax2.annotate("model="+str(np.round(opt_avg_visit_length, 1)), [1.1*opt_leaving_rate, 1.1*opt_avg_visit_length], color="blue")

        # x = y line: should cross the feeding rate curve at its peak (MVT result)
        ax1.plot(avg_feeding_rate_list, avg_feeding_rate_list, label="y=x curve", color="grey", linestyle="dashed")
        # Find the point where the x=y curve intersects our avg_feeding_rate curve
        index = np.min(np.argwhere(np.diff(np.sign(avg_feeding_rate_list - leaving_rate_list))).flatten())
        ax1.axvline(leaving_rate_list[index], color="grey", linestyle="dotted")
        ax2.annotate("mvt=" + str(np.round(avg_visit_length_list[index], 1)), [1.1*opt_leaving_rate, 1.3*opt_avg_visit_length])

        ax1.legend()
        plt.show()

    if is_print:
        print("Optimal is reached for a leaving rate of: ", opt_leaving_rate)
        print("Which corresponds to an average feeding rate of: ", opt_avg_feeding_rate)
        print("And an average visit length of: ", opt_avg_visit_length)

    return opt_leaving_rate


def print_opt_mvt(nb_of_points, nb_of_time_steps, parameter_list):
    t_travel, t_revisit, t_minimal, p_revisit, time_constant = parameter_list

    print("t_travel: ", t_travel)
    print("t_revisit: ", t_revisit)
    print("t_minimal: ", t_minimal)
    print("p_revisit: ", p_revisit)
    print("time constant: ", time_constant)

    optimal_rate = opt_mvt(nb_of_points, nb_of_time_steps, parameter_list, is_plot=False, is_print=True)

    print("-----")
    # Example of feeding rate evolution
    plot_f_rate(sim_mvt(10*t_travel, t_travel, t_revisit, t_minimal, optimal_rate, p_revisit, time_constant)[1])

    return optimal_rate


# TODO save those in a csv in analysis with a "generate_model_parameters" function
param_all = [303, 26, 0, 0.91, 92]
param_close02 = [81, 16, 0, 0.80, 92]
param_med02 = [413, 27, 0, 0.92, 92]
param_far02 = [1870, 34, 0, 0.94, 92]
param_cluster02 = [267, 24, 0, 0.89, 92]
param_close05 = [137, 20, 0, 0.91, 92]
param_med05 = [382, 29, 0, 0.95, 92]
param_far05 = [2038, 29, 0, 0.96, 92]
param_cluster05 = [236, 22, 0, 0.92, 92]

nb_of_leaving_rates_to_test = 100
nb_of_time_steps_per_leaving_rate = 500000

opt_mvt("param_close02", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 0.4, param_close02)
opt_mvt("param_med02", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 0.2, param_med02)
opt_mvt("param_far02", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 0.2, param_far02)
opt_mvt("param_close05", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 0.2, param_close05)
opt_mvt("param_med05", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 0.2, param_med05)
opt_mvt("param_far05", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 0.2, param_far05)
#print_opt_mvt(20, 1000, param_close05)
