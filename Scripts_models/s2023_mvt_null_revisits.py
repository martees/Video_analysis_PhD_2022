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


def mvt_avg_feeding_rate(t1, t2, leaving_threshold, revisit_probability, time_constant):
    """
    Compute numerically the average feeding rate given the parameters of the null mvt model, where revisits have a length of 0.
    """
    avg_duration_of_visits = exponential_visit_length(leaving_threshold, time_constant)
    avg_feeding_rate_in_a_patch = exponential_visit_feeding_rate(avg_duration_of_visits, time_constant)
    avg_nb_of_revisits = revisit_probability/(1-revisit_probability)
    return avg_duration_of_visits*avg_feeding_rate_in_a_patch / (avg_duration_of_visits + t1 + avg_nb_of_revisits*t2)


def mvt_avg_visit_length(leaving_threshold, revisit_probability, time_constant):
    """
    Compute numerically the average visit length given the parameters of the null mvt model, where revisits have a length of 0.
    """
    avg_duration_of_visits = exponential_visit_length(leaving_threshold, time_constant)
    avg_nb_of_revisits = revisit_probability/(1-revisit_probability)
    #return avg_duration_of_visits / (1 + avg_nb_of_revisits)
    return avg_duration_of_visits


def simulation_visit_length(leaving_threshold, time_constant, time_already_spent=0):
    """
    Will return the visit length for a given leaving_threshold, and a feeding rate that decays exponentially with a
    given time_constant ( f(t) = exp(-t/constant) ). Optionally, will take the time already spent in the patch as
    an argument, and remove that from output.
    """
    duration_of_visit = time_already_spent
    f_rate = feeding_rate(duration_of_visit, time_constant)
    while f_rate > leaving_threshold:
        duration_of_visit += 1
        f_rate = feeding_rate(duration_of_visit, time_constant)
    return duration_of_visit - time_already_spent


def exponential_visit_length(leaving_threshold, time_constant, time_already_spent=0):
    """
    Will return the visit length for a given leaving_threshold, and a feeding rate that decays exponentially with a
    given time_constant ( f(t) = exp(-t/constant) ). Optionally, will take the time already spent in the patch as
    an argument, and remove that from output.
    """
    # Return max between answer and 0, because the calculus can give negative visit durations
    return max(0, int(-time_constant*log(leaving_threshold) - time_already_spent))


def exponential_visit_feeding_rate(duration_of_visit, time_constant, time_already_spent=0):
    """
    Will return the avg amount of food gathered per time step over a visit in a patch, with exponential decay
    f(t) = exp(-t/constant). Optionally, will take the time already spent in the patch as an argument, and remove
    that from output.
    """
    return time_constant*(exp(-time_already_spent/time_constant)-exp(-duration_of_visit/time_constant))/max(1, duration_of_visit)


def simulation(n_time_steps, t1, t2, t_min, leaving_threshold, revisit_probability, time_constant):
    """
    Simulation for an agent leaving patches when feeding rate reaches leaving_threshold,
    and adding a certain revisit probability.
    """
    t = 0
    list_of_durations = []
    feeding_rate_list = -1 * np.ones(n_time_steps + 1)
    duration_of_first_visits = simulation_visit_length(leaving_threshold, time_constant)
    while t < n_time_steps:

        # Travel to the next patch
        travel = travel_time(t1)
        # Update global time counter
        t += travel  # travel to next patch
        # Add 0 feeding_rate for the whole travel time
        feeding_rate_list[t - travel:t] = [0 for _ in range(travel - abs(min(0, n_time_steps - t + 1)))]

        # Visit it
        duration_of_visit = duration_of_first_visits
        # Update global time counter
        t += duration_of_visit
        # Add average feeding rate during visit * time of visit to the feeding_rate list (NO DYNAMICS)
        avg_f_rate = exponential_visit_feeding_rate(duration_of_visit, time_constant)
        feeding_rate_list[t - duration_of_visit:t] = [avg_f_rate for _ in range(duration_of_visit - abs(min(0, n_time_steps - t + 1)))]
        # Update list of visits
        list_of_durations.append(duration_of_visit)

        # Visit is over, now will there be a revisit?
        # If there is a revisit
        time_already_spent = duration_of_visit
        while t < n_time_steps and is_revisit(revisit_probability):
            # Travel to same patch
            travel = revisit_time(t2)  # revisit travel
            # Update global time counter
            t += travel
            # Add it to feeding rate list. Weird shit is because travel < travel when t reaches n_time_steps
            feeding_rate_list[t - travel:t] = [0 for _ in range(travel - abs(min(0, n_time_steps - t + 1)))]

            # Revisit it
            duration_of_revisit = exponential_visit_length(leaving_threshold, time_constant, time_already_spent=time_already_spent)
            # Update global time counter
            t += duration_of_revisit
            # Add average feeding rate during revisit * time of visit to the feeding_rate list (NO DYNAMICS)
            avg_f_rate = exponential_visit_feeding_rate(duration_of_revisit, time_constant, time_already_spent=time_already_spent)
            feeding_rate_list[t - duration_of_revisit:t] = [avg_f_rate for _ in range(duration_of_revisit - abs(min(0, n_time_steps - t + 1)))]

            # Update list of visits
            list_of_durations.append(duration_of_revisit)
            # Update time already spent in patch
            time_already_spent += duration_of_revisit

    return list_of_durations, feeding_rate_list, np.mean(feeding_rate_list), np.mean(list_of_durations)


def plot_f_rate(f_rate_list):
    plt.plot(range(len(f_rate_list)), f_rate_list)
    plt.show()


def opt_mvt(condition_name, nb_of_points, nb_of_time_steps, max_leaving_rate, parameter_list, is_plot=True, is_print=True):
    """
    Will compute the average feeding rate of the environment for a range of thresholds to leave a patch.
    """
    sim_avg_feeding_rate_list = []
    sim_avg_visit_length_list = []
    mvt_feeding_rate_list = []
    mvt_visit_length_list = []
    t1, t2, t_min, p_rev, constant = parameter_list
    leaving_rate_list = np.linspace(0.001, max_leaving_rate, nb_of_points)

    # Run simulations for all leaving_rates
    for leaving_rate in leaving_rate_list:
        _, _, avg_feeding_rate, avg_visit_length = simulation(nb_of_time_steps, t1, t2, t_min, leaving_rate, p_rev, constant)
        sim_avg_feeding_rate_list.append(avg_feeding_rate)
        sim_avg_visit_length_list.append(avg_visit_length)
        mvt_feeding_rate_list.append(mvt_avg_feeding_rate(t1, t2, leaving_rate, p_rev, constant))
        mvt_visit_length_list.append(mvt_avg_visit_length(leaving_rate, p_rev, constant))

    # From there, compute the leaving rate and visit length where the maximum is reached
    # FOR SIMULATED NUMBERS
    opt_index = np.argmax(sim_avg_feeding_rate_list)  # Find index of maximal average feeding rate
    opt_avg_feeding_rate = sim_avg_feeding_rate_list[opt_index]  # Store how much it is
    opt_leaving_rate = leaving_rate_list[opt_index]  # Find corresponding leaving rate
    opt_avg_visit_length = sim_avg_visit_length_list[opt_index]  # Find corresponding visit length
    # FOR ANALYTICAL SOLUTION
    opt_index_mvt = np.argmax(mvt_feeding_rate_list)  # Find index of maximal average feeding rate
    opt_avg_feeding_rate_mvt = sim_avg_feeding_rate_list[opt_index_mvt]  # Store how much it is
    opt_leaving_rate_mvt = leaving_rate_list[opt_index_mvt]  # Find corresponding leaving rate
    opt_avg_visit_length_mvt = mvt_visit_length_list[opt_index_mvt]  # Find corresponding visit length

    if is_plot:

        # Plot average feeding_rate for every leaving rate (both in simulations and analytical solution)
        fig, ax1 = plt.subplots()

        # General parameters for the figure
        fig.set_size_inches(9, 7)
        fig.suptitle("Nb_of_time_steps: "+str(nb_of_time_steps)+" for condition "+condition_name+", tau ="+str(constant))

        # Plotting the average
        ax1.set_xlabel("Leaving rate f*")
        ax1.set_ylabel("Average feeding rate <f>")
        ax1.plot(leaving_rate_list, sim_avg_feeding_rate_list, color="orange", label="Simulated average feeding rate")
        ax1.plot(leaving_rate_list, mvt_feeding_rate_list, color="red", label="Analytic average feeding rate")
        ax1.tick_params(axis='y', labelcolor="orange")

        # Plot the list of average visit lengths
        ax2 = ax1.twinx()
        ax2.set_ylabel("Average visit length")
        ax2.plot(leaving_rate_list, sim_avg_visit_length_list, color="blue", label="Average visit length")
        ax2.plot(leaving_rate_list, mvt_visit_length_list, color="green", label="MVT visit length")
        ax2.tick_params(axis='y', labelcolor="blue")

        # Vertical line to indicate the simulation optimal (leaving rate that yields the maximal average feeding rate)
        ax1.axvline(opt_leaving_rate, color="orange", linestyle="dotted", label="Simulated maximum")
        # Write the corresponding visit length
        ax2.annotate("sim_max="+str(np.round(opt_avg_visit_length, 1)), [1.1*opt_leaving_rate_mvt, 4*opt_avg_visit_length_mvt], color="orange")

        # Vertical line to indicate the analytical optimal (leaving rate that yields the maximal average feeding rate)
        ax1.axvline(opt_leaving_rate_mvt, color="red", linestyle="dotted", label="Analytical maximum")
        # Write the corresponding visit length
        ax2.annotate("math_max="+str(np.round(opt_avg_visit_length_mvt, 1)), [1.1*opt_leaving_rate_mvt, 3*opt_avg_visit_length_mvt], color="red")

        # Plot the x = y line
        ax1.plot(sim_avg_feeding_rate_list, sim_avg_feeding_rate_list, label="y=x curve", color="grey", linestyle="dashed")

        # Vertical line for the intercept between SIMULATED avg_feeding_rate and x=y line (MVT-like simulated optimal)
        # Find the point where the x=y curve intersects our avg_feeding_rate curve
        # In theory, the x = y line should cross the feeding rate curve at its peak (MVT result)
        index = np.min(np.argwhere(np.diff(np.sign(sim_avg_feeding_rate_list - leaving_rate_list))).flatten())
        ax1.axvline(leaving_rate_list[index], color="orange", linestyle="dashdot", label="Simulation x=y-intercept")
        ax2.annotate("sim_intercept=" + str(np.round(sim_avg_visit_length_list[index], 1)), [1.1*opt_leaving_rate_mvt, 2*opt_avg_visit_length_mvt], color="orange")

        # Vertical line for the intercept between ANALYTICAL avg_feeding_rate and x=y line (MVT theoretical optimal)
        # Find the point where the x=y curve intersects our avg_feeding_rate curve
        index = np.min(np.argwhere(np.diff(np.sign(mvt_feeding_rate_list - leaving_rate_list))).flatten())
        ax1.axvline(leaving_rate_list[index], color="red", linestyle="dashdot", label="Analytical x=y-intercept")
        ax2.annotate("math_intercept=" + str(np.round(mvt_visit_length_list[index], 1)), [1.1*opt_leaving_rate_mvt, 1*opt_avg_visit_length_mvt], color="red")

        ax1.legend(loc='upper left')
        plt.show()

    if is_print:
        print("SIMULATION RESULTS")
        print("Optimal is reached for a leaving rate of: ", opt_leaving_rate)
        print("Which corresponds to an average feeding rate of: ", opt_avg_feeding_rate)
        print("And an average visit length of: ", opt_avg_visit_length)
        print("ANALYTICAL RESULTS")
        print("Optimal is reached for a leaving rate of: ", opt_leaving_rate_mvt)
        print("Which corresponds to an average feeding rate of: ", opt_avg_feeding_rate_mvt)
        print("And an average visit length of: ", opt_avg_visit_length_mvt)

    return opt_avg_visit_length_mvt


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
    plot_f_rate(simulation(10 * t_travel, t_travel, t_revisit, t_minimal, optimal_rate, p_revisit, time_constant)[1])

    return optimal_rate


# TODO save those in a csv in analysis with a "generate_model_parameters" function
tauu = 2000

#[t1, t2, t_min, revisit_probability, time_constant]
param_all = [303, 26, 0, 0.91, tauu]
param_close02 = [81, 16, 0, 0.80, tauu]
param_med02 = [413, 27, 0, 0.92, tauu]
param_far02 = [1870, 34, 0, 0.94, tauu]
param_cluster02 = [267, 24, 0, 0.89, tauu]
param_close05 = [137, 20, 0, 0.91, tauu]
param_med05 = [382, 29, 0, 0.95, tauu]
param_far05 = [2038, 29, 0, 0.96, tauu]
param_cluster05 = [236, 22, 0, 0.92, tauu]

nb_of_leaving_rates_to_test = 100
nb_of_time_steps_per_leaving_rate = 1000000

#opt_mvt("param_close02", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 1, param_close02)
#opt_mvt("param_med02", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 0.5, param_med02)
#opt_mvt("param_far02", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 0.5, param_far02)
#opt_mvt("param_close05", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 0.5, param_close05)
#opt_mvt("param_med05", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 0.5, param_med05)
#opt_mvt("param_far05", nb_of_leaving_rates_to_test, nb_of_time_steps_per_leaving_rate, 0.5, param_far05)
#print_opt_mvt(20, 1000, param_close05)
