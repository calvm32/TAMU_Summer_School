import numpy as np

def compute_percent_relative_error(coarse_solution, fine_solution):
    """
    Compute the percent relative error between two solutions.

    Parameters
    ----------
    coarse_solution : list
        List of coarse solution vectors.
        coarse_solution[n] is the coarse solution vector at time tau * n.
    fine_solution : list
        List of fine solution vectors.
        fine_solution[n] is the fine solution vector at time tau / 2 * n.
        Therefore, fine_solution[2*n] is a better approximation than coarse_solution[n] at the same solution time.

    Returns
    -------
    errors : list
        errors[n] is the percent relative error between coarse_solution[n] and fine_solution[2*n].
    """
    fine_solution_at_coarse_times = fine_solution[::2]
    errors = []
    for n in range(len(fine_solution_at_coarse_times)):
        percent_relative_error = np.linalg.norm(fine_solution_at_coarse_times[n]-coarse_solution[n])/np.linalg.norm(fine_solution_at_coarse_times[n])*100
        errors.append(percent_relative_error)
    return errors