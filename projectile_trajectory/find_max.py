def find_max(f,a,b,tol=1e-12):
    """
    Find the maximum of a function f on an interval (a,b) using a binary search.
    The function f is assumed to be continuous and unimodal on (a,b), meaning that there is a unique c in (a,b) such that f is increasing on (a,c) and decreasing on (c,b).
    This algorithm finds the pair (c,f(c)) up to a specified tolerance by only using point evaluations of f.

    Parameters
    ----------
    f : function(scalar) -> scalar
        Function to find the maximum of.
    a : scalar
        Lower bound of the interval.
    b : scalar
        Upper bound of the interval.
    tol : scalar
        Tolerance for the algorithm.

    Returns
    -------
    c : scalar
        x coordinate of the maximum of f on (a,b).
    f(c) : scalar
        Maximum of f on (a,b).
    """
    x_0 = a 
    x_3 = b 
    while (x_3 - x_0) / (b - a) > tol:
        x_1 = x_0 + (x_3 - x_0)/3 
        x_2 = x_3 - (x_3 - x_0)/3 
        if f(x_1) > f(x_2):
            x_3 = x_2 
        else:
            x_0 = x_1 
    c = (x_0 + x_3)/2
    return c, f(c)