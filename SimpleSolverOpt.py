import numpy as np
from VisualizeState import VisualizeState
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from functools import partial

def SimpleSolver(eval_f, x_start, p, eval_u, NumIter, w=1, visualize=True, gif_file_name="State_visualization.gif"):
    """
    an extremely basic iterative algorithm 
    that can surprisingly potentially solve very different problems
    (although sometimes not reliably or efficiently):
    e.g. nonlinear problems, i.e. finding vector x such that f(x,p,u)=0
    e.g. steady state step response x of nonlinear ODEs dx/dt = f(x,p,u)%     
        and of some Differential Algebraic Equations (ADEs)
            0 = f_a(x,p,u)
        dx/dt = f_d(x,p,u)
        simply setting f(x,p,u) = [f_a(x,p,u) f_d(x,p,u)]'
    e.g. can also produce a possibly rough approximation
        of the dynamics leading to steady state
    e.g. and it can even solve optimization problems 
        it is basically steepest descent when f is grad(Loss)

    x_start        is the initial guess for the iteration
    p         is a structure containing all parameters needed to evaluate f( )
    eval_u    contains values of the input sources 
    eval_f    is a text string with the name of the function that evaluates f(x,p,u) 
    NumIter   is the desired number of iterations 
    visualize = True (optional) shows intermediate results
    gif_file_name = "State_visualization.gif" (optional) name of the gif file to save the visualization

    w  is is an optional (yet important) scalar parameter 
    w=1 is used as default when w is not given
    but order of magnitude smaller values (or negative) 
    may be necessary to prevent wild numerical instabilities,
    at the expense of progressively slower computation times. 
    Intepretations can be given for w when solving:
        nonlinear systems, w controls the convergence of the sequence
        ODEs and ADEs, w can be interpreted as "timestep size"
        optimization problems, w can be interpreted as "learning rate"

    EXAMPLE
    [X,t] = SimpleSolver(eval_f, x_start, p, eval_u, NumIter, w, visualize)
    """

    print("started running a simple solver, please wait...")

    # Input validation and preprocessing
    NumIter = int(NumIter)
    x_start = np.asarray(x_start).flatten()
    n_states = len(x_start)
    
    # Pre-allocate arrays
    X = np.empty((n_states, NumIter + 1))
    t = np.arange(NumIter + 1) * w  # Vectorized time array generation
    
    # Set initial condition
    X[:, 0] = x_start
    
    # Main iteration loop
    for n in range(NumIter):
        u = eval_u(t[n])
        f = eval_f(X[:, n:n+1], p, u)  # Pass as column vector without reshaping overhead
        X[:, n+1] = X[:, n] + w * f.ravel()  # Use ravel() instead of reshape
    
    print("Solver completed.")
    return X, t