from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
import numpy as np

class solver:
    """
    Class to solve equation in form of:
        y = f(x, y)

    Args:
        x (list): list of x values
        y (list): list of y values
        equation (str): valid left hand side of equation
        init (dict): initial values dictionary in form of key:value, where
            key is symbol used in equation and value is initial value
            if None: initialize with 1's

    Example use:
        # define function
        >>> eq = "x*a*exp(y/b)"

        # for example, purposes x and y are defined analytically
        # normally those should be your experimental values
        >>> y = np.linspace(2,10,100)
        >>> eq_x = "y/(a*exp(y/b))"
        >>> x = y/(8*np.exp(y/4))

        # initialize solver with x, y and initial guesses for a and b
        >>> s = solver(x, y, eq,{"a":1, "b":2})

        # run train for 1 step to see how good is initial guess
        >>> s.train(1)
        >>> s.show_fit()

        # train model with small learning constant
        >>> s.train(1000, 0.01)
        >>> s.show_fit()

        # finalize training with larger learning constant
        >>> s.train(2000, 0.02)
        >>> s.show_fit()

        # display convergence
        >>> s.show_error()

        # get parameters in form of dictionary
        >>> params = s.get_params()
    """
    def __init__(self, x, y, equation, init=None):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.equation = equation
        self.init = init

        self.error_history = []

        pars = parse_expr(self.equation)
        self.symbols = list(pars.free_symbols)

        self.x_sym = Symbol("x")
        self.y_sym = Symbol("y")
        try:
            self.symbols.remove(self.x_sym)
            self.symbols.remove(self.y_sym)
        except KeyError:
            print("Missing x or y symbol in equation!\nEnter valid form.")
            return

        if len(self.symbols) == 0:
            print("For fitting process I need at least one free variable!")
            return
        # Create initial values
        if init is None:
            print("init dict empty, assuming 1 for all initial values.")
            self.params = dict(zip(self.symbols, [1]*len(self.symbols)))
        else:
            self.params = {}
            str_symbols = [ str(s) for s in self.symbols ]
            str_symbols.index("a")

            for key in self.init:
                if key in str_symbols:
                    i = str_symbols.index(key)
                    self.params[self.symbols[i]] = self.init[key]
                else:
                    print("Not needed symbol in init: {}".format(key))

            # check for uninitialized symbols in values
            for s in self.symbols:
                if s not in self.params:
                    print("Initializing {} with 1".format(s))
                    self.params[s] = 1

        print("Initialized with function in the form of:\n    y = {}".format(self.equation))
        self.error_fun_sym = parse_expr("({}-y)**2".format(self.equation))
        self.partial_diff_sym = []

        for part in self.symbols:
            self.partial_diff_sym.append(diff(self.error_fun_sym, part))

    def train(self, epohs, alpha=0.01):
        """
        Train fit with gradient descent

        Args:
            ephos (int): number of cycles to preform
            alpha (float): learing constant
                if too big solution will explode
                if too small converge will be slow
        """
        for i in range(epohs):
            partial_diff = []
            for part in self.partial_diff_sym:
                eq = part.subs(self.params)
                partial_diff_fun = lambdify((self.x_sym, self.y_sym), eq, np)

                partial_diff.append(np.mean(partial_diff_fun(self.x, self.y)))

            # update constants

            for i in range(len(self.symbols)):
                symbol = self.symbols[i]
                self.params[symbol] -= alpha*partial_diff[i]

            # evalueate error function
            error_fun = lambdify((self.x_sym, self.y_sym), self.error_fun_sym.subs(self.params), np)
            self.error_history.append(np.mean(error_fun(self.x, self.y)))


    def get_params(self):
        """
        Get fitted parameters
        """
        return self.params

    def show_error(self):
        """
        Display error function value over all epochs
        """
        plt.plot(self.error_history)
        plt.show()

    def show_fit(self):
        """
        Display comparison between fit and experimental values
        """
        plt.plot(self.x, self.y, "ro")

        pars = parse_expr(self.equation)
        fun = lambdify((self.x_sym, self.y_sym), pars.subs(self.params), np)

        plt.plot(self.x, fun(self.x, self.y))
        plt.show()




