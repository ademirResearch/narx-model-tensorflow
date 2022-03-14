from system_class import System, np

class Lorenz(System):
    def __init__(self) -> None:
        super().__init__()
        self.num_states = 3
        self.num_inputs = 1
        self.x0_vector = [0, 0, 0]
        self.column_names = ["u", "x", "y", "z"]

    @staticmethod
    def _dynamics(x, t, u):
        """
        Lorenz Attractor dynamic system 
        (Parameters as Scipy integrate form)
        :param x (ndarray) State vector
        :param t (ndarray) Time vector
        :param u (ndarray) Input vector
        return: (list) Differential array
        """
        sigma = 10.0
        beta = 8/3
        rho = 28.0
        # Rename states (x, y, z)
        x, y, z = x
        dx = sigma * (y - x) + u
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]


class FirstOrder(System):
    def __init__(self) -> None:
        super().__init__()
        self.num_inputs = 1
        self.num_states = 1
        self.x0_vector = [0]
        self.column_names = ["u", "y"]

    @staticmethod
    def _dynamics(x, t, u):
        dx = -1.0*x + u
        return dx


class Nonlinear(System):
    def __init__(self) -> None:
        super().__init__()
        self.num_inputs = 1
        self.num_states = 2
        self.x0_vector = [0, 0]
        self.column_names = ["u", "x", "y"]

    @staticmethod
    def _dynamics(x, t, u):
        b = 1
        m = 1
        k1 = 1
        k2 = 1
        x1, x2 = x
        dx1 = x2 
        dx2 = (-k1 + k2*x1*x1) / m + ((-b/m)* x2) + (1/m) * u
        return [dx1, dx2]
