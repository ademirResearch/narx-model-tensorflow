import numpy as np
from scipy.integrate import odeint

class System:
    def __init__(self) -> None:
        self.num_states = None
        self.num_inputs = None
        self.x0_vector = None
        pass

    @staticmethod
    def _dynamics(x, t, u):
        return

    def experiment(self, u, x0, ts=0.01):
        """Simulates the dynamic system's output from a given input signal u
        Args:
            u (_type_): Input signal
            x0 (_type_): Initial condition, state vector at t_0
            ts (float, optional): Sampling time. Defaults to 0.01.
        Returns:
            ndarray: System's response 
        """
        samples = len(u)
        experiment_result = np.zeros((samples + 1, self.num_states))  # + 1 To contain the initial condition x0
        experiment_result[0] = x0
        t0 = 0
        for i in range(samples):
            result_states = odeint(self._dynamics, y0=x0, t=[t0, t0 + ts], args=(u[i],), mxstep=2)
            x0 = result_states[-1]
            experiment_result[i + 1] = result_states[-1]
            t0 = t0 + ts
        return experiment_result[:samples]

    def step(self, num_steps=1, ts=0.01, x0=None, u=1.0):
        """
        :param num_steps (int) Number of future steps to simulate
        :param ts (float) Sample time 
        :param x0 (ndarray) Initial conditions
        :param u (ndarray) Input vector
        :return (ndarray) Simulation-steps results
        """
        if x0 is None:
            x0 = np.zeros(self.num_states)
        
        complete_result = np.zeros((num_steps, self.num_states))
        t = 0
        # Simulate steps
        for _step in range(num_steps):
            result_states = odeint(self._dynamics, y0=x0, t=[t, t + ts], args=(u,))
            t = ts
            x0 = result_states[-1].copy()
            complete_result[_step] = result_states[-1]

        return complete_result