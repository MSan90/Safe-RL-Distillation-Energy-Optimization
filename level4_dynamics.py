"""
LEVEL 4: Dynamic Approximation -- Introducing Time

The surrogate is steady-state only. RL needs temporal response.
We use first-order lag + pure dead-time as engineering approximation.

Model: FOPDT (First Order Plus Dead Time)

    For each output variable:
        1. Dead-time: The new target is delayed by N steps.
           During this delay, the system sees the OLD target.
        2. First-order lag: After the delay, the variable
           exponentially approaches the (delayed) target.

    x(t+1) = x(t) + (dt / tau) * (target_delayed(t) - x(t))

    where target_delayed(t) = target(t - dead_time)

DESIGN NOTE:
    Each process output (XD, TTOP, TMID, TBOTTOM) has its own
    independent time constant (tau) and shares the same dead-time.

    Physical justification:
        - Composition dynamics (XD) are slower than temperature
          because mass transfer is inherently slower than heat transfer.
        - Dead-time represents transport delay through the column
          (liquid holdup, vapor travel time, sensor delay).

    Default values:
        - tau_xd:      20.0  (composition -- slowest)
        - tau_ttop:     5.0  (temperature -- faster)
        - tau_tmid:     5.0
        - tau_tbottom:  5.0
        - dead_time:    3 steps

    For report / defense:
        "The dynamic model uses a First Order Plus Dead Time (FOPDT)
         approximation. Each output has an independent time constant,
         with composition (tau=20) responding slower than temperature
         (tau=5). A pure dead-time of 3 steps models the transport
         delay through the column. This is consistent with standard
         industrial process control modeling practice."
"""
import numpy as np
from collections import deque


class FirstOrderLag:
    """
    Single first-order lag element with pure dead-time for one process variable.

    The dead-time is implemented as a FIFO buffer (deque) of fixed length.
    New targets enter the buffer; the output of the buffer (delayed target)
    feeds into the first-order lag equation.

    x(t+1) = x(t) + alpha * (target_delayed(t) - x(t))

    where:
        alpha = clip(dt / tau, 0, 1)
        target_delayed(t) = target(t - dead_time_steps)
    """

    def __init__(self, tau, dt=1.0, dead_time_steps=0):
        assert tau > 0, "tau must be positive"
        assert dt > 0, "dt must be positive"
        assert dead_time_steps >= 0, "dead_time must be non-negative"

        self.tau = tau
        self.dt = dt
        self.dead_time_steps = dead_time_steps
        self.alpha = np.clip(dt / tau, 0.0, 1.0)

        # Fixed-length buffer for dead-time implementation
        # When dead_time_steps=0, buffer has length 1 (no delay)
        # When dead_time_steps=3, buffer has length 4
        #   (current + 3 delayed = output is 3 steps old)
        self._buffer_len = dead_time_steps + 1
        self._buffer = None

    def reset(self, initial_value=None):
        """
        Reset the dead-time buffer.
        If initial_value is given, fill the buffer with it
        so the system starts at steady state.
        """
        if initial_value is not None:
            self._buffer = deque([initial_value] * self._buffer_len,
                                 maxlen=self._buffer_len)
        else:
            self._buffer = None

    def step(self, current, ss_target):
        """
        Advance one time step.

        Args:
            current:    Current value of the process variable
            ss_target:  New steady-state target from surrogate

        Returns:
            Updated value after lag + dead-time
        """
        # Initialize buffer on first call
        if self._buffer is None:
            self._buffer = deque([current] * self._buffer_len,
                                 maxlen=self._buffer_len)

        # Push new target into buffer (right side)
        self._buffer.append(ss_target)

        # Pop delayed target from buffer (left side, automatically
        # handled by maxlen -- oldest value was already dropped)
        # The leftmost element is the one that entered dead_time_steps ago
        delayed_target = self._buffer[0]

        # First-order lag toward the delayed target
        new_value = current + self.alpha * (delayed_target - current)

        return float(new_value)


class DynamicsBank:
    """
    Bank of independent FOPDT models -- one per process output.

    This ensures that composition (XD) can have a different time constant
    than temperatures (TTOP, TMID, TBOTTOM), reflecting physical reality.

    ASSUMPTION STATEMENT (for report / defense):
        "Each process output is modeled with an independent First Order
         Plus Dead Time (FOPDT) model. Default time constants reflect
         the physical expectation that composition responds more slowly
         than temperature. The dead-time models transport delay through
         the column internals."
    """

    def __init__(self,
                 tau_xd=20.0,
                 tau_ttop=5.0,
                 tau_tmid=5.0,
                 tau_tbottom=5.0,
                 dt=1.0,
                 dead_time_steps=3):

        self.lags = {
            "XD":      FirstOrderLag(tau=tau_xd,      dt=dt, dead_time_steps=dead_time_steps),
            "TTOP":    FirstOrderLag(tau=tau_ttop,     dt=dt, dead_time_steps=dead_time_steps),
            "TMID":    FirstOrderLag(tau=tau_tmid,     dt=dt, dead_time_steps=dead_time_steps),
            "TBOTTOM": FirstOrderLag(tau=tau_tbottom,  dt=dt, dead_time_steps=dead_time_steps),
        }

        print("[DYNAMICS] FOPDT model initialized:")
        for name, lag in self.lags.items():
            print("   %8s -> tau=%.1f, dt=%.1f, "
                  "dead_time=%d steps, alpha=%.4f" %
                  (name, lag.tau, lag.dt,
                   lag.dead_time_steps, lag.alpha))

    def reset(self, initial_values=None):
        """
        Reset all lag buffers.

        Args:
            initial_values: Optional dict with initial values per variable.
                            Example: {"XD": 0.98, "TTOP": 80.0, ...}
                            If None, buffers initialize on first step call.
        """
        for name, lag in self.lags.items():
            init_val = None
            if initial_values is not None and name in initial_values:
                init_val = initial_values[name]
            lag.reset(initial_value=init_val)

    def step_all(self, current, ss_targets):
        """
        Advance all process variables by one time step.

        Args:
            current:    {"XD": float, "TTOP": float, "TMID": float, "TBOTTOM": float}
            ss_targets: {"XD_ss": float, "TTOP_ss": float, "TMID_ss": float, "TBOTTOM_ss": float}

        Returns:
            dict with updated values
        """
        return {
            "XD":      self.lags["XD"].step(current["XD"],      ss_targets["XD_ss"]),
            "TTOP":    self.lags["TTOP"].step(current["TTOP"],    ss_targets["TTOP_ss"]),
            "TMID":    self.lags["TMID"].step(current["TMID"],    ss_targets["TMID_ss"]),
            "TBOTTOM": self.lags["TBOTTOM"].step(current["TBOTTOM"], ss_targets["TBOTTOM_ss"]),
        }