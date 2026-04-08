import numpy as np


class PIDController:
    """
    PID baseline controller for XD tracking.
    Output: normalized action in [-1, 1] (matches GymWrapper action_space).
    """

    def __init__(
        self,
        kp,
        ki,
        kd,
        setpoint=0.95,
        dt=1.0,
        out_min=-1.0,
        out_max=1.0,
        integral_clip=1.0,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = float(setpoint)
        self.dt = float(dt)

        self.out_min = float(out_min)
        self.out_max = float(out_max)

        # Simple anti-windup: clamp integral term
        self.integral_clip = float(integral_clip)

        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, measurement):
        """
        measurement: current XD
        returns: np.array([action], dtype=np.float32) in [-1,1]
        """
        xd = float(measurement)
        error = self.setpoint - xd

        # Derivative on error
        derivative = (error - self.prev_error) / self.dt

        # Provisional (without updating integral yet)
        u_p = self.kp * error
        u_d = self.kd * derivative
        u_i = self.ki * self.integral
        u_unsat = u_p + u_i + u_d

        # Saturate
        u_sat = float(np.clip(u_unsat, self.out_min, self.out_max))

        # --- Conditional integration (anti-windup) ---
        pushing_high = (u_sat >= self.out_max - 1e-9) and (error > 0)
        pushing_low = (u_sat <= self.out_min + 1e-9) and (error < 0)
        if not (pushing_high or pushing_low):
            self.integral += error * self.dt
            self.integral = float(np.clip(self.integral, -self.integral_clip, self.integral_clip))

        self.prev_error = error

        # Recompute with updated integral
        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        u = float(np.clip(u, self.out_min, self.out_max))
        return np.array([u], dtype=np.float32)


def build_default_pid(env):
    """
    Fair baseline PID for margin tracking: setpoint=0.985
    """
    sp = 0.985
    return PIDController(
        kp=6.0,
        ki=0.4,
        kd=0.0,
        setpoint=sp,
        dt=1.0,
        integral_clip=2.0,
    )


class EconomicTrimPID:
    """
    Economic baseline:
    - If XD is comfortably above margin band -> gently reduce heat (save energy)
    - If XD falls below band -> PID pushes back up
    """

    def __init__(self, pid: PIDController, xd_soft=0.985, band=0.002, trim=-0.05,
                 out_min=-1.0, out_max=1.0):
        self.pid = pid
        self.xd_soft = float(xd_soft)
        self.band = float(band)
        self.trim = float(trim)
        self.out_min = float(out_min)
        self.out_max = float(out_max)

    def reset(self):
        self.pid.reset()

    def compute(self, xd):
        xd = float(xd)

        # If safely above margin -> trim down energy slowly
        if xd >= self.xd_soft + self.band:
            u = float(np.clip(self.trim, self.out_min, self.out_max))
            return np.array([u], dtype=np.float32)

        return self.pid.compute(xd)


def build_economic_pid(env):
    pid = PIDController(
        kp=6.0,
        ki=0.4,
        kd=0.0,
        setpoint=0.985,
        dt=1.0,
        integral_clip=2.0,
    )
    return EconomicTrimPID(pid, xd_soft=0.985, band=0.002, trim=-0.05)