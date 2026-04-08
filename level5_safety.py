"""
LEVEL 5: Safety Constraints (Sicherheit)
Hard constraints that are NEVER negotiable.
Any violation → episode terminates immediately.
"""
import numpy as np


class SafetyChecker:
    """
    Enforces hard operating constraints on the distillation column.
    This layer guarantees the project is truly 'Safe RL'.
    """

    def __init__(self, xd_min: float,
                 ttop_range: tuple = (50.0, 150.0),
                 tmid_range: tuple = (60.0, 180.0),
                 tbottom_range: tuple = (80.0, 220.0)):
        """
        Args:
            xd_min: Minimum acceptable purity (from Level 3 feasibility)
            ttop_range: (min, max) for top temperature [°C]
            tmid_range: (min, max) for mid temperature [°C]
            tbottom_range: (min, max) for bottom temperature [°C]
        """
        self.xd_min = xd_min
        self.ttop_range = ttop_range
        self.tmid_range = tmid_range
        self.tbottom_range = tbottom_range

    def check(self, xd: float, ttop: float, tmid: float, tbottom: float) -> dict:
        """
        Check all safety constraints.
        
        Returns:
            dict with:
                safe: bool — True if ALL constraints satisfied
                violations: list of string descriptions
        """
        violations = []

        if xd < self.xd_min:
            violations.append(f"XD={xd:.4f} < xd_min={self.xd_min:.4f}")

        if not (self.ttop_range[0] <= ttop <= self.ttop_range[1]):
            violations.append(f"TTOP={ttop:.1f} outside {self.ttop_range}")

        if not (self.tmid_range[0] <= tmid <= self.tmid_range[1]):
            violations.append(f"TMID={tmid:.1f} outside {self.tmid_range}")

        if not (self.tbottom_range[0] <= tbottom <= self.tbottom_range[1]):
            violations.append(f"TBOTTOM={tbottom:.1f} outside {self.tbottom_range}")

        is_safe = len(violations) == 0
        return {"safe": is_safe, "violations": violations}