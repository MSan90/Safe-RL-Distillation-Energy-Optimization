"""
LEVEL 0: Project Configuration — Immutable Assumptions
These constants are FROZEN for the entire project lifecycle.
"""

# ── Objective ──
# 1. Minimize Heat Duty (energy consumption)
# 2. Maintain XD → 1 (top product purity)
# 3. Zero safety constraint violations

# ── Frozen Assumptions ──
COLUMN_PRESSURE_BAR = 1.0          # Fixed, NOT a control variable
FEED_CONDITIONS_FIXED = True       # Feed composition, temperature, flow
COLUMN_STRUCTURE_FIXED = True      # Number of trays, feed tray location
CONTROL_VARIABLE = "HeatDuty"     # The ONLY manipulated variable

# ── Units ──
HEAT_DUTY_UNIT = "kW"
TEMPERATURE_UNIT = "°C"
PURITY_UNIT = "mole_fraction"     # dimensionless, 0 to 1