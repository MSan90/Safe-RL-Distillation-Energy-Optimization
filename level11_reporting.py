"""
Level 11: Reporting — generates final_report.txt
"""


def generate_report(results, xd_min_engineering=0.95, feasibility=None):
    """Generate the final text report."""

    lines = []
    lines.append("=" * 60)
    lines.append("  FINAL REPORT — Safe RL for Distillation Column")
    lines.append("=" * 60)

    lines.append("")
    lines.append("ENGINEERING SPECIFICATION:")
    lines.append(f"  Minimum purity (XD):  {xd_min_engineering}")
    lines.append(f"  Objective:            Maximize XD toward 1.0")
    lines.append(f"                        while minimizing energy (HeatDuty)")

    if feasibility is not None:
        lines.append("")
        lines.append("FEASIBILITY ANALYSIS (data-based, for reference):")
        lines.append(f"  Data-based xd_min (quantile): {feasibility['xd_min']:.4f}")
        if "achievable_pct" in feasibility:
            lines.append(f"  Achievable pct:               {feasibility['achievable_pct']:.1f}%")

    if results is not None:
        lines.append("")
        lines.append("EVALUATION RESULTS:")
        for key, value in results.items():
            if isinstance(value, float):
                lines.append(f"  {key:<30} {value:.4f}")
            else:
                lines.append(f"  {key:<30} {value}")

    lines.append("")
    lines.append("DESIGN DECISIONS:")
    lines.append("  1. XD constraint is engineering-based (0.95), not quantile-based")
    lines.append("  2. Soft termination: 5 grace steps before episode ends")
    lines.append("  3. Hinge reward: heavy penalty below 0.95, bonus above")
    lines.append("  4. Safe reset: episodes start in feasible HeatDuty region")
    lines.append("  5. Energy minimization as secondary objective")

    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)

    with open("final_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n[REPORT] Saved: final_report.txt")