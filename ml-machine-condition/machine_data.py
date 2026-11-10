"""Shared two-class dataset (NORMAL / FAULT). Both group members MUST use this same file."""
import pandas as pd

FEATURES = ["Temperature_C", "Vibration_mm_s", "Speed_RPM", "Operating_Hours"]
LABELS = ["NORMAL", "FAULT"]  # 0, 1

_rows = [
    # Temperature_C, Vibration_mm_s, Speed_RPM, Operating_Hours, Condition
    # ---- NORMAL (12) ----
    [48, 1.2, 1480,  900, "NORMAL"], [51, 1.5, 1490, 1200, "NORMAL"],
    [55, 1.8, 1475, 1500, "NORMAL"], [58, 2.0, 1500, 1800, "NORMAL"],
    [60, 2.2, 1460, 2100, "NORMAL"], [53, 1.4, 1510, 1100, "NORMAL"],
    [57, 1.7, 1495, 1600, "NORMAL"], [50, 1.3, 1505, 1000, "NORMAL"],
    [59, 2.1, 1485, 1900, "NORMAL"], [56, 1.9, 1498, 1400, "NORMAL"],
    [61, 2.3, 1470, 2000, "NORMAL"], [52, 1.6, 1492, 1300, "NORMAL"],
    # ---- FAULT (12) ----
    [82, 5.8, 1370, 5200, "FAULT"], [88, 6.5, 1340, 6100, "FAULT"],
    [91, 7.1, 1320, 6800, "FAULT"], [78, 5.2, 1390, 4800, "FAULT"],
    [95, 7.8, 1290, 7400, "FAULT"], [85, 6.1, 1360, 5700, "FAULT"],
    [99, 8.4, 1260, 8100, "FAULT"], [76, 4.9, 1400, 4500, "FAULT"],
    [90, 7.0, 1310, 7000, "FAULT"], [87, 6.7, 1330, 6400, "FAULT"],
    [102, 8.8, 1240, 8600, "FAULT"], [80, 5.5, 1380, 5000, "FAULT"],
]


def load_data() -> pd.DataFrame:
    df = pd.DataFrame(_rows, columns=FEATURES + ["Condition"])
    df.insert(0, "Machine_ID", range(1, len(df) + 1))
    return df
