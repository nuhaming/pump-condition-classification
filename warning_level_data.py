"""
WARNING-level records only (15 records).
Columns: Machine_ID, Temperature_C, Vibration_mm_s, Speed_RPM, Operating_Hours, Failure_Risk
Label: 1 = WARNING
IDs 31-45 so they can be appended after the original 30 records.
"""

warning_data = [
    [31, 67, 3.0, 1430, 3000, 1],
    [32, 70, 3.4, 1420, 3400, 1],
    [33, 72, 3.8, 1415, 3800, 1],
    [34, 68, 3.1, 1428, 3100, 1],
    [35, 74, 4.2, 1405, 4200, 1],
    [36, 69, 3.3, 1422, 3300, 1],
    [37, 71, 3.6, 1418, 3600, 1],
    [38, 73, 4.0, 1410, 4000, 1],
    [39, 75, 4.5, 1402, 4400, 1],
    [40, 67, 3.2, 1425, 3200, 1],
    [41, 70, 3.5, 1416, 3500, 1],
    [42, 72, 3.9, 1412, 3900, 1],
    [43, 74, 4.3, 1408, 4300, 1],
    [44, 68, 3.0, 1427, 3050, 1],
    [45, 73, 4.1, 1406, 4100, 1],
]

if __name__ == "__main__":
    import pandas as pd
    columns = ["Machine_ID", "Temperature_C", "Vibration_mm_s",
               "Speed_RPM", "Operating_Hours", "Failure_Risk"]
    print(pd.DataFrame(warning_data, columns=columns).to_string(index=False))
