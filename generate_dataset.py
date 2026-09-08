"""
generate_dataset.py
Creates a realistic synthetic dataset of student performance
(hours studied, attendance %, assignments completed -> final exam score)
and saves it as student_scores.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_students = 200

hours_studied = np.round(np.random.uniform(0, 10, n_students), 1)
attendance = np.round(np.random.uniform(50, 100, n_students), 1)
assignments = np.random.randint(0, 11, n_students)

# Formula with some realistic noise, capped between 0-100
score = (
    5 * hours_studied
    + 0.4 * attendance
    + 2 * assignments
    + np.random.normal(0, 5, n_students)
)
score = np.clip(score, 0, 100).round(1)

df = pd.DataFrame({
    "hours_studied": hours_studied,
    "attendance_percent": attendance,
    "assignments_completed": assignments,
    "final_score": score
})

df.to_csv("student_scores.csv", index=False)
print("Dataset created: student_scores.csv")
print(df.head())
