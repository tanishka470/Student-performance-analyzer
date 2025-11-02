import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("student_scores.csv")

print("📊 Dataset:")
print(df)


df["Total"] = df[["Math", "Science", "English"]].sum(axis=1)
df["Average"] = df["Total"] / 3

print("\n✅ With total and average:")
print(df)


math_mean = np.mean(df["Math"])
science_mean = np.mean(df["Science"])
english_mean = np.mean(df["English"])

print("\n📈 Subject-wise average using NumPy:")
print(f"Math: {math_mean:.2f}, Science: {science_mean:.2f}, English: {english_mean:.2f}")


correlation = np.corrcoef(df["Attendance"], df["Average"])[0, 1]
print(f"\n📉 Correlation between attendance and performance: {correlation:.2f}")


plt.figure(figsize=(10,6))


plt.bar(df["Name"], df["Average"], color='skyblue', label='Average Marks')
plt.plot(df["Name"], df["Attendance"], color='orange', marker='o', label='Attendance')

plt.title("Student Performance vs Attendance")
plt.xlabel("Student")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()


subject_avgs = [math_mean, science_mean, english_mean]
subjects = ["Math", "Science", "English"]

plt.figure(figsize=(6,6))
plt.pie(subject_avgs, labels=subjects, autopct="%1.1f%%", startangle=90)
plt.title("Average Marks by Subject")
plt.show()
