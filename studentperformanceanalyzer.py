import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os   

#Load and Process Data
# keep asking the user till they provide a valid file path
while True:
    # Ask the user for the file path
    file_path = input("Enter the path to your student scores CSV file: ")
    
    # Clean up the path (removes extra spaces)
    file_path = file_path.strip().strip('"') 
    
    # file should exist and be a .csv file
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'. Please try again.")
    elif not file_path.endswith('.csv'):
        print("Error: The file must be a .csv file. Please try again.")
    else:
        #valid file path
        break 

#Try to read the valid file path
try:
    df = pd.read_csv(file_path)
    if df.empty:
        # Handle empty file
        print(f"Error: '{file_path}' is empty.")
        exit()
        
except Exception as e:
    # Handle other read errors
    print(f"An error occurred while reading the file: {e}")
    exit()


print(f"📊 Original Dataset from '{file_path}':")
print(df)

#Feature Engineering 
#feature engineering : naye feature banaye from exisiting ones
# Here, we create Total, Average, Grade, and Rank features

df["Total"] = df[["Math", "Science", "English"]].sum(axis=1)
df["Average"] = df["Total"] / 3
bins = [0, 59, 69, 79, 89, 100]
labels = ["F", "D", "C", "B", "A"]
df["Grade"] = pd.cut(df["Average"], bins=bins, labels=labels, right=True)
df["Rank"] = df["Average"].rank(ascending=False, method="min").astype(int)
df = df.sort_values(by="Rank")

#a kind of new dataset is created after feature engineering which includes Total, Average, Grade, and Rank
# Display Processed Data
print("\n✅ Processed Data with Total, Average, Grade, and Rank:")
print(df)


# Statistical Analysis and Insights (used numpy)
math_mean = np.mean(df["Math"])
science_mean = np.mean(df["Science"])
english_mean = np.mean(df["English"])

print("\n📈 Subject-wise Average (using NumPy):")
print(f"Math: {math_mean:.2f}, Science: {science_mean:.2f}, English: {english_mean:.2f}")

correlation = np.corrcoef(df["Attendance"], df["Average"])[0, 1]
print(f"\n📉 Correlation between Attendance and Average: {correlation:.2f}")

top_student = df.loc[df["Rank"] == 1]
print("\n🏆 Top Student(s):")
if len(top_student) > 1:
    print("There is a tie for 1st place!")
    print(top_student[['Name', 'Average', 'Grade']])
else:
    print(top_student[['Name', 'Average', 'Grade']].to_string(index=False))


# Visualizations using Matplotlib

plt.style.use('ggplot') # we're using ggplot because i like it and i only know about it

# PLOT 1: Subject Average Comparison (made a bar plot)
plt.figure(figsize=(8, 6))
subjects = ["Math", "Science", "English"]
averages = [math_mean, science_mean, english_mean]
bars = plt.bar(subjects, averages, color=['#66b3ff', '#99ff99', '#ff9999'])

plt.title("Class Average Score by Subject")
plt.xlabel("Subject")
plt.ylabel("Average Score")
plt.ylim(0, 100) 

# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}', 
             ha='center', va='bottom')
plt.show()


# PLOT 2: Correlation Plot (made a scatter plot with a trend line)
plt.figure(figsize=(10, 6))
plt.scatter(df["Attendance"], df["Average"], alpha=0.7) # Main scatter plot

# Calculate and plot regression line using numpy
m, b = np.polyfit(df["Attendance"], df["Average"], 1) # m = slope, b = intercept
plt.plot(df["Attendance"], m * df["Attendance"] + b, color="red", linestyle="--", label="Trend Line")

# Add labels for each point
for i, row in df.iterrows():
    plt.text(row["Attendance"], row["Average"] + 0.5, row["Name"], 
             horizontalalignment='center', size='small', color='dimgray')

plt.title("Attendance vs. Average Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Average Score")
plt.legend()
plt.grid(True)
plt.show()


# PLOT 3: Subject Comparison (Box Plot)
plt.figure(figsize=(10, 6))
subject_scores = [df['Math'], df['Science'], df['English']]
labels = ['Math', 'Science', 'English']

# Create the boxplot
plt.boxplot(subject_scores, labels=labels, patch_artist=True, 
            medianprops={'color': 'black'})  

# Replicate stripplot with jitter
# We need to add a small random "jitter" to the x-coordinates
for i, scores in enumerate(subject_scores):
    # 'i + 1' is the x-position of the box (1 for Math, 2 for Science, etc.)
    # np.random.normal creates random jitter
    jitter = np.random.normal(0, 0.04, size=len(scores)) # Small random noise
    plt.scatter(np.full(len(scores), i + 1) + jitter, scores, 
                color='black', alpha=0.4)

plt.title("Distribution of Scores by Subject")
plt.ylabel("Score")
plt.xlabel("Subject")
plt.grid(True, axis='y') # Add gridlines for the y-axis
plt.show()


# PLOT 4: Overall Performance (Histogram) 
plt.figure(figsize=(10, 6))
# 'bins=10' is a good start, 'edgecolor' makes bins clearer
plt.hist(df["Average"], bins=10, color='skyblue', edgecolor='black')

# Add the mean line
class_mean = df["Average"].mean()
plt.axvline(class_mean, color='red', linestyle='--', 
            label=f'Class Mean: {class_mean:.2f}')

plt.title("Class Performance: Distribution of Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")
plt.legend()
plt.grid(True, axis='y')
plt.show()

# PLOT 5: Grade Distribution (Pie Chart)  
df["Grade"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, figsize=(6,6))
plt.title("Grade Distribution")
plt.show()

print("\n🏅 Top 3 Performers:")
print(df.nlargest(3, "Average")[["Name", "Average", "Grade"]])

print("\n⚠️ Bottom 3 Performers:")
print(df.nsmallest(3, "Average")[["Name", "Average", "Grade"]])

print("\n🎉 Analysis Complete!")
