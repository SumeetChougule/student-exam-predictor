import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Data

df = pd.read_csv("stud.csv")

df.shape
df["parental_level_of_education"].unique()

df.info()

df.isna().sum()

# There are no missing values in the dataset

df.duplicated().sum()

# There are no duplicated values in the dataset

df.nunique()

df.describe()
df.columns

print(f'Categories in "gender" variable: {df["gender"].unique()} ')
print(f'Categories in "race ethnicity" variable: {df["race_ethnicity"].unique()} ')
print(
    f'Categories in "parental level of education" variable: {df["parental_level_of_education"].unique()} '
)
print(f'Categories in "lunch" variable: {df["lunch"].unique()} ')
print(
    f'Categories in "test preparation course" variable: {df["test_preparation_course"].unique()} '
)


# define numerical & categorical columns

numerical_features = [feature for feature in df.columns if df[feature].dtype != "O"]

categorical_features = [feature for feature in df.columns if df[feature].dtype == "O"]


# Extra Features

df["total_score"] = df["math_score"] + df["reading_score"] + df["writing_score"]
df["average"] = df["total_score"] / 3

reading_full = df[df["reading_score"] == 100]["average"].count()
writing_full = df[df["writing_score"] == 100]["average"].count()
math_full = df[df["math_score"] == 100]["average"].count()

print(f"Number of students with full marks in math {math_full}")
print(f"Number of students with full marks in reading {reading_full}")
print(f"Number of students with full marks in writing {writing_full}")


# Visualization

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df, x="average", bins=30, kde=True, color="g")
plt.subplot(122)
sns.histplot(data=df, x="average", kde=True, hue="gender")
plt.show()


plt.subplots(1, 3, figsize=(25, 6))
plt.subplot(141)
ax = sns.histplot(data=df, x="average", kde=True, hue="race_ethnicity")
plt.subplot(142)
ax = sns.histplot(
    data=df[df.gender == "female"], x="average", kde=True, hue="race_ethnicity"
)
plt.subplot(143)
ax = sns.histplot(
    data=df[df.gender == "male"], x="average", kde=True, hue="race_ethnicity"
)
plt.show()

df.columns


plt.figure(figsize=(18, 8))
plt.subplot(1, 3, 1)
plt.title("MATH SCORES")
sns.violinplot(y="math_score", data=df, color="red", linewidth=3)
plt.subplot(1, 3, 2)
plt.title("READING SCORES")
sns.violinplot(y="reading_score", data=df, color="green", linewidth=3)
plt.subplot(1, 3, 3)
plt.title("WRITING SCORES")
sns.violinplot(y="writing_score", data=df, color="blue", linewidth=3)
plt.show()


####### Multivariate analysis

plt.rcParams["figure.figsize"] = (30, 12)

plt.subplot(1, 5, 1)
size = df["gender"].value_counts()
labels = "Female", "Male"
color = ["red", "green"]


plt.pie(size, colors=color, labels=labels, autopct=".%2f%%")
plt.title("Gender", fontsize=20)
plt.axis("off")


plt.subplot(1, 5, 2)
size = df["race_ethnicity"].value_counts()
labels = "Group C", "Group D", "Group B", "Group E", "Group A"
color = ["red", "green", "blue", "cyan", "orange"]

plt.pie(size, colors=color, labels=labels, autopct=".%2f%%")
plt.title("race_ethnicity", fontsize=20)
plt.axis("off")


plt.subplot(1, 5, 3)
size = df["lunch"].value_counts()
labels = "Standard", "Free"
color = ["red", "green"]

plt.pie(size, colors=color, labels=labels, autopct=".%2f%%")
plt.title("Lunch", fontsize=20)
plt.axis("off")


plt.subplot(1, 5, 4)
size = df["test_preparation_course"].value_counts()
labels = "None", "Completed"
color = ["red", "green"]

plt.pie(size, colors=color, labels=labels, autopct=".%2f%%")
plt.title("Test Course", fontsize=20)
plt.axis("off")


plt.subplot(1, 5, 5)
size = df["parental_level_of_education"].value_counts()
labels = (
    "Some College",
    "Associate's Degree",
    "High School",
    "Some High School",
    "Bachelor's Degree",
    "Master's Degree",
)
color = ["red", "green", "blue", "cyan", "orange", "grey"]

plt.pie(size, colors=color, labels=labels, autopct=".%2f%%")
plt.title("Parental Education", fontsize=20)
plt.axis("off")


plt.tight_layout()
plt.grid()

plt.show()

# Univariate analysis

f, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.countplot(x=df["gender"], data=df, palette="bright", ax=ax[0], saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container, color="black", size=20)

plt.pie(
    x=df["gender"].value_counts(),
    labels=["Female", "Male"],
    explode=[0, 0.1],
    autopct="%1.1f%%",
    shadow=True,
    colors=["#ff4d4d", "#ff8000"],
)
plt.show()

# Bivariate analysis


# Select only numeric columns
numeric_columns = df.select_dtypes(include=["number"])

# Group by gender and calculate the mean for numeric columns
gender_group = numeric_columns.groupby(df["gender"])
gender_mean = gender_group.mean()

# Visualize the data using a bar plot
plt.figure(figsize=(10, 8))

X = ["Total Average", "Math Average"]


female_scores = [gender_mean["average"][0], gender_mean["math_score"][0]]
male_scores = [gender_mean["average"][1], gender_mean["math_score"][1]]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, male_scores, 0.4, label="Male")
plt.bar(X_axis + 0.2, female_scores, 0.4, label="Female")

plt.xticks(X_axis, X)
plt.ylabel("Marks")
plt.title("Total average v/s Math average marks of both the genders", fontweight="bold")
plt.legend()
plt.show()


f, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.countplot(
    x=df["race_ethnicity"], data=df, palette="bright", ax=ax[0], saturation=0.95
)
for container in ax[0].containers:
    ax[0].bar_label(container, color="black", size=20)

plt.pie(
    x=df["race_ethnicity"].value_counts(),
    labels=df["race_ethnicity"].value_counts().index,
    explode=[0.1, 0, 0, 0, 0],
    autopct="%1.1f%%",
    shadow=True,
)
plt.show()

# ---------------------------

Group_data2 = df.groupby("race_ethnicity")
f, ax = plt.subplots(1, 3, figsize=(20, 8))
sns.barplot(
    x=Group_data2["math_score"].mean().index,
    y=Group_data2["math_score"].mean().values,
    palette="mako",
    ax=ax[0],
)
ax[0].set_title("Math score", color="#005ce6", size=20)

for container in ax[0].containers:
    ax[0].bar_label(container, color="black", size=15)

sns.barplot(
    x=Group_data2["reading_score"].mean().index,
    y=Group_data2["reading_score"].mean().values,
    palette="flare",
    ax=ax[1],
)
ax[1].set_title("Reading score", color="#005ce6", size=20)

for container in ax[1].containers:
    ax[1].bar_label(container, color="black", size=15)

sns.barplot(
    x=Group_data2["writing_score"].mean().index,
    y=Group_data2["writing_score"].mean().values,
    palette="coolwarm",
    ax=ax[2],
)
ax[2].set_title("Writing score", color="#005ce6", size=20)

for container in ax[2].containers:
    ax[2].bar_label(container, color="black", size=15)


df[numerical_features].groupby(df["parental_level_of_education"]).agg("mean").plot(
    kind="barh", figsize=(10, 10)
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()


plt.subplots(1, 4, figsize=(16, 5))
plt.subplot(141)
sns.boxplot(df["math_score"], color="skyblue")
plt.subplot(142)
sns.boxplot(df["reading_score"], color="hotpink")
plt.subplot(143)
sns.boxplot(df["writing_score"], color="yellow")
plt.subplot(144)
sns.boxplot(df["average"], color="lightgreen")
plt.show()

sns.pairplot(df, hue="gender")
plt.show()
