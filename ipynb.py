import ipynb as py
import ipynb. as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("Dataset loaded successfully!\n")

print("First 5 rows of the dataset:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

grouped = df.groupby('species').mean()
print("\nMean of numerical columns grouped by species:")
print(grouped)

print("\nObservation:")
print("Setosa generally has smaller sepal and petal sizes compared to Virginica and Versicolor.")

sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue')
plt.title('Sepal Length Trend (Simulated Time)')
plt.xlabel('Index (Simulated Time)')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
sns.barplot(x='species', y='petal length (cm)', data=df, palette='pastel')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='Set1')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()

print("\nFinal Observations:")
print("- The dataset is clean with no missing values.")
print("- Setosa has distinctly smaller petals and sepals.")
print("- Virginica species tends to have the largest petal length.")
print("- There's a clear positive correlation between sepal length and petal length.")
print("- The species groups show visible patterns and separations in scatter plots, useful for classification.")
