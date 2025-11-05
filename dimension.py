import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Load local CSV
penguins = pd.read_csv("penguins.csv").dropna()

# Select numeric columns
data = penguins.select_dtypes(float)

# PCA
pcs = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=0)).fit_transform(data)
df = pd.DataFrame(pcs, columns=['PC1','PC2'])
df['Species'], df['Sex'] = penguins.species.values, penguins.sex.values

# Plot
plt.figure(figsize=(12,10))
sns.scatterplot(x='PC1', y='PC2', data=df, hue='Species', style='Sex', s=100)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA", size=24)
plt.savefig("PCA.png", dpi=75)
plt.show()

