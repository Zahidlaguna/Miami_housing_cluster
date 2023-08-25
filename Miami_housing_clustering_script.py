# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import matplotlib.image as mpimg

# %%
miami_housing = pd.read_csv('/Users/zahidlaguna/Downloads/miami-housing.csv', usecols= ['LATITUDE', 'LONGITUDE', 'SALE_PRC'])
miami_housing.head(10)

# %%
miami_housing.shape

# %%
miami_housing.describe()

# %%
miami_housing.info()

# %%
miami_housing.isnull().sum()

# %%
scaler = StandardScaler()
print(scaler.fit(miami_housing))

# %%
print(scaler.transform(miami_housing))

# %%
fig, ax = plt.subplots(figsize=(6,6))
sns.scatterplot(data= miami_housing, x = 'LONGITUDE', y = 'LATITUDE', hue = 'SALE_PRC')

# %%
X_train, X_test, y_train, y_test = train_test_split(miami_housing[['LONGITUDE', 'LATITUDE']], miami_housing[['SALE_PRC']], test_size=0.3, random_state=0)
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

# %%
kmeans = KMeans(n_clusters = 5, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)
sns.scatterplot(data = X_train, x = 'LONGITUDE', y = 'LATITUDE', hue = kmeans.labels_)

# %%
sns.boxplot(x = kmeans.labels_, y = y_train['SALE_PRC'])

# %%
silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')

# %%
K = range(2, 8)
fits = []
score = []

for k in K:
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(X_train_norm)
    fits.append(model)
    score.append(silhouette_score(X_train_norm, model.labels_, metric='haversine'))

# %%
sns.scatterplot(data = X_train, x = 'LONGITUDE', y = 'LATITUDE', hue = fits[0].labels_)

# %%
sns.scatterplot(data = X_train, x = 'LONGITUDE', y = 'LATITUDE', hue = fits[2].labels_)

# %%
sns.scatterplot(data = X_train, x = 'LONGITUDE', y = 'LATITUDE', hue = fits[5].labels_)

# %%
sns.lineplot(x = K, y = score)

# %%
sns.boxplot(x = fits[3].labels_, y = y_train['SALE_PRC'])

# %%
miami = mpimg.imread('/Users/zahidlaguna/Downloads/Miami.png')
img_height, img_width = miami.shape[:2]
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X_train['LONGITUDE'], X_train['LATITUDE'], c = fits[3].labels_, cmap = 'plasma', alpha = 0.8)
ax.imshow(miami, extent = [-80.329, -80.131, 25.694, 25.872], alpha = 0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Miami Housing Price Clusters')
labels = ['Low', 'Medium', 'High'] 
plt.legend(labels)
plt.show()


