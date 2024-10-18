import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import matplotlib

# Set the backend for matplotlib
matplotlib.use('TkAgg')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GDELT Event Analysis") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

df = spark.read.csv("data.csv", header=True, inferSchema=True)
df = df.sample(fraction=0.2, seed=42)  # Sample 20% of the data for testing
df = df.dropna()

# Split dataset into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Feature selection
features = ["ActionGeoLong", "ActionGeoLat", "NumArts"]
assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
train_features = assembler.transform(train_df).cache()

# Scale the features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(train_features)
scaled_data = scaler_model.transform(train_features).cache()

kmeans = KMeans().setK(4).setSeed(1).setFeaturesCol("scaled_features").setMaxIter(50).setTol(0.1)
model = kmeans.fit(scaled_data)
train_predictions = model.transform(scaled_data)

# Evaluate
evaluator = ClusteringEvaluator(featuresCol="scaled_features", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette = evaluator.evaluate(train_predictions)
print(f"Silhouette with squared Euclidean distance: {silhouette}")

# Testing
test_features = assembler.transform(test_df).cache()
test_scaled = scaler_model.transform(test_features).cache()
test_predictions = model.transform(test_scaled)

centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Visualization - Plotting the clustering results on a map
train_predictions_sample = train_predictions.select("ActionGeoLong", "ActionGeoLat", "prediction").limit(1000).toPandas()
geometry = [Point(xy) for xy in zip(train_predictions_sample["ActionGeoLong"], train_predictions_sample["ActionGeoLat"])]
gdf = gpd.GeoDataFrame(train_predictions_sample, geometry=geometry)

# Plot the clusters
fig, ax = plt.subplots(figsize=(15, 10))
world = gpd.read_file("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson")
world.plot(ax=ax, color='lightgray')
gdf.plot(column='prediction', ax=ax, legend=True, cmap='Set1', markersize=10)
plt.title("Clustering of Global Events")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

model.write().overwrite().save("gdelt_kmeans_model")
spark.stop()
