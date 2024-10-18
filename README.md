# GDELT Event Clustering Project

## Project Overview
This project aims to identify global hotspots of social and political events using the GDELT dataset (1979-2013). By applying K-means clustering, the project visualizes key regions of activity, helping us understand patterns in global events.

## Prerequisites
- Python 3.8 or higher
- PySpark (`pip install pyspark`)
- GeoPandas (`pip install geopandas`)
- Matplotlib (`pip install matplotlib`)
- Shapely (`pip install shapely`)

## How to Run

1. **Prepare the dataset**: Place `data.csv` in the project directory. Ensure it contains the correct GDELT dataset. Note: Data downloaded directly from GDELT is not named as data.csv, I changed it.

2. **Run the script**:  
   ```bash
   python Code.py
   ```
   This script will execute data preprocessing, clustering, and generate a geographical visualization of clusters.

## Output
- **Silhouette Score**: Evaluates cluster quality.
- **Map Visualization**: Displays event hotspots globally.

## Notes
- Adjust the `fraction` parameter in the script to sample different sizes of data for quicker testing.
- Ensure sufficient memory allocation when running with larger datasets to avoid errors.


