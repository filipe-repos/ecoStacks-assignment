import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import numpy as np
from shapely.geometry import mapping

# -----------------
# Load CSV points
# -----------------
df = pd.read_csv("point_locations.csv", names=["lon", "lat"], skiprows=1)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

# -----------------
# Function to reproject a raster to UTM
# -----------------
def reproject_to_utm(src, utm_crs, out_path):
    transform, width, height = calculate_default_transform(
        src.crs, utm_crs, src.width, src.height, *src.bounds
    )
    profile = src.profile.copy()
    profile.update({
        "crs": utm_crs,
        "transform": transform,
        "width": width,
        "height": height
    })
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=utm_crs,
                resampling=Resampling.nearest
            )
    return out_path

# -----------------
# Function to process a single TIFF
# -----------------
def process_tiff(tiff_path, gdf, out_dir="ndvi_outputs"):
    results = []

    with rasterio.open(tiff_path) as src:
        # Skip single-band rasters (like NDVI results)
        if src.count < 8:
            print(f"Skipping {tiff_path} (only {src.count} band(s))")
            return pd.DataFrame()

        # Determine UTM zone from points
        utm_crs = gdf.estimate_utm_crs()

        # Reproject raster into UTM
        utm_path = os.path.join(out_dir, "tmp_utm.tif")
        os.makedirs(out_dir, exist_ok=True)
        reproject_to_utm(src, utm_crs, utm_path)

    # Open reprojected raster
    with rasterio.open(utm_path) as src:
        # Reproject points to UTM
        gdf_utm = gdf.to_crs(src.crs)

        # Create 1 km buffer in meters
        gdf_utm["buffer"] = gdf_utm.geometry.buffer(1000)

        # Read red (B4) and nir (B8)
        red = src.read(4).astype("float32")
        nir = src.read(8).astype("float32")

        # Compute NDVI
        ndvi = (nir - red) / (nir + red + 1e-10)

        # Save NDVI raster
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", compress="lzw")
        ndvi_path = os.path.join(out_dir, os.path.basename(tiff_path).replace(".tif", "_ndvi.tif"))
        with rasterio.open(ndvi_path, "w", **profile) as dst:
            dst.write(ndvi, 1)

        # Compute stats per buffer
        for idx, row in gdf_utm.iterrows():
            geom = [mapping(row["buffer"])]
            try:
                red_masked, _ = mask(src, geom, crop=True, filled=True, indexes=4)
                nir_masked, _ = mask(src, geom, crop=True, filled=True, indexes=8)
            except ValueError:
                print(f" Skipping point {idx} — no overlap with {os.path.basename(tiff_path)}")
                continue

            ndvi_masked = (nir_masked.astype("float32") - red_masked.astype("float32")) / (
                nir_masked.astype("float32") + red_masked.astype("float32") + 1e-10
            )

            vals = ndvi_masked[np.isfinite(ndvi_masked)]
            if vals.size > 0:
                results.append({
                    "tiff_file": os.path.basename(tiff_path),
                    "point_id": idx,
                    "mean_ndvi": float(vals.mean()),
                    "min_ndvi": float(vals.min()),
                    "max_ndvi": float(vals.max()),
                    "std_ndvi": float(vals.std())
                })

    return pd.DataFrame(results)

# -----------------
# Process all TIFFs in folder
# -----------------
tiff_folder = "./"
all_results = []

for fname in os.listdir(tiff_folder):
    if fname.lower().endswith(".tif") or fname.lower().endswith(".tiff"):
        tiff_path = os.path.join(tiff_folder, fname)
        print(f"Processing {tiff_path}...")
        df_out = process_tiff(tiff_path, gdf, out_dir="ndvi_outputs")
        if not df_out.empty:
            all_results.append(df_out)

# Combine all results
if all_results:
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv("ndvi_stats.csv", index=False)
    print("Finished. Stats saved to ndvi_stats.csv and NDVI rasters in ndvi_outputs/")
else:
    print("No results generated.")