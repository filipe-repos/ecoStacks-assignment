import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import numpy as np
from shapely.geometry import mapping


#indexes of bands for NDVI
RED_BAND = 4 #Red
NIR_BAND = 8 #InfraRed

# -----------------
# Load CSV points
# -----------------
# Load CSV normally (use headers X, Y)
df = pd.read_csv("point_locations.csv")

# Create GeoDataFrame with CRS = WGS84
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs="EPSG:4326")

print(gdf.head())
print("Original CRS:", gdf.crs)

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
# Create buffered union of points
# -----------------
def make_buffer_union(gdf, crs, buffer_radius=1000):
    gdf_utm = gdf.to_crs(crs)
    #gdf_utm["buffer"] = gdf_utm.geometry.buffer(buffer_radius)
    buffers = gdf_utm.buffer(buffer_radius)
    union = buffers.unary_union
    return gdf_utm, [mapping(union)], buffers



# -----------------
# Compute NDVI array
# -----------------
def compute_ndvi(src, geom):
    red_masked, out_transform = mask(src, geom, crop=True, filled=True, indexes=4, nodata=np.nan)
    nir_masked, _ = mask(src, geom, crop=True, filled=True, indexes=8, nodata=np.nan)

    if red_masked.size == 0 or nir_masked.size == 0:
        return None, None

    # Force shape (1, rows, cols)
    if red_masked.ndim == 2: red_masked = red_masked[np.newaxis, :, :]
    if nir_masked.ndim == 2: nir_masked = nir_masked[np.newaxis, :, :]

    red = red_masked.astype("float32")[0]
    nir = nir_masked.astype("float32")[0]

    eps = 1e-10
    ndvi = (nir - red) / (nir + red + eps)

    # Back to shape (1, rows, cols) for rasterio
    ndvi = np.expand_dims(ndvi, axis=0)

    return ndvi, out_transform


# -----------------
# Save NDVI raster
# -----------------
def save_ndvi(ndvi, out_transform, src, out_path):
    rows, cols = ndvi.shape[1], ndvi.shape[2]
    profile = src.profile.copy()
    profile.update(
        count=1,
        dtype="float32",
        compress="lzw",
        transform=out_transform,
        height=rows,
        width=cols,
        nodata=np.nan,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(ndvi)


# -----------------
# Compute per-buffer NDVI stats
# -----------------
def compute_stats(src, gdf_utm, buffers):
    results = []
    for idx, row in gdf_utm.iterrows():
        geom = [mapping(buffers[idx])]
        try:
            red_masked, _ = mask(src, geom, crop=True, filled=True, indexes=4, nodata=np.nan)
            nir_masked, _ = mask(src, geom, crop=True, filled=True, indexes=8, nodata=np.nan)
        except ValueError:
            continue

        ndvi = (nir_masked.astype("float32") - red_masked.astype("float32")) / (
            nir_masked.astype("float32") + red_masked.astype("float32") + 1e-10
        )

        vals = ndvi[np.isfinite(ndvi)]
        if vals.size > 0:
            results.append({
                "point_id": idx,
                "mean_ndvi": float(vals.mean()),
                "min_ndvi": float(vals.min()),
                "max_ndvi": float(vals.max()),
                "std_ndvi": float(vals.std())
            })
    return results


# -----------------
# Main TIFF processor
# -----------------
def process_tiff(tiff_path, gdf, out_dir="ndvi_outputs", buffer_radius=1000):
    results = []

    with rasterio.open(tiff_path) as src:
        if src.count < 8:
            print(f"Skipping {tiff_path} (only {src.count} band(s))")
            return pd.DataFrame()

        # Reproject raster to match UTM CRS from points
        utm_crs = gdf.estimate_utm_crs()
        os.makedirs(out_dir, exist_ok=True)
        utm_path = os.path.join(out_dir, "tmp_utm.tif")
        reproject_to_utm(src, utm_crs, utm_path)

    with rasterio.open(utm_path) as src:
        # Reproject points to UTM and Buffer
        gdf_utm, geom_union, buffers = make_buffer_union(gdf, src.crs, buffer_radius)
        # Compute NDVI array
        ndvi, out_transform = compute_ndvi(src, geom_union)
        if ndvi is None:
            return pd.DataFrame()

        # Save NDVI raster
        ndvi_path = os.path.join(out_dir, os.path.basename(tiff_path).replace(".tif", "_ndvi.tif"))
        save_ndvi(ndvi, out_transform, src, ndvi_path)

        # Compute per-buffer NDVI stats
        stats = compute_stats(src, gdf_utm, buffers)
        for s in stats:
            s["tiff_file"] = os.path.basename(tiff_path)
        results.extend(stats)

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