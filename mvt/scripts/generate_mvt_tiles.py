# # # from __future__ import annotations
# # # from pathlib import Path
# # # from typing import Tuple, List
# # # import geopandas as gpd
# # # from shapely.geometry import box, mapping
# # # from shapely import ops
# # # import mapbox_vector_tile
# # # from tqdm import tqdm

# # # # =========================================================
# # # # CONFIGURATION
# # # # =========================================================
# # # OUTPUT_DIR = Path("roads_tiles_out")
# # # ZOOM_LEVELS = range(0, 8)
# # # EXTENT = 4096

# # # WORLD_BOUNDS = (
# # #     -20037508.342789244,  # minx
# # #     -20037508.342789244,  # miny
# # #      20037508.342789244,  # maxx
# # #      20037508.342789244   # maxy
# # # )

# # # # =========================================================
# # # # PYRAMID PARTITION FUNCTION
# # # # =========================================================
# # # def pyramid_partition(mbr: Tuple[float, float, float, float],
# # #                       min_zoom: int,
# # #                       max_zoom: int) -> List[Tuple[int, int, int]]:
# # #     """Return list of (z, x, y) tiles overlapping geometry MBR."""
# # #     xmin, ymin, xmax, ymax = mbr
# # #     WORLD_MINX, WORLD_MINY, WORLD_MAXX, WORLD_MAXY = WORLD_BOUNDS
# # #     WORLD_WIDTH = WORLD_MAXX - WORLD_MINX
# # #     WORLD_HEIGHT = WORLD_MAXY - WORLD_MINY

# # #     results = []
# # #     for z in range(min_zoom, max_zoom + 1):
# # #         n = 2 ** z
# # #         tile_w = WORLD_WIDTH / n
# # #         tile_h = WORLD_HEIGHT / n

# # #         xmin_c = max(WORLD_MINX, xmin)
# # #         xmax_c = min(WORLD_MAXX, xmax)
# # #         ymin_c = max(WORLD_MINY, ymin)
# # #         ymax_c = min(WORLD_MAXY, ymax)

# # #         x1 = int((xmin_c - WORLD_MINX) / tile_w)
# # #         x2 = int((xmax_c - WORLD_MINX) / tile_w)

# # #         # ✅ FIX: flip Y index to top-left origin (Mapbox convention)
# # #         y1 = int((WORLD_MAXY - ymax_c) / tile_h)
# # #         y2 = int((WORLD_MAXY - ymin_c) / tile_h)

# # #         for x in range(x1, x2 + 1):
# # #             for y in range(y1, y2 + 1):
# # #                 results.append((z, x, y))
# # #     return results

# # # # =========================================================
# # # # TILE BOUNDS (TOP-LEFT ORIGIN)
# # # # =========================================================
# # # def mercator_tile_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
# # #     """Return EPSG:3857 bounds for tile (z,x,y) with top-left origin."""
# # #     n = 2 ** z
# # #     tile_size = (WORLD_BOUNDS[2] - WORLD_BOUNDS[0]) / n
# # #     minx = WORLD_BOUNDS[0] + x * tile_size
# # #     maxx = WORLD_BOUNDS[0] + (x + 1) * tile_size
# # #     maxy = WORLD_BOUNDS[3] - y * tile_size
# # #     miny = WORLD_BOUNDS[3] - (y + 1) * tile_size
# # #     return (minx, miny, maxx, maxy)

# # # # =========================================================
# # # # SCALING FUNCTION
# # # # =========================================================
# # # def scale_to_tile(x: float, y: float, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
# # #     """Scale coordinates into [0, EXTENT] without vertical flipping."""
# # #     sx = (x - bounds[0]) / (bounds[2] - bounds[0]) * EXTENT
# # #     sy = (y - bounds[1]) / (bounds[3] - bounds[1]) * EXTENT
# # #     return sx, sy

# # # # =========================================================
# # # # TILE GENERATION
# # # # =========================================================
# # # def generate_tiles(gdf: gpd.GeoDataFrame):
# # #     print(f"Input GeoDataFrame: {len(gdf)} features")
# # #     print("Original CRS:", gdf.crs)

# # #     if gdf.crs is None:
# # #         print("⚠️  No CRS found, assuming EPSG:4326")
# # #         gdf = gdf.set_crs(4326)

# # #     print("Reprojecting to EPSG:3857 ...")
# # #     gdf = gdf.to_crs(3857)

# # #     print("Cleaning geometries ...")
# # #     gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
# # #     print(f"After cleaning: {len(gdf)} valid geometries")

# # #     print("Geometry types:")
# # #     print(gdf.geom_type.value_counts())
# # #     print(f"Data bounds (EPSG:3857): {gdf.total_bounds}")

# # #     # =========================================================
# # #     # Partition geometries
# # #     # =========================================================
# # #     print("\n=== Partitioning geometries into tiles ===")
# # #     tile_map: dict[Tuple[int, int, int], list] = {}
# # #     for idx, geom in tqdm(list(enumerate(gdf.geometry)), desc="Assigning features"):
# # #         xmin, ymin, xmax, ymax = geom.bounds
# # #         tiles = pyramid_partition((xmin, ymin, xmax, ymax),
# # #                                   min(ZOOM_LEVELS),
# # #                                   max(ZOOM_LEVELS))
# # #         for z, x, y in tiles:
# # #             tile_map.setdefault((z, x, y), []).append(idx)
# # #     print(f"Total tiles with data: {len(tile_map)}")

# # #     # =========================================================
# # #     # Generate MVTs
# # #     # =========================================================
# # #     for z in ZOOM_LEVELS:
# # #         print(f"\n=== Generating tiles for zoom {z} ===")
# # #         tiles_written = 0
# # #         zoom_tiles = {k: v for k, v in tile_map.items() if k[0] == z}

# # #         for (z, x, y), indices in tqdm(zoom_tiles.items(), desc=f"Zoom {z}"):
# # #             bounds = mercator_tile_bounds(z, x, y)
# # #             tile_box = box(*bounds)
# # #             subset = gdf.iloc[indices]
# # #             if subset.empty:
# # #                 continue

# # #             subset = subset.copy()
# # #             subset["geom_clip"] = subset.geometry.intersection(tile_box)

# # #             features = []
# # #             for idx, row in subset.iterrows():
# # #                 geom = row["geom_clip"]
# # #                 if geom.is_empty:
# # #                     continue

# # #                 geom_scaled = ops.transform(lambda x, y, z=None: scale_to_tile(x, y, bounds), geom)
# # #                 features.append({
# # #                     "geometry": mapping(geom_scaled),
# # #                     "properties": {k: str(v) for k, v in row.items()
# # #                                    if k not in ["geometry", "geom_clip"]},
# # #                     "id": int(idx) if isinstance(idx, (int, float)) else None
# # #                 })

# # #             if not features:
# # #                 continue

# # #             layer = {"name": "layer0", "features": features, "extent": EXTENT}
# # #             tile_data = mapbox_vector_tile.encode([layer])

# # #             out_dir = OUTPUT_DIR / f"{z}/{x}"
# # #             out_dir.mkdir(parents=True, exist_ok=True)
# # #             with open(out_dir / f"{y}.mvt", "wb") as f:
# # #                 f.write(tile_data)
# # #             tiles_written += 1

# # #         print(f"✅ Zoom {z}: wrote {tiles_written} non-empty tiles")

# # #     print("\n✅ Done! Vector tiles written to:", OUTPUT_DIR)

# # # # =========================================================
# # # # ENTRY POINT
# # # # =========================================================
# # # if __name__ == "__main__":
# # #     parquet_path = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/original_datasets/highways/roads.parquet"
# # #     print(f"Reading {parquet_path} ...")
# # #     gdf = gpd.read_parquet(parquet_path)
# # #     generate_tiles(gdf)


# # from __future__ import annotations
# # from pathlib import Path
# # from typing import Tuple, List
# # import geopandas as gpd
# # from shapely.geometry import box, mapping
# # from shapely import ops
# # import mapbox_vector_tile
# # from tqdm import tqdm
# # import numpy as np
# # import math

# # # =========================================================
# # # CONFIGURATION
# # # =========================================================
# # OUTPUT_DIR = Path("roads_tiles_out")
# # ZOOM_LEVELS = range(0, 8)
# # EXTENT = 4096
# # HIST_THRESHOLD = 1  # ✅ minimum histogram value to render
# # count = 0
# # WORLD_BOUNDS = (
# #     -20037508.342789244, -20037508.342789244,
# #      20037508.342789244,  20037508.342789244
# # )

# # # =========================================================
# # # HISTOGRAM LOOKUP
# # # =========================================================
# # def hist_value(hist: np.ndarray, z: int, x: int, y: int) -> float:
# #     """Return histogram value at any zoom level relative to histogram's base resolution."""
# #     base_size = hist.shape[0]
# #     base_level = int(round(math.log2(base_size)))
# #     if base_size != 2 ** base_level:
# #         raise ValueError(f"Histogram size {base_size} not power of two.")

# #     # same level
# #     if z == base_level:
# #         return float(hist[y, x]) if (0 <= y < base_size and 0 <= x < base_size) else 0.0

# #     # coarser
# #     elif z < base_level:
# #         factor = 2 ** (base_level - z)
# #         x0, y0 = x * factor, y * factor
# #         x1, y1 = min(x0 + factor, base_size), min(y0 + factor, base_size)
# #         return float(hist[y0:y1, x0:x1].sum())

# #     # finer
# #     else:
# #         parent_x = x // (2 ** (z - base_level))
# #         parent_y = y // (2 ** (z - base_level))
# #         parent_val = hist[parent_y, parent_x]
# #         return parent_val / (4 ** (z - base_level))

# # # =========================================================
# # # PYRAMID PARTITION FUNCTION
# # # =========================================================
# # def pyramid_partition(mbr: Tuple[float, float, float, float],
# #                       min_zoom: int,
# #                       max_zoom: int) -> List[Tuple[int, int, int]]:
# #     xmin, ymin, xmax, ymax = mbr
# #     WORLD_MINX, WORLD_MINY, WORLD_MAXX, WORLD_MAXY = WORLD_BOUNDS
# #     WORLD_WIDTH = WORLD_MAXX - WORLD_MINX
# #     WORLD_HEIGHT = WORLD_MAXY - WORLD_MINY

# #     results = []
# #     for z in range(min_zoom, max_zoom + 1):
# #         n = 2 ** z
# #         tile_w = WORLD_WIDTH / n
# #         tile_h = WORLD_HEIGHT / n

# #         xmin_c = max(WORLD_MINX, xmin)
# #         xmax_c = min(WORLD_MAXX, xmax)
# #         ymin_c = max(WORLD_MINY, ymin)
# #         ymax_c = min(WORLD_MAXY, ymax)

# #         x1 = int((xmin_c - WORLD_MINX) / tile_w)
# #         x2 = int((xmax_c - WORLD_MINX) / tile_w)
# #         y1 = int((WORLD_MAXY - ymax_c) / tile_h)
# #         y2 = int((WORLD_MAXY - ymin_c) / tile_h)

# #         for x in range(x1, x2 + 1):
# #             for y in range(y1, y2 + 1):
# #                 results.append((z, x, y))
# #     return results

# # # =========================================================
# # # TILE BOUNDS + SCALE
# # # =========================================================
# # def mercator_tile_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
# #     n = 2 ** z
# #     tile_size = (WORLD_BOUNDS[2] - WORLD_BOUNDS[0]) / n
# #     minx = WORLD_BOUNDS[0] + x * tile_size
# #     maxx = WORLD_BOUNDS[0] + (x + 1) * tile_size
# #     maxy = WORLD_BOUNDS[3] - y * tile_size
# #     miny = WORLD_BOUNDS[3] - (y + 1) * tile_size
# #     return (minx, miny, maxx, maxy)

# # def scale_to_tile(x: float, y: float, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
# #     sx = (x - bounds[0]) / (bounds[2] - bounds[0]) * EXTENT
# #     sy = (y - bounds[1]) / (bounds[3] - bounds[1]) * EXTENT
# #     return sx, sy

# # # =========================================================
# # # TILE GENERATION
# # # =========================================================
# # def generate_tiles(gdf: gpd.GeoDataFrame, hist: np.ndarray):
# #     print(f"Input GeoDataFrame: {len(gdf)} features")
# #     # Use the module-level counter to track processed tiles
# #     global count
# #     if gdf.crs is None:
# #         gdf = gdf.set_crs(4326)
# #     gdf = gdf.to_crs(3857)
# #     gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
# #     print(f"After cleaning: {len(gdf)} valid geometries")

# #     tile_map: dict[Tuple[int, int, int], list] = {}
# #     for idx, geom in tqdm(list(enumerate(gdf.geometry)), desc="Assigning features"):
# #         xmin, ymin, xmax, ymax = geom.bounds
# #         tiles = pyramid_partition((xmin, ymin, xmax, ymax),
# #                                   min(ZOOM_LEVELS),
# #                                   max(ZOOM_LEVELS))
# #         for z, x, y in tiles:
# #             tile_map.setdefault((z, x, y), []).append(idx)

# #     for z in ZOOM_LEVELS:
# #         print(f"\n=== Generating tiles for zoom {z} ===")
# #         tiles_written = 0
# #         zoom_tiles = {k: v for k, v in tile_map.items() if k[0] == z}
# #         tolerance = 1000 / (2 ** z)

# #         for (z, x, y), indices in tqdm(zoom_tiles.items(), desc=f"Zoom {z}"):
# #             # 🔎 Histogram filter
# #             hist_value_at_tile = hist_value(hist, z, x, y)
# #             print(f"Tile z={z}, x={x}, y={y} has hist value {hist_value_at_tile}")
# #             if hist_value_at_tile < HIST_THRESHOLD:
# #                 continue
# #             count += 1
# #             bounds = mercator_tile_bounds(z, x, y)
# #             tile_box = box(*bounds)
# #             subset = gdf.iloc[indices]
# #             if subset.empty:
# #                 continue

# #             subset = subset.copy()
# #             subset["geom_clip"] = subset.geometry.intersection(tile_box)

# #             features = []
# #             for idx, row in subset.iterrows():
# #                 geom = row["geom_clip"]
# #                 if geom.is_empty:
# #                     continue

# #                 geom = geom.simplify(tolerance, preserve_topology=True)
# #                 if geom.is_empty:
# #                     continue

# #                 geom_scaled = ops.transform(lambda x, y, z=None: scale_to_tile(x, y, bounds), geom)
# #                 features.append({
# #                     "geometry": mapping(geom_scaled),
# #                     "properties": {k: str(v) for k, v in row.items()
# #                                    if k not in ["geometry", "geom_clip"]},
# #                     "id": int(idx) if isinstance(idx, (int, float)) else None
# #                 })

# #             if not features:
# #                 continue

# #             layer = {"name": "layer0", "features": features, "extent": EXTENT}
# #             tile_data = mapbox_vector_tile.encode([layer])

# #             out_dir = OUTPUT_DIR / f"{z}/{x}"
# #             out_dir.mkdir(parents=True, exist_ok=True)
# #             with open(out_dir / f"{y}.mvt", "wb") as f:
# #                 f.write(tile_data)
# #             tiles_written += 1

# #         print(f"✅ Zoom {z}: wrote {tiles_written} non-empty tiles")

# #     print("\n✅ Done! Vector tiles written to:", OUTPUT_DIR)

# # # =========================================================
# # # ENTRY POINT
# # # =========================================================
# # if __name__ == "__main__":
# #     parquet_path = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/original_datasets/us_only.parquet"
# #     hist_path = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/tile-geoparquet/tile_geoparquet/output/global.npy"
# #     print(f"Reading {parquet_path} ...")
# #     gdf = gpd.read_parquet(parquet_path)
# #     hist = np.load(hist_path)
# #     generate_tiles(gdf, hist)
# #     print(f"Total tiles processed: {count}")

# from __future__ import annotations
# from pathlib import Path
# from typing import Tuple, List
# import geopandas as gpd
# from shapely.geometry import (
#     box, mapping, GeometryCollection
# )
# from shapely import ops
# from shapely import make_valid
# import mapbox_vector_tile
# from tqdm import tqdm
# import numpy as np

# # =========================================================
# # CONFIGURATION
# # =========================================================
# OUTPUT_DIR = Path("roads_tiles_out")
# ZOOM_LEVELS = range(0, 8)
# EXTENT = 4096
# HIST_THRESHOLD = 0  # Minimum histogram mass for tile output

# WORLD_BOUNDS = (
#     -20037508.342789244, -20037508.342789244,
#      20037508.342789244,  20037508.342789244,
# )
# WORLD_MINX, WORLD_MINY, WORLD_MAXX, WORLD_MAXY = WORLD_BOUNDS
# WORLD_W = WORLD_MAXX - WORLD_MINX
# WORLD_H = WORLD_MAXY - WORLD_MINY


# # =========================================================
# # FIX GEOMETRY COLLECTIONS FOR MVT
# # =========================================================
# def explode_geometry(g):
#     """Convert GeometryCollection → list of valid geometries."""
#     if g.is_empty:
#         return []
#     if not isinstance(g, GeometryCollection):
#         return [g]

#     out = []
#     for part in g.geoms:
#         if part.is_empty:
#             continue
#         out.extend(explode_geometry(part))
#     return out


# # =========================================================
# # HISTOGRAM → TILE MAPPING (CORRECT)
# # =========================================================
# # def hist_value_for_tile(hist: np.ndarray, z: int, x: int, y: int) -> float:
# #     H, W = hist.shape  # histogram resolution

# #     minx, miny, maxx, maxy = mercator_tile_bounds(z, x, y)

# #     j0 = int((minx - WORLD_MINX) / WORLD_W * W)
# #     j1 = int((maxx - WORLD_MINX) / WORLD_W * W)

# #     i0 = int((WORLD_MAXY - maxy) / WORLD_H * H)
# #     i1 = int((WORLD_MAXY - miny) / WORLD_H * H)

# #     i0 = max(0, min(i0, H))
# #     i1 = max(0, min(i1, H))
# #     j0 = max(0, min(j0, W))
# #     j1 = max(0, min(j1, W))

# #     if i0 >= i1 or j0 >= j1:
# #         return 0.0

# #     return float(hist[i0:i1, j0:j1].sum())

# def hist_value(hist: np.ndarray, z: int, x: int, y: int) -> float:
#     H = hist.shape[0]
#     baseZ = int(math.log2(H))  # histogram zoom level

#     # ------------------------------
#     # Case 1: Same zoom → direct lookup
#     # ------------------------------
#     if z == baseZ:
#         if 0 <= x < H and 0 <= y < H:
#             return float(hist[y, x])
#         return 0.0

#     # ------------------------------
#     # Case 2: Coarser zoom (z < baseZ)
#     # Need to sum a block of size 2^(baseZ - z)
#     # ------------------------------
#     if z < baseZ:
#         factor = 1 << (baseZ - z)  # 2^(baseZ - z)

#         x0 = x * factor
#         y0 = y * factor
#         x1 = min(x0 + factor, H)
#         y1 = min(y0 + factor, H)

#         if x0 >= H or y0 >= H:
#             return 0.0

#         return float(hist[y0:y1, x0:x1].sum())

#     # ------------------------------
#     # Case 3: Finer zoom (z > baseZ)
#     # Need to split parent histogram cell
#     # ------------------------------
#     finer_factor = 1 << (z - baseZ)  # 2^(z - baseZ)

#     parent_x = x // finer_factor
#     parent_y = y // finer_factor

#     if 0 <= parent_x < H and 0 <= parent_y < H:
#         # Split the parent pixel into 4^(z-baseZ) pieces
#         return float(hist[parent_y, parent_x]) / (finer_factor * finer_factor)

#     return 0.0



# # =========================================================
# # TILE SYSTEM
# # =========================================================
# def pyramid_partition(mbr, min_zoom, max_zoom):
#     xmin, ymin, xmax, ymax = mbr
#     results = []

#     for z in range(min_zoom, max_zoom + 1):
#         n = 2 ** z
#         tile_w = WORLD_W / n
#         tile_h = WORLD_H / n

#         xmin_c = max(WORLD_MINX, xmin)
#         xmax_c = min(WORLD_MAXX, xmax)
#         ymin_c = max(WORLD_MINY, ymin)
#         ymax_c = min(WORLD_MAXY, ymax)

#         x1 = int((xmin_c - WORLD_MINX) / tile_w)
#         x2 = int((xmax_c - WORLD_MINX) / tile_w)

#         y1 = int((WORLD_MAXY - ymax_c) / tile_h)
#         y2 = int((WORLD_MAXY - ymin_c) / tile_h)

#         for x in range(x1, x2 + 1):
#             for y in range(y1, y2 + 1):
#                 results.append((z, x, y))

#     return results


# def mercator_tile_bounds(z, x, y):
#     n = 2 ** z
#     tile_size = WORLD_W / n

#     minx = WORLD_MINX + x * tile_size
#     maxx = minx + tile_size

#     maxy = WORLD_MAXY - y * tile_size
#     miny = maxy - tile_size

#     return (minx, miny, maxx, maxy)


# def scale_to_tile(x, y, bounds):
#     minx, miny, maxx, maxy = bounds
#     sx = (x - minx) / (maxx - minx) * EXTENT
#     sy = (y - miny) / (maxy - miny) * EXTENT
#     return sx, sy


# # =========================================================
# # TILE GENERATION
# # =========================================================
# def generate_tiles(gdf: gpd.GeoDataFrame, hist: np.ndarray):
#     print(f"Input GeoDataFrame: {len(gdf)} features")

#     if gdf.crs is None:
#         print("⚠ Warning: No CRS, assuming EPSG:4326")
#         gdf = gdf.set_crs(4326)

#     gdf = gdf.to_crs(3857)
#     gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

#     print("Fixing invalid geometries...")
#     gdf["geometry"] = gdf.geometry.apply(make_valid).buffer(0)

#     print(f"{len(gdf)} valid geometries remain")

#     # ---- Partition into tiles ----
#     print("\nAssigning features to tiles...")
#     tile_map = {}

#     for idx, geom in tqdm(list(enumerate(gdf.geometry))):
#         xmin, ymin, xmax, ymax = geom.bounds
#         tiles = pyramid_partition((xmin, ymin, xmax, ymax),
#                                   min(ZOOM_LEVELS), max(ZOOM_LEVELS))
#         for tile_id in tiles:
#             tile_map.setdefault(tile_id, []).append(idx)

#     # ---- Generate vector tiles ----
#     for z in ZOOM_LEVELS:
#         print(f"\n=== Generating tiles for zoom {z} ===")

#         tiles_for_z = {k: v for k, v in tile_map.items() if k[0] == z}
#         tiles_written = 0

#         tolerance = 1000 / (2 ** z)

#         for (z, x, y), indices in tqdm(tiles_for_z.items(), desc=f"Zoom {z}"):

#             # histogram-based culling
#             if hist_value_for_tile(hist, z, x, y) < HIST_THRESHOLD:
#                 continue

#             bounds = mercator_tile_bounds(z, x, y)
#             tile_bounds_geom = box(*bounds)

#             subset = gdf.iloc[indices].copy()
#             subset["geom_clip"] = subset.geometry.intersection(tile_bounds_geom)

#             features = []

#             for idx2, row in subset.iterrows():
#                 geom = row["geom_clip"]
#                 if geom.is_empty:
#                     continue

#                 geom = geom.simplify(tolerance, preserve_topology=True)
#                 if geom.is_empty:
#                     continue

#                 # ---- handle geometry collections ----
#                 parts = explode_geometry(geom)
#                 if not parts:
#                     continue

#                 for part in parts:
#                     geom_scaled = ops.transform(
#                         lambda a, b, c=None: scale_to_tile(a, b, bounds),
#                         part
#                     )
#                     features.append({
#                         "geometry": mapping(geom_scaled),
#                         "properties": {
#                             k: str(v) for k, v in row.items()
#                             if k not in ("geometry", "geom_clip")
#                         },
#                         "id": int(idx2),
#                     })

#             if not features:
#                 continue

#             layer = {
#                 "name": "layer0",
#                 "features": features,
#                 "extent": EXTENT,
#             }

#             tile_data = mapbox_vector_tile.encode([layer])

#             out_dir = OUTPUT_DIR / f"{z}/{x}"
#             out_dir.mkdir(parents=True, exist_ok=True)

#             with open(out_dir / f"{y}.mvt", "wb") as f:
#                 f.write(tile_data)

#             tiles_written += 1

#         print(f"✔ Zoom {z}: wrote {tiles_written} tiles")

#     print("\n🎉 Done! Vector tiles written to:", OUTPUT_DIR)


# # =========================================================
# # ENTRY POINT
# # =========================================================
# if __name__ == "__main__":
#     parquet_path = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/original_datasets/us_only.parquet"
#     hist_path = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/tile-geoparquet/tile_geoparquet/output/global.npy"

#     print(f"Loading dataset: {parquet_path}")
#     gdf = gpd.read_parquet(parquet_path)

#     print(f"Loading histogram: {hist_path}")
#     hist = np.load(hist_path)

#     generate_tiles(gdf, hist)

from __future__ import annotations
from pathlib import Path
import logging
from typing import Tuple, List
import geopandas as gpd
from shapely.geometry import (
    box, mapping, GeometryCollection
)
from shapely import ops
from shapely import make_valid
import mapbox_vector_tile
from tqdm import tqdm
import numpy as np
import math


# =========================================================
# LOGGING SETUP
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
log = logging.getLogger("hist_debug")


# =========================================================
# CONFIGURATION
# =========================================================
OUTPUT_DIR = Path("roads_tiles_out")
ZOOM_LEVELS = range(0, 3)
EXTENT = 4096
HIST_THRESHOLD = 0  # Minimum histogram mass for tile output

WORLD_BOUNDS = (
    -20037508.342789244, -20037508.342789244,
     20037508.342789244,  20037508.342789244,
)
WORLD_MINX, WORLD_MINY, WORLD_MAXX, WORLD_MAXY = WORLD_BOUNDS
WORLD_W = WORLD_MAXX - WORLD_MINX
WORLD_H = WORLD_MAXY - WORLD_MINY


# =========================================================
# HELPERS
# =========================================================
def explode_geometry(g):
    """Convert GeometryCollection → list of valid geometries."""
    if g.is_empty:
        return []
    if not isinstance(g, GeometryCollection):
        return [g]

    out = []
    for part in g.geoms:
        if part.is_empty:
            continue
        out.extend(explode_geometry(part))
    return out


# =========================================================
# HISTOGRAM PYRAMID LOOKUP
# =========================================================
def hist_value(hist: np.ndarray, z: int, x: int, y: int) -> float:
    """
    Compute pyramid histogram value:
    - z == baseZ → direct lookup
    - z < baseZ → aggregate block of parent histogram
    - z > baseZ → subdivide parent histogram value
    """
    H = hist.shape[0]
    baseZ = int(math.log2(H))

    # DEBUG ONLY once
    if z == 0 and x == 0 and y == 0:
        log.info(f"[DEBUG] Histogram resolution = {H}x{H}, baseZ = {baseZ}")

    # ------------------------------
    # Case 1: Same zoom → direct lookup
    # ------------------------------
    if z == baseZ:
        if 0 <= x < H and 0 <= y < H:
            v = float(hist[y, x])
            log.debug(f"[BASE] z={z} x={x} y={y} → {v}")
            return v
        log.debug(f"[BASE] z={z} x={x} y={y} → out-of-bounds")
        return 0.0

    # ------------------------------
    # Case 2: Coarser zoom
    # ------------------------------
    if z < baseZ:
        factor = 1 << (baseZ - z)
        x0, y0 = x * factor, y * factor
        x1, y1 = min(x0 + factor, H), min(y0 + factor, H)

        if x0 >= H or y0 >= H:
            return 0.0

        block_sum = float(hist[y0:y1, x0:x1].sum())

        log.debug(
            f"[COARSER] z={z} x={x} y={y} factor={factor} "
            f"block=({x0}:{x1},{y0}:{y1}) sum={block_sum}"
        )
        return block_sum

    # ------------------------------
    # Case 3: Finer zoom
    # ------------------------------
    finer_factor = 1 << (z - baseZ)
    parent_x = x // finer_factor
    parent_y = y // finer_factor

    if not (0 <= parent_x < H and 0 <= parent_y < H):
        log.debug(f"[FINER] z={z} x={x} y={y} → off-parent")
        return 0.0

    parent_val = float(hist[parent_y, parent_x])
    split_val = parent_val / (finer_factor * finer_factor)

    log.debug(
        f"[FINER] z={z} x={x} y={y} parent=({parent_x},{parent_y}) "
        f"parent_val={parent_val} finer_factor={finer_factor} → {split_val}"
    )
    return split_val


# =========================================================
# TILE SYSTEM
# =========================================================
def pyramid_partition(mbr, min_zoom, max_zoom):
    xmin, ymin, xmax, ymax = mbr
    results = []

    for z in range(min_zoom, max_zoom + 1):
        n = 2 ** z
        tile_w = WORLD_W / n
        tile_h = WORLD_H / n

        xmin_c = max(WORLD_MINX, xmin)
        xmax_c = min(WORLD_MAXX, xmax)
        ymin_c = max(WORLD_MINY, ymin)
        ymax_c = min(WORLD_MAXY, ymax)

        x1 = int((xmin_c - WORLD_MINX) / tile_w)
        x2 = int((xmax_c - WORLD_MINX) / tile_w)
        y1 = int((WORLD_MAXY - ymax_c) / tile_h)
        y2 = int((WORLD_MAXY - ymin_c) / tile_h)

        # FIX: clamp to valid tile index range
        x1 = max(0, min(x1, n - 1))
        x2 = max(0, min(x2, n - 1))
        y1 = max(0, min(y1, n - 1))
        y2 = max(0, min(y2, n - 1))

        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                results.append((z, x, y))

    return results



def mercator_tile_bounds(z, x, y):
    n = 2 ** z
    tile_size = WORLD_W / n

    minx = WORLD_MINX + x * tile_size
    maxx = minx + tile_size

    maxy = WORLD_MAXY - y * tile_size
    miny = maxy - tile_size

    return (minx, miny, maxx, maxy)


def scale_to_tile(x, y, bounds):
    minx, miny, maxx, maxy = bounds
    sx = (x - minx) / (maxx - minx) * EXTENT
    sy = (y - miny) / (maxy - miny) * EXTENT
    return sx, sy


# =========================================================
# TILE GENERATION
# =========================================================
def generate_tiles(gdf: gpd.GeoDataFrame, hist: np.ndarray):
    log.info(f"Input GeoDataFrame size: {len(gdf)} features")

    # Histogram check
    H = hist.shape[0]
    baseZ = int(math.log2(H))
    if H != 1 << baseZ:
        raise ValueError(f"Histogram size {H} is not power-of-two")
    log.info(f"Histogram size OK → {H}x{H}, base zoom = {baseZ}")

    # CRS + cleaning
    if gdf.crs is None:
        log.warning("No CRS detected; assuming EPSG:4326")
        gdf = gdf.set_crs(4326)

    gdf = gdf.to_crs(3857)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    log.info(f"After cleaning: {len(gdf)} features")

    log.info("Fixing invalid geometries…")
    gdf["geometry"] = gdf.geometry.apply(make_valid).buffer(0)

    log.info("Assigning features to spatial tiles…")
    tile_map = {}

    for idx, geom in tqdm(list(enumerate(gdf.geometry))):
        xmin, ymin, xmax, ymax = geom.bounds
        tiles = pyramid_partition((xmin, ymin, xmax, ymax),
                                  min(ZOOM_LEVELS), max(ZOOM_LEVELS))
        for tile_id in tiles:
            tile_map.setdefault(tile_id, []).append(idx)

    # ---------------------------------------------------------
    # TILE GENERATION
    # ---------------------------------------------------------
    for z in ZOOM_LEVELS:
        log.info(f"\n=== Generating tiles for zoom {z} ===")
        tiles_for_z = {k: v for k, v in tile_map.items() if k[0] == z}

        tiles_written = 0
        tolerance = 1000 / (2 ** z)

        for (z, x, y), indices in tqdm(tiles_for_z.items(), desc=f"Zoom {z}"):

            hv = hist_value(hist, z, x, y)

            if hv < HIST_THRESHOLD:
                log.debug(f"[SKIP] z={z} x={x} y={y} → hist={hv}")
                continue

            log.info(f"[KEEP] z={z} x={x} y={y} hist={hv}")

            bounds = mercator_tile_bounds(z, x, y)
            tile_geom = box(*bounds)

            subset = gdf.iloc[indices].copy()
            subset["geom_clip"] = subset.geometry.intersection(tile_geom)

            features = []

            for idx2, row in subset.iterrows():
                geom = row["geom_clip"]
                if geom.is_empty:
                    continue

                geom = geom.simplify(tolerance, preserve_topology=True)
                if geom.is_empty:
                    continue

                parts = explode_geometry(geom)
                if not parts:
                    continue

                for part in parts:
                    geom_scaled = ops.transform(
                        lambda a, b, c=None: scale_to_tile(a, b, bounds),
                        part
                    )

                    features.append({
                        "geometry": mapping(geom_scaled),
                        "properties": {
                            k: str(v)
                            for k, v in row.items()
                            if k not in ("geometry", "geom_clip")
                        },
                        "id": int(idx2),
                    })

            if not features:
                continue

            # Encode tile
            layer = {
                "name": "layer0",
                "features": features,
                "extent": EXTENT,
            }

            tile_data = mapbox_vector_tile.encode([layer])

            out_dir = OUTPUT_DIR / f"{z}/{x}"
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_dir / f"{y}.mvt", "wb") as f:
                f.write(tile_data)

            tiles_written += 1

        # Summary logs
        log.info(f"✔ Zoom {z}: wrote {tiles_written} tiles out of {len(tiles_for_z)}")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    parquet_path = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/original_datasets/us_only.parquet"
    hist_path = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/tile-geoparquet/tile_geoparquet/output/global.npy"

    log.info(f"Loading dataset: {parquet_path}")
    gdf = gpd.read_parquet(parquet_path)

    log.info(f"Loading histogram: {hist_path}")
    hist = np.load(hist_path)

    generate_tiles(gdf, hist)

    log.info("🎉 DONE — tiles written to: roads_tiles_out")
