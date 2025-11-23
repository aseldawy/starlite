from __future__ import annotations
from pathlib import Path
import logging
from typing import List
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
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
log = logging.getLogger("hist_tiler")


# =========================================================
# CONFIGURATION
# =========================================================
OUTPUT_DIR = Path("roads_tiles_out")
ZOOM_LEVELS = range(0, 4)
EXTENT = 4096
HIST_THRESHOLD = 0  # Minimum histogram mass for tile output

WORLD_MINX = -20037508.342789244
WORLD_MINY = -20037508.342789244
WORLD_MAXX = 20037508.342789244
WORLD_MAXY = 20037508.342789244

WORLD_W = WORLD_MAXX - WORLD_MINX
WORLD_H = WORLD_MAXY - WORLD_MINY


# =========================================================
# GEOMETRY HELPERS
# =========================================================
def explode_geometry(g):
    """Convert GeometryCollection → list of valid geometries."""
    if g.is_empty:
        return []
    if not isinstance(g, GeometryCollection):
        return [g]

    out = []
    for part in g.geoms:
        if not part.is_empty:
            out.extend(explode_geometry(part))
    return out


# =========================================================
# HISTOGRAM PYRAMID LOOKUP
# =========================================================
def hist_value(hist: np.ndarray, z: int, x: int, y: int) -> float:
    """
    Histogram pyramid logic:
    - z = baseZ → direct lookup
    - z < baseZ → sum parent cells
    - z > baseZ → subdivide parent cell
    """
    H = hist.shape[0]
    baseZ = int(math.log2(H))

    # Base level → direct
    if z == baseZ:
        if 0 <= x < H and 0 <= y < H:
            return float(hist[y, x])
        return 0.0

    # Coarser
    if z < baseZ:
        factor = 1 << (baseZ - z)
        x0 = x * factor
        y0 = y * factor
        x1 = min(x0 + factor, H)
        y1 = min(y0 + factor, H)

        if x0 >= H or y0 >= H:
            return 0.0

        return float(hist[y0:y1, x0:x1].sum())

    # Finer
    factor = 1 << (z - baseZ)

    parent_x = x // factor
    parent_y = y // factor

    if not (0 <= parent_x < H and 0 <= parent_y < H):
        return 0.0

    return float(hist[parent_y, parent_x]) / (factor * factor)

def hist_value_from_prefix(prefix: np.ndarray, z: int, x: int, y: int) -> float:
    H, W = prefix.shape

    minx, miny, maxx, maxy = mercator_tile_bounds(z, x, y)

    j0 = int((minx - WORLD_MINX) / WORLD_W * W)
    j1 = int((maxx - WORLD_MINX) / WORLD_W * W) - 1

    i0 = int((WORLD_MAXY - maxy) / WORLD_H * H)
    i1 = int((WORLD_MAXY - miny) / WORLD_H * H) - 1

    j0 = max(0, min(j0, W - 1))
    j1 = max(0, min(j1, W - 1))
    i0 = max(0, min(i0, H - 1))
    i1 = max(0, min(i1, H - 1))

    if i0 > i1 or j0 > j1:
        return 0.0

    A = prefix[i1, j1]
    B = prefix[i0 - 1, j1] if i0 > 0 else 0
    C = prefix[i1, j0 - 1] if j0 > 0 else 0
    D = prefix[i0 - 1, j0 - 1] if (i0 > 0 and j0 > 0) else 0

    return A - B - C + D



# =========================================================
# TILE BOUNDS + SCALING
# =========================================================
def mercator_tile_bounds(z: int, x: int, y: int):
    """Return EPSG:3857 bounds for tile (z, x, y)."""
    n = 1 << z
    tile_w = WORLD_W / n
    tile_h = WORLD_H / n

    minx = WORLD_MINX + x * tile_w
    maxx = minx + tile_w

    maxy = WORLD_MAXY - y * tile_h
    miny = maxy - tile_h

    return (minx, miny, maxx, maxy)


def scale_to_tile(x, y, bounds):
    """Scale EPSG:3857 coordinates into tile-local [0, EXTENT]."""
    minx, miny, maxx, maxy = bounds
    sx = (x - minx) / (maxx - minx) * EXTENT
    sy = (y - miny) / (maxy - miny) * EXTENT
    return sx, sy



# =========================================================
# MAIN TILE GENERATOR — NEW, CLEAN, CORRECT
# =========================================================
def generate_tiles(gdf: gpd.GeoDataFrame, hist: np.ndarray, prefix_hist: np.ndarray):
    log.info(f"Loaded {len(gdf)} geometries")

    # Ensure CRS
    if gdf.crs is None:
        log.warning("No CRS found; assuming EPSG:4326")
        gdf = gdf.set_crs(4326)

    gdf = gdf.to_crs(3857)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

    # Fix invalid geometry
    log.info("Fixing invalid geometries…")
    gdf["geometry"] = gdf.geometry.apply(make_valid).buffer(0)

    # Histogram stats
    H = hist.shape[0]
    baseZ = int(math.log2(H))
    if H != (1 << baseZ):
        raise ValueError("Histogram is not power of two")
    log.info(f"Histogram OK ({H}x{H}), base zoom = {baseZ}")

    # Loop over all tiles
    for z in ZOOM_LEVELS:
        log.info(f"\n=== ZOOM {z} ===")
        tiles_written = 0

        tiles_per_side = 1 << z
        tolerance = 1000 / (2 ** z)

        for x in range(tiles_per_side):
            for y in range(tiles_per_side):

                # histogram culling
                hv = hist_value(hist, z, x, y)
                prefix_hv = hist_value_from_prefix(prefix_hist, z, x, y)
                print(f"Tile z={z} x={x} y={y} hist={hv} prefix_hist={prefix_hv}")

                if(prefix_hv != hv):
                    exit("mismatch between hist and prefix hist")
                if hv < HIST_THRESHOLD:
                    continue

                log.info(f"[KEEP] z={z} x={x} y={y} hist={hv}")

                # compute tile bounds
                bounds = mercator_tile_bounds(z, x, y)
                tile_geom = box(*bounds)

                # spatial intersection test
                subset = gdf[gdf.intersects(tile_geom)]
                if subset.empty:
                    continue

                subset = subset.copy()
                subset["geom_clip"] = subset.geometry.intersection(tile_geom)

                features = []

                for idx2, row in subset.iterrows():
                    geom = row["geom_clip"]
                    if geom.is_empty:
                        continue

                    geom = geom.simplify(tolerance, preserve_topology=False)
                    if geom.is_empty:
                        continue

                    parts = explode_geometry(geom)

                    for part in parts:
                        geom_scaled = ops.transform(
                            lambda a, b, c=None: scale_to_tile(a, b, bounds),
                            part,
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

                # write tile
                layer = {"name": "layer0", "features": features, "extent": EXTENT}
                tile_data = mapbox_vector_tile.encode([layer])

                out_dir = OUTPUT_DIR / f"{z}/{x}"
                out_dir.mkdir(parents=True, exist_ok=True)

                with open(out_dir / f"{y}.mvt", "wb") as f:
                    f.write(tile_data)

                tiles_written += 1

        log.info(f"✔ Zoom {z}: wrote {tiles_written} tiles")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    parquet_path = (
        "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/original_datasets/us_only.parquet"
    )
    hist_path = (
        "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/tile-geoparquet/tile_geoparquet/output/global.npy"
    )

    prefix_hist_path = (
        "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/tile-geoparquet/tile_geoparquet/output/global_prefix.npy"
    )

    log.info(f"Loading dataset: {parquet_path}")
    gdf = gpd.read_parquet(parquet_path)

    log.info(f"Loading histogram: {hist_path}")
    hist = np.load(hist_path)
    prefix_hist = np.load(prefix_hist_path)

    generate_tiles(gdf, hist, prefix_hist)

    log.info("🎉 DONE — tiles written to: roads_tiles_out")
