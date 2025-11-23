from __future__ import annotations
from pathlib import Path
import json
import logging
import math

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
from shapely.geometry import box, GeometryCollection, mapping
from shapely import ops
from shapely import make_valid
import mapbox_vector_tile
from pyproj import Transformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mvt")

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

WORLD_MINX = -20037508.342789244
WORLD_MINY = -20037508.342789244
WORLD_MAXX =  20037508.342789244
WORLD_MAXY =  20037508.342789244

WORLD_W = WORLD_MAXX - WORLD_MINX
WORLD_H = WORLD_MAXY - WORLD_MINY

EXTENT = 4096
ZOOM_LEVELS = range(0, 2)
HIST_THRESHOLD = 0

# Transformer for bbox fix (4326 → 3857)
TF_4326_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# -------------------------------------------------------------------
# HISTOGRAM LOOKUP
# -------------------------------------------------------------------

def hist_value(hist: np.ndarray, z: int, x: int, y: int) -> float:
    H = hist.shape[0]
    baseZ = int(math.log2(H))

    if z == baseZ:
        return float(hist[y, x]) if (0 <= x < H and 0 <= y < H) else 0.0

    if z < baseZ:
        factor = 1 << (baseZ - z)
        x0 = x * factor
        y0 = y * factor
        x1 = min(x0 + factor, H)
        y1 = min(y0 + factor, H)
        return float(hist[y0:y1, x0:x1].sum())

    finer = 1 << (z - baseZ)
    px = x // finer
    py = y // finer
    if 0 <= px < H and 0 <= py < H:
        return float(hist[py, px]) / (finer * finer)
    return 0.0


# -------------------------------------------------------------------
# TILE HELPERS (WEB MERCATOR)
# -------------------------------------------------------------------

def mercator_tile_bounds(z: int, x: int, y: int):
    n = 2 ** z
    tile_w = WORLD_W / n

    minx = WORLD_MINX + x * tile_w
    maxx = minx + tile_w

    maxy = WORLD_MAXY - y * tile_w
    miny = maxy - tile_w

    return (minx, miny, maxx, maxy)


def scale_to_tile(xx, yy, tile_bounds):
    minx, miny, maxx, maxy = tile_bounds
    xs = (xx - minx) / (maxx - minx) * EXTENT
    ys = (yy - miny) / (maxy - miny) * EXTENT
    return xs, ys


def explode_geometry(g):
    if g.is_empty:
        return []
    if not isinstance(g, GeometryCollection):
        return [g]
    out = []
    for part in g.geoms:
        out.extend(explode_geometry(part))
    return out


# -------------------------------------------------------------------
# BBOX REPROJECTION FIX
# -------------------------------------------------------------------

def bbox_4326_to_3857(bbox):
    minx, miny, maxx, maxy = bbox
    x1, y1 = TF_4326_3857.transform(minx, miny)
    x2, y2 = TF_4326_3857.transform(maxx, maxy)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


# -------------------------------------------------------------------
# READ GEO-PARQUET BBOX METADATA
# -------------------------------------------------------------------

def load_partition_metadata(dir_path: str):
    partitions = []

    for pf in Path(dir_path).rglob("*.parquet"):
        pqfile = pq.ParquetFile(pf)
        schema = pqfile.schema_arrow
        meta = schema.metadata

        if meta is None or b"geo" not in meta:
            # logger.warning(f"{pf}: no 'geo' metadata, skipping")
            continue

        try:
            geo_json = json.loads(meta[b"geo"].decode("utf8"))
        except Exception as e:
            logger.warning(f"{pf}: failed to parse 'geo' metadata: {e}")
            continue

        geom_cols = geo_json.get("columns", {})
        if not geom_cols:
            logger.warning(f"{pf}: 'geo' metadata has no columns, skipping")
            continue

        geom_col = next(iter(geom_cols.keys()))
        geom_info = geom_cols.get(geom_col, {})
        geom_bbox = geom_info.get("bbox")
        if geom_bbox is None:
            logger.warning(f"{pf}: geometry column {geom_col} has no bbox, skipping")
            continue

        bbox_3857 = bbox_4326_to_3857(geom_bbox)

        partitions.append({
            "path": pf,
            "bbox_4326": geom_bbox,
            "bbox_3857": bbox_3857,
            "geom_col": geom_col
        })

    logger.info(f"Loaded {len(partitions)} partitions.")
    return partitions


def debug_print_partition_and_tile_bboxes(partitions, max_zoom=4):
    logger.info("\n=== DEBUG: PRINTING ALL PARTITION BBOXES (EPSG:3857) ===")
    for p in partitions:
        logger.info(f"PARTITION | {p['path'].name} | bbox_3857={p['bbox_3857']}")

    logger.info("\n=== DEBUG: TILE BBOXES AND INTERSECTION CHECKS ===")
    for z in range(0, max_zoom+1):
        logger.info(f"\n--- ZOOM {z} ---")
        n = 2**z
        for x in range(n):
            for y in range(n):
                tile_bounds = mercator_tile_bounds(z, x, y)
                tile_poly = box(*tile_bounds)
                logger.info(f"\nTILE z={z} x={x} y={y} bbox3857={tile_bounds}")

                # intersection list
                intersects = []
                for p in partitions:
                    part_poly = box(*p["bbox_3857"])
                    if part_poly.intersects(tile_poly):
                        intersects.append(p["path"].name)

                if intersects:
                    logger.info(f"  → INTERSECTS {len(intersects)} partitions: {intersects}")
                else:
                    logger.info("  → NO INTERSECTION")

# -------------------------------------------------------------------
# MVT GENERATION
# -------------------------------------------------------------------

def generate_tiles(partitions_dir: str, hist_path: str, out_dir="asia_output_mvt"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load histogram
    hist = np.load(hist_path)
    H = hist.shape[0]
    baseZ = int(math.log2(H))
    logger.info(f"Histogram OK ({H}×{H}), base zoom = {baseZ}")
    logger.info(f"Histogram non-zero cells: {np.count_nonzero(hist)}")


    # Load partitions
    partitions = load_partition_metadata(partitions_dir)

    # DEBUG: print all bounding boxes and intersections
    debug_print_partition_and_tile_bboxes(partitions, max_zoom=3)


    total_tiles_written = 0

    for z in ZOOM_LEVELS:
        logger.info(f"\n=== ZOOM {z} ===")

        n = 2 ** z
        tolerance = 1000 / (2 ** z)

        tiles_kept_by_hist = 0
        tiles_with_partitions = 0
        tiles_with_features = 0
        tiles_written = 0

        for x in range(n):
            for y in range(n):

                # 1) histogram gate
                hval = hist_value(hist, z, x, y)
                print(f"Tile z={z} x={x} y={y} hist={hval}")
                
                if hval < HIST_THRESHOLD:
                    # logger.info(
                    #     f"[SKIP] z={z} x={x} y={y} "
                    #     f"hist={hval:.6f} < threshold={HIST_THRESHOLD}"
                    # )
                    continue

                tiles_kept_by_hist += 1
                # logger.info(f"[KEEP-HIST] z={z} x={x} y={y} hist={hval:.6f}")

                # 2) tile bounds
                tile_bounds = mercator_tile_bounds(z, x, y)
                tile_poly = box(*tile_bounds)

                # 3) partitions that intersect this tile (in 3857)
                hit = []
                for p in partitions:
                    part_poly = box(*p["bbox_3857"])
                    if part_poly.intersects(tile_poly):
                        hit.append(p)

                if not hit:
                    logger.info(
                        f"[SKIP] z={z} x={x} y={y} "
                        f"after hist={hval:.6f}: 0 intersecting partitions"
                    )
                    continue

                tiles_with_partitions += 1
                # logger.info(
                #     f"[TILE] z={z} x={x} y={y} "
                #     f"hist={hval:.6f} → {len(hit)} intersecting partitions"
                # )

                features = []

                # 4) load and clip geometries
                for p in hit:
                    # logger.info(
                    #     f"  [PART] path={p['path'].name} "
                    #     f"bbox4326={p['bbox_4326']} geom_col={p['geom_col']}"
                    # )
                    gdf = gpd.read_parquet(p["path"], columns=[p["geom_col"]])
                    # logger.info(f"    rows in parquet: {len(gdf)}, CRS: {gdf.crs}")

                    # CRS normalization
                    if gdf.crs is None:
                        # logger.info("    CRS is None, assuming EPSG:4326")
                        gdf = gdf.set_crs(4326)
                    else:
                        crs_up = str(gdf.crs).upper()
                        # if "4326" in crs_up or "CRS84" in crs_up:
                        #     logger.info("    CRS is EPSG:4326/CRS84")
                        # elif "3857" in crs_up:
                        #     logger.info("    CRS is already EPSG:3857")
                        # else:
                        #     raise ValueError(f"    CRS {gdf.crs} not recognized as 4326/3857")

                    # Reproject to 3857 if needed
                    if gdf.crs.to_epsg() != 3857:
                        gdf = gdf.to_crs(3857)
                        # logger.info(f"    CRS after to_crs: {gdf.crs}")

                    # Clean geometries
                    gdf = gdf[gdf[p["geom_col"]].notnull()]
                    gdf = gdf.set_geometry(p["geom_col"])
                    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
                    # logger.info(f"    after dropping null/empty: {len(gdf)} geometries")

                    if gdf.empty:
                        continue

                    gdf["geometry"] = gdf["geometry"].apply(make_valid)

                    # Clip FIRST
                    clipped = gdf.geometry.intersection(tile_poly)
                    clipped = clipped[~clipped.is_empty]
                    # logger.info(f"    after clipping to tile: {len(clipped)} non-empty geometries")

                    if clipped.empty:
                        continue

                    # Simplify AFTER clip
                    clipped = clipped.apply(
                        lambda g: g.simplify(tolerance, preserve_topology=True)
                    )
                    clipped = clipped[~clipped.is_empty]
                    # logger.info(
                    #     f"    after simplify(tol={tolerance}): {len(clipped)} non-empty geometries"
                    # )

                    if clipped.empty:
                        continue

                    for geom in clipped:
                        for part in explode_geometry(geom):

                            def to_tile(xx, yy, zz=None):
                                return scale_to_tile(xx, yy, tile_bounds)

                            geom_scaled = ops.transform(to_tile, part)

                            features.append({
                                "geometry": mapping(geom_scaled),
                                "properties": {},
                            })

                if not features:
                    logger.info(
                        f"[SKIP] z={z} x={x} y={y} "
                        f"had {len(hit)} partitions but produced 0 features"
                    )
                    continue

                tiles_with_features += 1
                logger.info(
                    f"[WRITE] z={z} x={x} y={y} features={len(features)}"
                )

                # 5) build and write tile
                layer = {
                    "name": "layer0",
                    "features": features,
                    "extent": EXTENT,
                }
                tile_data = mapbox_vector_tile.encode([layer])

                out_path = out_dir / f"{z}/{x}"
                out_path.mkdir(parents=True, exist_ok=True)
                with open(out_path / f"{y}.mvt", "wb") as f:
                    f.write(tile_data)

                tiles_written += 1
                total_tiles_written += 1

        logger.info(
            f"✔ Zoom {z}: "
            f"kept_by_hist={tiles_kept_by_hist}, "
            f"with_partitions={tiles_with_partitions}, "
            f"with_features={tiles_with_features}, "
            f"written={tiles_written}"
        )

    logger.info(f"TOTAL tiles written across all zooms = {total_tiles_written}")


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    partitions_dir = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/tile-geoparquet/tile_geoparquet/output_asia/"
    hist_path = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/tile-geoparquet/tile_geoparquet/asia_output/global.npy"
    generate_tiles(partitions_dir, hist_path)
