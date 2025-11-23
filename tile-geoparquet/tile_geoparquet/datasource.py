from typing import Iterable, Optional, List, Dict, Any
import logging
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

logger = logging.getLogger(__name__)


class DataSource:
    def schema(self) -> pa.Schema:
        raise NotImplementedError

    def iter_tables(self) -> Iterable[pa.Table]:
        raise NotImplementedError


# ------------------------- GeoParquet source ------------------------- #
class GeoParquetSource(DataSource):
    def __init__(self, path: str):
        self._pf = pq.ParquetFile(path)
        self._schema = self._pf.schema_arrow
        self._num_row_groups = self._pf.num_row_groups
        logger.info("GeoParquetSource opened %s with %d row groups", path, self._num_row_groups)

    def schema(self) -> pa.Schema:
        logger.debug("GeoParquet source schema metadata: %s", self._schema.metadata)
        return self._schema

    def iter_tables(self) -> Iterable[pa.Table]:
        for i in range(self._num_row_groups):
            logger.debug("Reading row group %d/%d", i, self._num_row_groups)
            yield self._pf.read_row_group(i)


# ------------------------- Helpers ------------------------- #
def is_geojson_path(path: str) -> bool:
    p = path.lower()
    return p.endswith(".geojson") or p.endswith(".json")


def _attach_geoparquet_metadata(schema: pa.Schema, crs_hint: Optional[str]) -> pa.Schema:
    """
    Return a copy of `schema` with a minimal GeoParquet 'geo' JSON block so
    downstream writers (WriterPool) can inject tile bbox.

    Includes:
      - version: 1.1.0
      - primary_column: geometry
      - columns.geometry.encoding: WKB
      - columns.geometry.crs: <crs_hint> (string hint if provided)
    """
    md = dict(schema.metadata or {})
    if b"geo" in md:
        return pa.schema(schema, metadata=md)

    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB"}}
    }
    if crs_hint:
        try:
            geo["columns"]["geometry"]["crs"] = crs_hint
        except Exception:
            pass

    md[b"geo"] = json.dumps(geo, separators=(",", ":")).encode("utf-8")
    return pa.schema(schema, metadata=md)


def _infer_crs_from_fiona_meta(meta: Dict[str, Any]) -> Optional[str]:
    """
    Try to extract a reasonable CRS hint from Fiona collection meta.
    Prefer EPSG code if present; else return None (we avoid stuffing raw WKT).
    """
    try:
        crs = meta.get("crs")
        if isinstance(crs, dict):
            # Fiona may return {'init': 'epsg:4326'} or {'EPSG': 4326} depending on GDAL
            init = crs.get("init") or crs.get("INIT")
            if isinstance(init, str) and init.lower().startswith("epsg:"):
                return init.upper().replace("EPSG:", "EPSG:")
            epsg = crs.get("epsg") or crs.get("EPSG")
            if isinstance(epsg, int):
                return f"EPSG:{epsg}"
        # Fallback: crs_wkt is too verbose to embed as the simple string hint
    except Exception:
        pass
    return None


# ------------------------- GeoJSON source (Fiona → Arrow) ------------------------- #
class GeoJSONSource(DataSource):
    """
    Streams GeoJSON as Arrow Tables using Fiona, converting geometry to WKB.

    - Reads features with Fiona (GDAL).
    - Collects features in batches of `batch_rows`.
    - Geometry dicts → shapely.shape → WKB bytes (binary Arrow column 'geometry').
    - Attaches minimal GeoParquet metadata (version, primary_column, encoding, crs hint).
    """

    def __init__(
        self,
        path: str,
        batch_rows: int = 50_000,
        src_crs: str = "EPSG:4326",
        target_crs: Optional[str] = None,
        keep_null_geoms: bool = False,
    ):
        try:
            import fiona  # noqa: F401
        except ImportError as e:
            raise ImportError("GeoJSONSource(Fiona) requires 'fiona'. Install via: pip install fiona") from e

        if target_crs:
            logger.warning("target_crs requested (%s) but Fiona batch reader does not reproject on the fly; "
                           "data will be read as-is.", target_crs)

        self.path = path
        self.batch_rows = int(batch_rows)
        self.src_crs = src_crs
        self.target_crs = target_crs  # informational only here
        self.keep_null_geoms = keep_null_geoms

        self._schema: Optional[pa.Schema] = None
        self._crs_hint: Optional[str] = None  # filled from Fiona metadata on first open

        logger.info(
            "GeoJSONSource(Fiona) opened %s (batch_rows=%d, src_crs=%s)",
            path, self.batch_rows, self.src_crs
        )

    # ---------------- schema ---------------- #
    def schema(self) -> pa.Schema:
        if self._schema is None:
            first = self._read_batch_with_fiona(0, max(1, self.batch_rows))
            if first is None or first.num_rows == 0:
                base = pa.schema([("geometry", pa.binary())])
                self._schema = _attach_geoparquet_metadata(
                    base, self._crs_hint or self.target_crs or self.src_crs
                )
            else:
                self._schema = _attach_geoparquet_metadata(
                    first.schema, self._crs_hint or self.target_crs or self.src_crs
                )
        return self._schema

    # ---------------- iterator ---------------- #
    def iter_tables(self) -> Iterable[pa.Table]:
        skip = 0
        while True:
            chunk = self._read_batch_with_fiona(skip, self.batch_rows)
            if chunk is None or chunk.num_rows == 0:
                break

            if "geometry" not in chunk.column_names:
                raise ValueError("Missing 'geometry' column from Fiona reader")

            if not self.keep_null_geoms:
                mask_null = pc.is_null(chunk["geometry"])
                if pc.any(mask_null).as_py():
                    chunk = chunk.filter(pc.invert(mask_null))

            if self._schema is None:
                # Lock schema (and metadata) from first non-empty chunk
                self._schema = _attach_geoparquet_metadata(
                    chunk.schema, self._crs_hint or self.target_crs or self.src_crs
                )
                yield chunk.combine_chunks()
            else:
                yield self._coerce_to_schema(chunk, self._schema).combine_chunks()

            skip += chunk.num_rows
            if chunk.num_rows == 0:
                break  # safety

    # ---------------- internal helpers ---------------- #
    def _read_batch_with_fiona(self, skip: int, n: int) -> Optional[pa.Table]:
        """
        Open the GeoJSON with Fiona, skip `skip` features, read up to `n` features,
        convert properties to Arrow columns and geometry to WKB, return pa.Table.

        NOTE: This is O(total_features) per call due to skipping, but is robust and simple.
        """
        import fiona
        from shapely.geometry import shape as shapely_shape
        import numpy as np

        rows_props: List[Dict[str, Any]] = []
        wkb_list: List[Any] = []

        read = 0
        seen = 0

        # Open and detect CRS hint (once)
        with fiona.open(self.path, "r") as src:
            if self._crs_hint is None:
                try:
                    self._crs_hint = _infer_crs_from_fiona_meta(src.meta or {})
                except Exception:
                    self._crs_hint = None

            for feat in src:
                if seen < skip:
                    seen += 1
                    continue

                # Collect
                props = feat.get("properties") or {}
                geom = feat.get("geometry", None)

                if geom is None:
                    wkb = None
                else:
                    try:
                        g = shapely_shape(geom)
                        wkb = g.wkb if g is not None else None
                    except Exception:
                        wkb = None

                rows_props.append(props)
                wkb_list.append(wkb)

                read += 1
                seen += 1
                if read >= n:
                    break

        if read == 0:
            logger.info(f"Fiona read returned 0 rows for slice {skip}:{skip+n}")
            return None

        # Build Arrow table:
        # 1) property columns -> Arrow via pandas-free approach (Arrow can infer types from Python lists)
        #    but properties across features may have heterogeneous keys; unify all keys seen.
        # Gather union of keys
        all_keys: List[str] = []
        seen_keys = set()
        for d in rows_props:
            for k in d.keys():
                if k not in seen_keys:
                    seen_keys.add(k)
                    all_keys.append(k)

        # Build columns as lists aligned with rows
        cols: List[pa.Array] = []
        names: List[str] = []

        for k in all_keys:
            col_py = [ row.get(k, None) for row in rows_props ]
            cols.append(pa.array(col_py))
            names.append(k)

        # 2) geometry column (binary)
        wkb_arr = pa.array(np.array(wkb_list, dtype=object), type=pa.binary())
        cols.append(wkb_arr)
        names.append("geometry")

        t = pa.table(cols, names=names)
        logger.info(f"Fiona batch {skip}:{skip+n} -> {t.num_rows} rows, {len(names)} columns (including 'geometry')")
        return t

    def _coerce_to_schema(self, t: pa.Table, schema: pa.Schema) -> pa.Table:
        out_cols = []
        for fld in schema:
            name = fld.name
            if name in t.column_names:
                col = t[name]
                if not col.type.equals(fld.type):
                    try:
                        col = col.cast(fld.type)
                    except Exception:
                        logger.warning(
                            "Type mismatch for column '%s': %s -> %s (kept original)",
                            name, col.type, fld.type
                        )
                out_cols.append(col)
            else:
                out_cols.append(pa.nulls(t.num_rows, type=fld.type))
        return pa.table(out_cols, names=[f.name for f in schema])
