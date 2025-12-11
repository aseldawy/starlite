# streamer.py

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import make_valid
from shapely.ops import transform as shapely_transform
import shapely.wkb as swkb
from pathlib import Path
from pyproj import Transformer
import logging

logger = logging.getLogger("bucket_mvt")


class GeometryStreamer:
    """
    Streams geometries from GeoParquet using PyArrow, row group by row group,
    exactly like your GeoParquetSource pattern.
    """

    def __init__(self, parquet_dir: str):
        self.parquet_dir = Path(parquet_dir)
        self.to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def _decode_table(self, table: pa.Table):
        """
        Convert Arrow WKB geometry column to shapely geometries, reproject to EPSG 3857,
        and yield geometry objects one by one.
        """
        if "geometry" not in table.column_names:
            raise ValueError("Expected GeoParquet with a 'geometry' column")

        geom_col = table["geometry"]

        for wkb in geom_col.to_pylist():
            if wkb is None:
                continue

            try:
                geom = swkb.loads(wkb)
            except Exception as e:
                logger.warning("Invalid WKB geometry skipped: %s", e)
                continue

            geom = make_valid(geom)

            # reproject from EPSG 4326 to WebMercator 3857
            geom = shapely_transform(self.to_3857.transform, geom)

            if geom.is_empty:
                continue

            yield geom

    def iter_geometries(self):
        """
        Main generator: iterate all parquet files, stream row groups,
        decode geometries, and yield shapely objects.
        """
        parquet_files = list(self.parquet_dir.rglob("*.parquet"))

        for pf in parquet_files:
            logger.info("Streaming GeoParquet file %s", pf)

            pf_obj = pq.ParquetFile(pf)
            num_row_groups = pf_obj.num_row_groups

            for rg in range(num_row_groups):
                table = pf_obj.read_row_group(rg)
                yield from self._decode_table(table)
