from flask import Flask, Response
from flask_cors import CORS
from tiler.tiler import VectorTiler
from pathlib import Path
import argparse
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=os.environ.get("TILE_ROOT", "datasets"))
args = parser.parse_args()

DATA_ROOT = Path(args.root)

TILER_CACHE = {}

def get_tiler(dataset):
    if dataset not in TILER_CACHE:
        TILER_CACHE[dataset] = VectorTiler(str(DATA_ROOT / dataset))
    return TILER_CACHE[dataset]

@app.get("/<dataset>/<int:z>/<int:x>/<int:y>.mvt")
def serve_tile(dataset, z, x, y):
    tiler = get_tiler(dataset)
    return Response(tiler.get_tile(z, x, y), mimetype="application/vnd.mapbox-vector-tile")

if __name__ == "__main__":
    app.run(debug=True)
