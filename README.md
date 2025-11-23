# Tile Geoparquet Pipeline

This project creates tiled GeoParquet datasets and vector tiles from any input GeoParquet or GeoJSON file.
It also includes a small development server for previewing the generated tiles.

## Requirements

Install dependencies:

pip install -r requirements.txt

## Directory structure

The pipeline expects the following layout:

project/
    Makefile
    tile_geoparquet/
    mvt/
    server/
    datasets/          output GeoParquet tiles
    mvt_out/           output MVT tiles

You can place your input dataset anywhere, as long as you pass the path to make tiles.

## Tiling a dataset

Run the tiler:

make tiles INPUT=path/to/your/data.parquet

Example:

make tiles INPUT=../extras/original_datasets/highways/roads.parquet

The Makefile will:

1. Extract the dataset name from the input file, for example roads.parquet becomes roads.
2. Run the tiling pipeline and generate GeoParquet tiles inside:

datasets/roads/

3. Automatically generate histogram data used for visualization and analysis.
4. Write logs to:

logs_roads.txt

## Generating MVT tiles

After GeoParquet tiles are created, run:

make mvt INPUT=path/to/your/data.parquet

This reads the dataset from:

datasets/<dataset_name>

and writes vector tiles to:

mvt_out/<dataset_name>

Example:

make mvt INPUT=../extras/original_datasets/highways/roads.parquet

## Running the local tile server

The server reads all datasets inside the datasets directory.
You do not need to pass INPUT:

make server

This will:

1. Start the Flask server with:

python3 server/server.py --root datasets

2. Open the viewer in your default browser:

server/view_mvt.html

Example tile URLs:

http://localhost:5000/roads/0/0/0.mvt
http://localhost:5000/roads/5/12/18.mvt

Any dataset that exists in datasets is automatically served.

## Cleaning generated data

To remove all generated tiles and logs:

make clean

This deletes:

datasets/*
mvt_out/*
logs_*.txt

## Summary of commands

make tiles INPUT=../path/to/data.parquet
make mvt INPUT=../path/to/data.parquet
make all INPUT=../path/to/data.parquet
make server
make clean

This provides a workflow for converting any GeoJSON or GeoParquet input into tiled GeoParquet, generating histograms, producing vector tiles, and visualizing the data in a browser.
