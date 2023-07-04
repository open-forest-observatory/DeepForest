import argparse

import shapely
import geopandas as gpd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-polygon-file", default="/ofo-share/repos-david/GaiaColabData/data/level_02/annotations/per_collect/stowe_anew/2022_07_14/collect_000/automated_000/trees.shp")
    parser.add_argument("--output-box-file", default="/ofo-share/repos-david/GaiaColabData/data/level_02/annotations/per_collect/stowe_anew/2022_07_14/collect_000/automated_000/trees_boxes.geojson")
    parser.add_argument("--shrink-fraction", type=float, default=0.8, help="How much to shrink from the max")
    args = parser.parse_args()
    return args 

def main(input_polygon_file, output_box_file, shrink_fraction):
    in_gdf = gpd.read_file(input_polygon_file)
    bounds = in_gdf.bounds
    width = bounds["maxx"] - bounds["minx"]
    height = bounds["maxy"] - bounds["miny"]

    inset_fraction = (1-shrink_fraction)/2
    minx_inset = bounds["minx"] + inset_fraction * width
    maxx_inset = bounds["maxx"] - inset_fraction * width
    miny_inset = bounds["miny"] + inset_fraction * height
    maxy_inset = bounds["maxy"] - inset_fraction * height

    box_coords = zip(minx_inset, miny_inset, maxx_inset, maxy_inset)
    box_geoms = [
        shapely.geometry.box(xmin, ymin, xmax, ymax)
        for xmin, ymin, xmax, ymax in box_coords
    ]

    out_gdf = gpd.GeoDataFrame(geometry=box_geoms)
    out_gdf.crs = in_gdf.crs

    out_gdf.to_file(output_box_file)
 
if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)