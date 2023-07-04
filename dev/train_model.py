import argparse
from deepforest.utilities import shapefile_to_annotations, crop_to_window
import geopandas as gpd
from pathlib import Path
import os
from deepforest import main as deepforest_main
import shutil
import numpy as np
from deepforest.callbacks import images_callback
import matplotlib.pyplot as plt


import rasterio as rio

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapefile", default="/ofo-share/repos-david/GaiaColabData/data/level_02/annotations/per_collect/stowe_anew/2022_07_14/collect_000/automated_000/trees.shp")
    parser.add_argument("--image-file", default="/ofo-share/repos-david/GaiaColabData/data/level_02/photogrametry/metashape/per_collect/stowe_anew/2023_06_15/collect_000/manual_000/exports/stow_anew_2023_06_15_collect_000_manual_000_ortho_mesh.tif")
    parser.add_argument("--model-savefile", default="/ofo-share/repos-david/GaiaColabData/data/level_02/photogrametry/metashape/per_collect/stowe_anew/2023_06_15/collect_000/manual_000/exports/stow_anew_2023_06_15_collect_000_manual_000_ortho_mesh.tif")
    parser.add_argument("--workdir", default="data/temp_workdir/")
    parser.add_argument("--n-epochs", default=50, type=int)
    args = parser.parse_args()
    return args

# Taken from 
# https://rasterio.readthedocs.io/en/stable/topics/reproject.html


def main(shapefile, image_file, workdir, model_savefile, n_epochs=25):
    # Make the workdir if it doesn't exist
    os.makedirs(workdir, exist_ok=True)

    gdf = gpd.read_file(shapefile)
    with rio.open(image_file) as src:
        gdf = gdf.to_crs(src.crs.to_epsg())
    
    min_x = np.min(gdf.geometry.bounds.minx.to_numpy())
    max_x = np.max(gdf.geometry.bounds.maxx.to_numpy())
    min_y = np.min(gdf.geometry.bounds.miny.to_numpy()) 
    max_y = np.max(gdf.geometry.bounds.maxy.to_numpy())

    cropped_file = Path(workdir, Path(image_file).name)

    crop_to_window(image_file, cropped_file,
                    min_x_geospatial=min_x,
                    min_y_geospatial=min_y,
                    max_x_geospatial=max_x,
                    max_y_geospatial=max_y,
                    padding_geospatial=1,
                    offset_geospatial_xy = (0,0)
                    )

    anns_in_cropped_img = shapefile_to_annotations(shapefile, cropped_file, savedir=workdir)

    cropped_min_x = np.min(anns_in_cropped_img.xmin.to_numpy())
    cropped_min_y = np.min(anns_in_cropped_img.ymin.to_numpy())

    cropped_max_x = np.max(anns_in_cropped_img.xmax.to_numpy())
    cropped_max_y = np.max(anns_in_cropped_img.ymax.to_numpy())
    print(f"cropped_min_x: {cropped_min_x}, cropped_min_y: {cropped_min_y}, cropped_max_x: {cropped_max_x}, cropped_max_y: {cropped_max_y}")
    annotations_file = Path(workdir, "annotations.csv") 
    anns_in_cropped_img.to_csv(annotations_file)

    model = deepforest_main.deepforest()
    model.use_release()

    model.config["save-snapshot"] = False
    model.config["train"]["epochs"] = n_epochs
    model.config["train"]["log_every_n_steps"] = 1 
    model.config["train"]["csv_file"] = annotations_file
    model.config["train"]["root_dir"] = os.path.dirname(annotations_file)
    #model.config["train"]["fast_dev_run"] = True

    images_callback_object = images_callback(annotations_file, workdir, workdir)

    model.create_trainer()

    model.trainer.fit(model)
    model.save_model(model_savefile)

if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)