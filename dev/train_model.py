import argparse
from deepforest.utilities import shapefile_to_annotations
import geopandas as gpd
from pathlib import Path
import os
from deepforest import main as deepforest_main
import shutil
import numpy as np
from deepforest.callbacks import images_callback
import matplotlib.pyplot as plt

import rasterio as rio
from rasterio.windows import Window

import numpy as np
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapefile", default="/ofo-share/repos-david/GaiaColabData/data/level_02/annotations/per_collect/stowe_anew/2022_07_14/collect_000/automated_000/trees.shp")
    parser.add_argument("--image-file", default="/ofo-share/repos-david/GaiaColabData/data/level_02/photogrametry/metashape/per_collect/stowe_anew/2023_06_15/collect_000/manual_000/exports/stow_anew_2023_06_15_collect_000_manual_000_ortho_mesh.tif")
    parser.add_argument("--workdir", default="data/temp_workdir/")
    args = parser.parse_args()
    return args

# Taken from 
# https://rasterio.readthedocs.io/en/stable/topics/reproject.html

def resample_to_target_gsd(src_file, dst_file, target_gsd=0.1, shift=None):
    with rio.open(src_file) as dataset:
        profile = dataset.profile.copy()

        current_resolution = np.array(dataset.res)

        original_width = dataset.width
        original_height = dataset.height

        if current_resolution[0] < 1e-5:
            current_resolution[0] = current_resolution[0] * 111139
            lon = np.deg2rad(dataset.bounds.top)
            cos_lon = np.cos(lon)
            current_resolution[1] = current_resolution[1] * 111139 / cos_lon

        resize_factor = current_resolution / target_gsd
        scale_factor_x, scale_factor_y = resize_factor

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(np.ceil(original_height * scale_factor_y)),
                int(np.ceil(original_width * scale_factor_x))
            ),
            resampling=Resampling.bilinear
        )
        #data = data[:3]
        _, output_height, output_width = data.shape
        plt.imshow(np.transpose(data[:3], (1, 2, 0)))
        plt.show()
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (original_width / output_width ),
            (original_height / output_height )
        )
        breakpoint()
        profile.update({"height": output_height,
                        "width": output_width,
                       "transform": transform})

    with rio.open(dst_file, "w", **profile) as dataset:
        dataset.write(data)

# Taken from
# https://gis.stackexchange.com/questions/367832/using-rasterio-to-crop-image-using-pixel-coordinates-instead-of-geographic-coord
def crop_to_window(input_file, output_file,
                    min_x_geospatial, min_y_geospatial, max_x_geospatial, max_y_geospatial,
                    padding_geospatial=1, offset_geospatial=None):
    """
    Locations in the units of the CRS
    padding in the units of the CRS
    shift in the units of the CRS
    """
    with rio.open(input_file) as src:
        min_x_geospatial = min_x_geospatial - padding_geospatial 
        max_x_geospatial = max_x_geospatial + padding_geospatial 

        min_y_geospatial = min_y_geospatial - padding_geospatial 
        max_y_geospatial = max_y_geospatial + padding_geospatial 

        # Note that y values are switched because of different convention
        min_i_pixels, min_j_pixels = src.index(min_x_geospatial, max_y_geospatial)
        max_i_pixels, max_j_pixels = src.index(max_x_geospatial, min_y_geospatial)     
        # Create a Window and calculate the transform from the source dataset    
        window = Window.from_slices((min_i_pixels, max_i_pixels), (min_j_pixels, max_j_pixels))
        transform = src.window_transform(window)

        width = max_j_pixels - min_j_pixels
        height = max_i_pixels - min_i_pixels

        # Create a new cropped raster to write to
        profile = src.profile
        profile.update({
            'height': height,
            'width': width,
            'transform': transform})

        with rio.open(output_file, 'w', **profile) as dst:
            # Read the data from the window and write it to the output raster
            dst.write(src.read(window=window))
        resample_to_target_gsd(output_file, output_file )

def main(shapefile, image_file, workdir):
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
                    )
    breakpoint()

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
    model.config["train"]["epochs"] = 5
    model.config["train"]["log_every_n_steps"] = 1 
    model.config["train"]["csv_file"] = annotations_file
    model.config["train"]["root_dir"] = os.path.dirname(annotations_file)
    #model.config["train"]["fast_dev_run"] = True

    images_callback_object = images_callback(annotations_file, workdir, workdir)

    model.create_trainer()

    model.trainer.fit(model)
    model.save_model(Path(workdir, "model.pth"))

if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)