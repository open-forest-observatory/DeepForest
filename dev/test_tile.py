from deepforest import main as deepforest_main
from deepforest import get_data
import os
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
from imageio import imwrite
import numpy as np
import rasterio as rio
from rasterio.warp import transform_bounds


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--target-gsds", type=float, nargs="+", default=(0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125))
    parser.add_argument("--brighten-factors", type=float, nargs="+", default=[1.0])
    parser.add_argument("--crop-file", help="File to crop extent based on")
    parser.add_argument("--output-folder", default="vis")
    args = parser.parse_args()

    return args

def get_matching_region(small_filepath, large_filepath, return_coords=True):
    """
    Load crop of large dataset matching small one
    """
    small_dataset = rio.open(small_filepath)
    large_dataset = rio.open(large_filepath)

    # Goal: find the pixel space crop into the large image that matches the small one

    # Find bounds of the small image 
    # Convert these into the CRS of the large one
    # Convert these into the pixels of the large one
    small_bounds = small_dataset.bounds
    xmin, ymin, xmax, ymax = transform_bounds(
        small_dataset.crs,
        large_dataset.crs,
        *small_bounds
    )
    if return_coords:
        return xmin, ymin, xmax, ymax

    min_i, min_j = large_dataset.index(xmin, ymax)
    max_i, max_j = large_dataset.index(xmax, ymin)

    small_img = small_dataset.read()
    small_img = np.moveaxis(small_img, 0, 2)

    large_img = large_dataset.read()
    large_img = np.moveaxis(large_img, 0, 2)
    large_img_crop = large_img[min_i:max_i, min_j:max_j, :3]
    return large_img_crop

def main(input_file, target_gsd, brighten_factor, output_folder, crop_file=None):
    model = deepforest_main.deepforest()
    model.use_release()
    if crop_file is not None:
        geospatial_crop = get_matching_region(crop_file, input_file)
    else:
        geospatial_crop = None

    predicted_raster = model.predict_tile(input_file,
                                          return_plot = True,
                                          patch_size=300,
                                          patch_overlap=0.25,
                                          target_gsd=target_gsd,
                                          geospatial_crop=geospatial_crop,
                                          brighten_factor=brighten_factor)

    os.makedirs(output_folder, exist_ok=True)
    output_file = Path(output_folder, f"preds_gsd_{target_gsd}_brighen_{brighten_factor}.png")
    imwrite(output_file, predicted_raster)

if __name__ == "__main__":
    args = parse_args() 
    for target_gsd in args.target_gsds:#args.resize_factors:
        for brighten_factor in args.brighten_factors: #args.brighten_factors:
            main(args.input_file, target_gsd=target_gsd, brighten_factor=brighten_factor, output_folder=args.output_folder, crop_file=args.crop_file)
