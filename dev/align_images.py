import argparse
from deepforest.utilities import resample_to_target_gsd, crop_to_window
import rasterio as rio
from rasterio.warp import transform_bounds
from shapely.geometry import box

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-file", default="/ofo-share/repos-david/GaiaColabData/data/level_03/tree_detections/experiments/naip_ortho_registration_retraining/data/m_4407237_nw_18_060_20210920.tif")
    parser.add_argument("--output-file", default="/ofo-share/repos-david/GaiaColabData/data/level_03/tree_detections/experiments/naip_ortho_registration_retraining/data/m_4407237_nw_18_060_20210920_cropped_0.1GSD.tif")
    parser.add_argument("--reference-file", default="/ofo-share/repos-david/GaiaColabData/data/level_03/tree_detections/experiments/naip_ortho_registration_retraining/data/stow_anew_2023_06_15_collect_000_manual_000_ortho_mesh.tif")
    parser.add_argument("--output-shift", default=(-1,-3), type=float, nargs=2)
    parser.add_argument("--output-padding", default=1, type=float, help="Padding in input CRS units for crop")
    parser.add_argument("--output-GSD", default=0.1, type=float)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--resample", action="store_true")
    args = parser.parse_args()
    return args

def main(reference_file, working_file, output_file, output_padding, output_shift, output_GSD, crop, resample):
    if crop:
        # Find the points in the working image image that 
        # align with the reference bounds
        with rio.open(working_file) as working:
            with rio.open(reference_file) as ref:
                crop_bounds = transform_bounds(
                    ref.crs,
                    working.crs,
                    *ref.bounds,
                )
                min_x_geospatial, min_y_geospatial, max_x_geospatial, max_y_geospatial = crop_bounds
        crop_to_window(input_file=working_file,
                       output_file=output_file,
                       min_x_geospatial=min_x_geospatial,
                       min_y_geospatial=min_y_geospatial,
                       max_x_geospatial=max_x_geospatial,
                       max_y_geospatial=max_y_geospatial,
                       padding_geospatial=output_padding,
                       offset_geospatial_xy=output_shift)
    if resample:
        resample_to_target_gsd(working_file, output_file, target_gsd=output_GSD)
        return

if __name__ == "__main__" :
    args = parse_args()
    main(**args.__dict__)

