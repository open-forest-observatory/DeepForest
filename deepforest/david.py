from deepforest.utilities import resample_to_target_gsd, crop_to_window
import rasterio as rio
from rasterio.warp import transform_bounds
from shapely.geometry import box


def align_to_reference(
    reference_file,
    working_file,
    output_file,
    output_padding,
    output_GSD,
    output_shift=(0,0),
    crop=True,
    resample=True,
):
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
                (
                    min_x_geospatial,
                    min_y_geospatial,
                    max_x_geospatial,
                    max_y_geospatial,
                ) = crop_bounds
        crop_to_window(
            input_file=working_file,
            output_file=output_file,
            min_x_geospatial=min_x_geospatial,
            min_y_geospatial=min_y_geospatial,
            max_x_geospatial=max_x_geospatial,
            max_y_geospatial=max_y_geospatial,
            padding_geospatial=output_padding,
            offset_geospatial_xy=output_shift,
        )
    if resample:
        resample_to_target_gsd(output_file, output_file, target_gsd=output_GSD)
        return
