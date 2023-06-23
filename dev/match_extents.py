import rasterio as rio
from rasterio.warp import transform_bounds
import rasterio.warp
from rasterio.crs import CRS
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt

SMALL_FILEPATH = "/ofo-share/repos-david/GaiaColabData/data/level_02/photogrametry/metashape/per_collect/stowe_anew/2023_06_15/collect_001/manual_000/exports/stowe_anew_2023_06_15_collect_001_manual_000_mesh_ortho.tif" 
LARGE_FILEPATH = "/ofo-share/repos-david/DeepForest/data/m_4407237_nw_18_060_20210920.tif"

small_dataset = rio.open(SMALL_FILEPATH)
large_dataset = rio.open(LARGE_FILEPATH)

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

min_i, min_j = large_dataset.index(xmin, ymax)
max_i, max_j = large_dataset.index(xmax, ymin)

small_img = small_dataset.read()
small_img = np.moveaxis(small_img, 0, 2)

large_img = large_dataset.read()
large_img = np.moveaxis(large_img, 0, 2)
large_img_crop = large_img[min_i:max_i, min_j:max_j, :]
fig, ax = plt.subplots(1,2)
ax[0].imshow(large_img_crop[..., :3])
ax[1].imshow(small_img)
plt.show()


model = deepforest_main.deepforest()
model.use_release()
predicted_raster = model.predict_tile(input_file,
                                       return_plot = True,
                                       patch_size=300,
                                       patch_overlap=0.25,
                                       resize_factor=resize_factor,
                                       brighten_factor=brighten_factor)
os.makedirs(output_folder, exist_ok=True)
output_file = Path(output_folder, f"preds_resize_{resize_factor}_brighen_{brighten_factor}.png")
imwrite(output_file, predicted_raster)