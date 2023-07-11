import argparse
from deepforest.david import align_to_reference

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-file", default="/ofo-share/repos-david/GaiaColabData/data/level_03/tree_detections/experiments/naip_ortho_registration_retraining/data/m_4407237_nw_18_060_20210920.tif")
    parser.add_argument("--reference-file", default="/ofo-share/repos-david/GaiaColabData/data/level_03/tree_detections/experiments/naip_ortho_registration_retraining/data/stow_anew_2023_06_15_collect_000_manual_000_ortho_mesh_georef.tif")
    parser.add_argument("--output-file", default="/ofo-share/repos-david/GaiaColabData/data/level_03/tree_detections/experiments/multi_resolution_training_exp")
    parser.add_argument("--output-shift", default=(0,0), type=float, nargs=2)
    parser.add_argument("--output-padding", default=1, type=float, help="Padding in input CRS units for crop")
    parser.add_argument("--output-GSD", default=0.1, type=float)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--resample", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__" :
    args = parse_args()
    align_to_reference(**args.__dict__)
