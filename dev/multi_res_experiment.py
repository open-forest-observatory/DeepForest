import argparse
import os
import shutil
from pathlib import Path

from deepforest.david import align_to_reference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-annotations",
        default="/ofo-share/repos-david/GaiaColabData/data/level_02/annotations/per_collect/stowe_anew/2022_07_14/collect_000/automated_000/tree_boxes_David_R.geojson.json",
    )
    parser.add_argument(
        "--ortho-file",
        default="/ofo-share/repos-david/GaiaColabData/data/level_02/photogrametry/metashape/per_collect/stowe_anew/2023_06_15/collect_000/manual_000/exports/stow_anew_2023_06_15_collect_000_manual_000_ortho_mesh_georef.tif",
    )
    parser.add_argument(
        "--remote-sensing-file",
        default="/ofo-share/repos-david/GaiaColabData/data/level_00/remote_sensing/naip_crop_m_4407237_nw_18_060_20210920.tif",
    )
    parser.add_argument("--workdir", default="data/temp_workdir/multi_res_exp")
    parser.add_argument("--GSD", type=float, default=0.1)
    parser.add_argument("--n-epochs-annotations", default=50, type=int)
    args = parser.parse_args()
    return args


def main(
    training_annotations,
    ortho_file,
    remote_sensing_file,
    workdir,
    GSD,
    n_epochs_annotations,
):

    images_folder = Path(workdir, "images")
    os.makedirs(images_folder, exist_ok=True)
    resampled_remote_sensing = Path(images_folder, Path(remote_sensing_file).name)
    resampled_ortho = Path(images_folder, Path(ortho_file).name)
    shutil.copyfile(training_annotations, Path(images_folder, Path(training_annotations).name))
    # Align rs
    align_to_reference(
        reference_file=ortho_file,
        working_file=remote_sensing_file,
        output_file=resampled_remote_sensing,
        output_padding=0,
        output_GSD=GSD,
    )
    align_to_reference(
        reference_file=ortho_file,
        working_file=ortho_file,
        output_file=resampled_ortho,
        output_padding=0,
        output_GSD=GSD,
    )


if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)
