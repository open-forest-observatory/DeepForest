import argparse
import os
import shutil
from pathlib import Path

from deepforest.preprocess import split_raster
from deepforest.david import align_to_reference
from deepforest.utilities import shapefile_to_annotations
from deepforest import main as deepforest_main
from deepforest.utilities import boxes_to_shapefile

def create_base_model():
    model = deepforest_main.deepforest()
    model.use_release()
    return model

def predict_and_write(model, input_file, output_file):
    # Predict in aligned images
    prediction = model.predict_tile(
        input_file, return_plot=False, patch_size=400, patch_overlap=0.25
    )
    shapefile = boxes_to_shapefile(
        prediction, str(Path(input_file).parent), flip_y_axis=True  # Needed for QGis
    )
    shapefile.to_file(output_file)

def train_model(annotations_file, n_epochs, model_savefile):
    model = create_base_model()

    model.config["save-snapshot"] = False
    model.config["train"]["epochs"] = n_epochs
    model.config["train"]["log_every_n_steps"] = 1 
    model.config["train"]["csv_file"] = annotations_file
    model.config["train"]["root_dir"] = os.path.dirname(annotations_file)

    model.create_trainer()
    model.trainer.fit(model)
    model.save_model(model_savefile)

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
    # Create sub directories
    inputs_folder = Path(workdir, "inputs")
    preds_folder = Path(workdir, "preds")
    models_folder = Path(workdir, "models")
    anns_folder = Path(workdir, "anns")
    ortho_crops_folder = Path(workdir, "crops", "ortho")
    RS_crops_folder = Path(workdir, "crops", "RS")
    # Create sub directoies
    [os.makedirs(folder, exist_ok=True) for folder in (inputs_folder, preds_folder, models_folder, anns_folder,ortho_crops_folder, RS_crops_folder)]

    # Resampled files and copy annotation
    resampled_RS = Path(inputs_folder, Path(remote_sensing_file).name)
    resampled_ortho = Path(inputs_folder, Path(ortho_file).name)
    shutil.copyfile(
        training_annotations, Path(inputs_folder, Path(training_annotations).name)
    )

    # Crop both datasets to the extent of the ortho and resample
    align_to_reference(
        reference_file=ortho_file,
        working_file=remote_sensing_file,
        output_file=resampled_RS,
        output_padding=0,
        output_GSD=GSD,
    )
    # This just does resampling
    align_to_reference(
        reference_file=ortho_file,
        working_file=ortho_file,
        output_file=resampled_ortho,
        output_padding=0,
        output_GSD=GSD,
    )

    # Create the model
    base_model = create_base_model()

    # Generate predictions with the base model
    predict_and_write(model=base_model, input_file=resampled_ortho, output_file=Path(preds_folder, "ortho_base_preds.geojson"))
    predict_and_write(model=base_model, input_file=resampled_RS,    output_file=Path(preds_folder, "RS_base_preds.geojson"))


    # Generated training chips
    ortho_annotations = shapefile_to_annotations(training_annotations, resampled_ortho, savedir=workdir)
    ortho_annotations_file = Path(anns_folder, "ortho_anns.csv")
    ortho_annotations.to_csv(ortho_annotations_file)
    cropped_ortho_annotations = split_raster(
        annotations_file=ortho_annotations_file,
        path_to_raster=resampled_ortho,
        base_dir=ortho_crops_folder,
    )[1]

    RS_annotations = shapefile_to_annotations(training_annotations, resampled_RS, savedir=workdir)
    RS_annotations_file =Path(anns_folder, "RS_anns.csv")
    RS_annotations.to_csv(RS_annotations_file)
    cropped_RS_annotations = split_raster(
        annotations_file=RS_annotations_file,
        path_to_raster=resampled_RS,
        base_dir=RS_crops_folder,
    )[1]

    # Finetune new models
    #train_model()

if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)
