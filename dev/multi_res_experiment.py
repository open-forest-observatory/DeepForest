import argparse
import os
import shutil
from pathlib import Path
import math
import geopandas as gpd
import rasterio as rio
import numpy as np

from deepforest import evaluate
from deepforest.preprocess import split_raster
from deepforest.david import align_to_reference
from deepforest.utilities import shapefile_to_annotations, crop_to_window
from deepforest.visualize import plot_prediction_dataframe
from deepforest import main as deepforest_main
from deepforest.utilities import boxes_to_shapefile


def create_base_model():
    model = deepforest_main.deepforest()
    model.use_release()
    return model


def create_reload_model(model_path):
    model = deepforest_main.deepforest.load_from_checkpoint(model_path)
    model.model.score_thresh = 0.1
    return model


def create_crops(
    input_annotations_shapefile,
    input_image_file,
    workdir,
    annotations_csv,
    crop_folder,
):
    annotations = shapefile_to_annotations(
        input_annotations_shapefile, input_image_file, savedir=workdir
    )
    annotations.to_csv(annotations_csv, index_label=False)
    df, cropped_annotations_csv = split_raster(
        annotations_file=annotations_csv,
        path_to_raster=input_image_file,
        base_dir=crop_folder,
    )
    vis_folder = str(Path(crop_folder, "vis"))
    os.makedirs(vis_folder, exist_ok=True)
    plot_prediction_dataframe(df, root_dir=crop_folder, savedir=vis_folder)
    return cropped_annotations_csv


def crop_to_annotations(image_file, geofile, output_file):
    gdf = gpd.read_file(geofile)
    gdf = gdf.dropna()
    with rio.open(image_file) as src:
        gdf = gdf.to_crs(src.crs.to_epsg())

    min_x = np.min(gdf.geometry.bounds.minx.to_numpy())
    max_x = np.max(gdf.geometry.bounds.maxx.to_numpy())
    min_y = np.min(gdf.geometry.bounds.miny.to_numpy())
    max_y = np.max(gdf.geometry.bounds.maxy.to_numpy())

    crop_to_window(
        image_file,
        output_file,
        min_x_geospatial=min_x,
        min_y_geospatial=min_y,
        max_x_geospatial=max_x,
        max_y_geospatial=max_y,
        offset_geospatial_xy=(0, 0),
    )


def predict_and_write(model, input_file, output_file):
    # Predict in aligned images
    prediction = model.predict_tile(
        input_file, return_plot=False, patch_size=400, patch_overlap=0.25
    )
    if len(prediction) < 10:
        breakpoint()
    shapefile = boxes_to_shapefile(
        prediction, str(Path(input_file).parent), flip_y_axis=True  # Needed for QGis
    )
    shapefile.to_file(output_file)
    return prediction


def predict_and_eval(model, input_image_file, output_preds_file, gt_file):
    preds = predict_and_write(
        model=model,
        input_file=input_image_file,
        output_file=output_preds_file,
    )
    ground_df = shapefile_to_annotations(gt_file, input_image_file)
    eval_dict = evaluate.evaluate(
        predictions=preds, ground_df=ground_df, root_dir=str(input_image_file.parent)
    )
    return eval_dict


def train_model(annotations_file, n_epochs, model_savefile):
    model = create_base_model()

    model.config["save-snapshot"] = False
    model.config["train"]["epochs"] = n_epochs
    model.config["train"]["log_every_n_steps"] = 1
    model.config["train"]["csv_file"] = annotations_file
    model.config["train"]["root_dir"] = str(os.path.dirname(annotations_file))

    model.create_trainer()
    model.trainer.fit(model)
    model.save_model(model_savefile)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-annotations",
        default="/ofo-share/repos-david/GaiaColabData/data/level_02/annotations/per_collect/stowe_anew/2022_07_14/collect_000/automated_000/tree_boxes_David_R.shp",
    )
    parser.add_argument(
        "--ortho-file",
        default="/ofo-share/repos-david/GaiaColabData/data/level_02/photogrametry/metashape/per_collect/stowe_anew/2023_06_15/collect_000/manual_000/exports/stow_anew_2023_06_15_collect_000_manual_000_ortho_mesh_georef.tif",
    )
    parser.add_argument(
        "--remote-sensing-file",
        default="/ofo-share/repos-david/GaiaColabData/data/level_00/remote_sensing/naip_crop_m_4407237_nw_18_060_20210920.tif",
    )
    parser.add_argument(
        "--workdir",
        default="/ofo-share/repos-david/GaiaColabData/data/level_03/tree_detections/deep_forest/per_collect/stowe_anew/2023_06_15/collect_000/multi_res_experiment",
    )
    parser.add_argument("--GSD", type=float, default=0.1)
    # parser.add_argument("--test-crop", nargs="4", default=(), type=float)
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
    shutil.rmtree(workdir, ignore_errors=True)
    ## Create sub directories
    inputs_folder = Path(workdir, "inputs")
    preds_folder = Path(workdir, "preds")
    models_folder = Path(workdir, "models")
    anns_folder = Path(workdir, "anns")
    ortho_crops_folder = Path(workdir, "crops", "ortho")
    RS_crops_folder = Path(workdir, "crops", "RS")
    ortho_preds_base_crops_folder = Path(workdir, "crops", "ortho_preds_base")
    ortho_preds_finetuned_crops_folder = Path(workdir, "crops", "ortho_preds_finetuned")
    # Ensure that they exist
    [
        os.makedirs(folder, exist_ok=True)
        for folder in (
            inputs_folder,
            preds_folder,
            models_folder,
            anns_folder,
            ortho_crops_folder,
            RS_crops_folder,
            ortho_preds_base_crops_folder,
            ortho_preds_finetuned_crops_folder,
        )
    ]

    ## Resampled files and copy annotation
    resampled_RS_file = Path(inputs_folder, Path(remote_sensing_file).name)
    resampled_ortho_file = Path(inputs_folder, Path(ortho_file).name)

    # TODO update them to not include the test set
    drone_RS_file = resampled_RS_file
    drone_ortho_file = resampled_ortho_file
    shutil.copyfile(
        training_annotations, Path(inputs_folder, Path(training_annotations).name)
    )

    # Crop both datasets to the extent of the ortho and resample
    align_to_reference(
        reference_file=ortho_file,
        working_file=remote_sensing_file,
        output_file=resampled_RS_file,
        output_padding=0,
        output_GSD=GSD,
    )
    # This just does resampling
    align_to_reference(
        reference_file=ortho_file,
        working_file=ortho_file,
        output_file=resampled_ortho_file,
        output_padding=0,
        output_GSD=GSD,
    )
    train_ortho_file = Path(
        resampled_ortho_file.parent,
        resampled_ortho_file.name + "_train" + resampled_ortho_file.suffix,
    )
    train_RS_file = Path(
        resampled_RS_file.parent,
        resampled_RS_file.name + "_train" + resampled_RS_file.suffix,
    )

    test_ortho_file = Path(
        resampled_ortho_file.parent,
        resampled_ortho_file.name + "_test" + resampled_ortho_file.suffix,
    )
    test_RS_file = Path(
        resampled_RS_file.parent,
        resampled_RS_file.name + "_test" + resampled_RS_file.suffix,
    )

    crop_to_annotations(resampled_ortho_file, training_annotations, train_ortho_file)
    crop_to_annotations(resampled_RS_file, training_annotations, train_RS_file)
    # TODO update this to actually be for the test
    crop_to_annotations(resampled_ortho_file, training_annotations, test_ortho_file)
    crop_to_annotations(resampled_RS_file, training_annotations, test_RS_file)

    # Create the model
    base_model = create_base_model()

    # Generate predictions with the base model
    ortho_base_preds_file = Path(preds_folder, "ortho_base_preds.geojson")
    RS_base_preds_file = Path(preds_folder, "RS_base_preds.geojson")
    base_ortho_eval_dict = predict_and_eval(
        model=base_model,
        input_image_file=test_ortho_file,
        output_preds_file=ortho_base_preds_file,
        gt_file=training_annotations,
    )

    base_RS_eval_dict = predict_and_eval(
        model=base_model,
        input_image_file=test_RS_file,
        output_preds_file=RS_base_preds_file,
        gt_file=training_annotations,
    )

    # Generated training chips
    train_ortho_annotations_file = create_crops(
        input_annotations_shapefile=training_annotations,
        input_image_file=train_ortho_file,
        workdir=workdir,
        annotations_csv=Path(anns_folder, "ortho_anns.csv"),
        crop_folder=ortho_crops_folder,
    )

    train_RS_annotations_file = create_crops(
        input_annotations_shapefile=training_annotations,
        input_image_file=train_RS_file,
        workdir=workdir,
        annotations_csv=Path(anns_folder, "RS_anns.csv"),
        crop_folder=RS_crops_folder,
    )

    # Finetune new models based on small ammount of labeled data
    finetuned_model_ortho_file = Path(models_folder, "ortho_retrained.pth")
    finetuned_model_RS_file = Path(models_folder, "RS_retrained.pth")

    train_model(
        annotations_file=train_ortho_annotations_file,
        n_epochs=n_epochs_annotations,
        model_savefile=finetuned_model_ortho_file,
    )
    train_model(
        annotations_file=train_RS_annotations_file,
        n_epochs=n_epochs_annotations,
        model_savefile=finetuned_model_RS_file,
    )

    # Generate predictions
    finetune_ortho_eval_dict = predict_and_eval(
        model=create_reload_model(finetuned_model_ortho_file),
        input_image_file=test_ortho_file,
        output_preds_file=Path(preds_folder, "ortho_finetuned_preds.geojson"),
        gt_file=training_annotations,
    )
    finetune_RS_eval_dict = predict_and_eval(
        model=create_reload_model(finetuned_model_RS_file),
        input_image_file=test_RS_file,
        output_preds_file=Path(preds_folder, "RS_finetuned_preds.geojson"),
        gt_file=training_annotations,
    )

    # Generate predictions on the whole drone region
    ortho_base_preds_whole_region_file = Path(
        preds_folder, "ortho_base_preds_whole_region.geojson"
    )
    ortho_finetuned_preds_whole_region_file = Path(
        preds_folder, "ortho_finetuned_preds_whole_region.geojson"
    )

    predict_and_write(
        model=create_base_model(),
        input_file=drone_ortho_file,
        output_file=ortho_base_preds_whole_region_file,
    )
    predict_and_write(
        model=create_reload_model(finetuned_model_ortho_file),
        input_file=drone_ortho_file,
        output_file=ortho_finetuned_preds_whole_region_file,
    )

    # Generated training chips from the drone data predictions
    cropped_ortho_preds_base_annotations_file = create_crops(
        input_annotations_shapefile=ortho_base_preds_whole_region_file,
        input_image_file=drone_ortho_file,
        workdir=workdir,
        annotations_csv=Path(anns_folder, "ortho_preds_base_anns.csv"),
        crop_folder=ortho_preds_base_crops_folder,
    )
    cropped_ortho_preds_finetuned_annotations_file = create_crops(
        input_annotations_shapefile=ortho_finetuned_preds_whole_region_file,
        input_image_file=drone_ortho_file,
        workdir=workdir,
        annotations_csv=Path(anns_folder, "ortho_preds_finetuned_anns.csv"),
        crop_folder=ortho_preds_finetuned_crops_folder,
    )
    # Train new model from RS data from predictions from the drone
    finetuned_model_ortho_preds_base_file = Path(
        models_folder, "ortho_preds_base_retrained_RS.pth"
    )
    finetuned_model_ortho_preds_finetuned_file = Path(
        models_folder, "ortho_preds_finetuned_retrained_RS.pth"
    )
    train_model(
        annotations_file=cropped_ortho_preds_base_annotations_file,
        n_epochs=int(
            math.ceil(n_epochs_annotations / 10)
        ),  # TODO update this to be scaled by the number of annotations
        model_savefile=finetuned_model_ortho_preds_base_file,
    )
    train_model(
        annotations_file=cropped_ortho_preds_finetuned_annotations_file,
        n_epochs=int(math.ceil(n_epochs_annotations / 10)),
        model_savefile=finetuned_model_ortho_preds_finetuned_file,
    )
    # Predict
    fineteuned_RS_on_ortho_base_eval_dict = predict_and_eval(
        model=create_reload_model(finetuned_model_ortho_preds_base_file),
        input_image_file=test_RS_file,
        output_preds_file=Path(
            preds_folder, "RS_finetuned_preds_from_ortho_base_preds.geojson"
        ),
        gt_file=training_annotations,
    )
    fineteuned_RS_on_ortho_finetuned_eval_dict = predict_and_eval(
        model=create_reload_model(finetuned_model_ortho_preds_finetuned_file),
        input_image_file=test_RS_file,
        output_preds_file=Path(
            preds_folder, "RS_finetuned_preds_from_ortho_finetuned_preds.geojson"
        ),
        gt_file=training_annotations,
    )
    breakpoint()
    print(
        " name & recall & precision & IoU \n"
        + f" base ortho & {base_ortho_eval_dict['box_recall']:.3f} & {base_ortho_eval_dict['box_precision']:.3f} & {base_ortho_eval_dict['box_IoU']:.3f}\\\\ \\hline \n"
        + f" finetuned ortho & {finetune_ortho_eval_dict['box_recall']:.3f} & {finetune_ortho_eval_dict['box_precision']:.3f} & {finetune_ortho_eval_dict['box_IoU']:.3f}\\\\ \\hline \n"
        + f" base RS & {base_RS_eval_dict['box_recall']:.3f} & {base_RS_eval_dict['box_precision']:.3f} & {base_RS_eval_dict['box_IoU']:.3f}\\\\ \\hline \n"
        + f" finetune RS & {finetune_RS_eval_dict['box_recall']:.3f} & {finetune_RS_eval_dict['box_precision']:.3f} & {finetune_RS_eval_dict['box_IoU']:.3f}\\\\ \\hline \n"
        + f" finetune RS on ortho base & {fineteuned_RS_on_ortho_base_eval_dict['box_recall']:.3f} & {fineteuned_RS_on_ortho_base_eval_dict['box_precision']:.3f} & {fineteuned_RS_on_ortho_base_eval_dict['box_IoU']:.3f}\\\\ \\hline \n"
        + f" finetune RS on ortho finetuned & {fineteuned_RS_on_ortho_finetuned_eval_dict['box_recall']:.3f} & {fineteuned_RS_on_ortho_finetuned_eval_dict['box_precision']:.3f} & {fineteuned_RS_on_ortho_finetuned_eval_dict['box_IoU']:.3f}\\\\ \\hline \n"
    )
    breakpoint()


if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)
