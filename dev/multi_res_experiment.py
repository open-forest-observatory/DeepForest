import argparse
import os
import shutil
from pathlib import Path
import math
import geopandas as gpd
import rasterio as rio
import numpy as np
from imageio import imwrite
from copy import copy
import matplotlib.pyplot as plt

from deepforest import evaluate
from deepforest.preprocess import split_raster
from deepforest.david import align_to_reference
from deepforest.utilities import shapefile_to_annotations, crop_to_window
from deepforest.visualize import plot_prediction_dataframe
from deepforest import main as deepforest_main
from deepforest.utilities import boxes_to_shapefile


def create_base_model():
    model = deepforest_main.deepforest()
    model.use_release(check_release=False)
    return model


def create_reload_model(model_path):
    model = deepforest_main.deepforest.load_from_checkpoint(model_path)
    model.model.score_thresh = 0.1
    return model


def print_results(
    all_metrics,
    names=(
        "Base ortho",
        "Finetuned ortho",
        "Base RS",
        "Finetune RS",
        "Finetune RS on ortho base",
        "Finetune RS on ortho finetuned",
    ),
    required_data=(
        "drone",
        "field + drone",
        "NAIP",
        "field + NAIP",
        "drone + NAIP",
        "field + drone + NAIP",
    ),
):
    all_metrics = np.stack(all_metrics, axis=2)
    mean_metrics = np.mean(all_metrics, axis=2)
    std_metrics = np.std(all_metrics, axis=2)
    res_str = "\\textbf{Required data} & \\textbf{Experiment} & \\textbf{Recall} & \\textbf{Precision} \\\\ \n"

    for i in range(mean_metrics.shape[0]):
        res_str += f"{required_data[i]} & {names[i]} & {mean_metrics[i,0]:.3f} \pm {std_metrics[i,0]:.3f} & {mean_metrics[i,1]:.3f} \pm {std_metrics[i,1]:.3f} & {mean_metrics[i,2]:.3f} \pm {std_metrics[i,2]:.3f}\\\\ \\hline \n"
    print(res_str)


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
    plot, prediction = model.predict_tile(
        input_file, return_plot=True, patch_size=400, patch_overlap=0.25
    )
    if len(prediction) < 10:
        breakpoint()
    shapefile = boxes_to_shapefile(
        prediction, str(Path(input_file).parent), flip_y_axis=True  # Needed for QGis
    )
    shapefile.to_file(output_file)
    imwrite(Path(output_file.parent, output_file.stem + ".png"), plot)

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


def preprocess_files(
    workdir,
    remote_sensing_file,
    ortho_file,
    training_annotations,
    train_box_file,
    test_box_file,
    GSD,
):
    shutil.rmtree(workdir, ignore_errors=True)
    ## Create sub directories
    folders = {
        "inputs": Path(workdir, "inputs"),
        "preds": Path(workdir, "preds"),
        "models": Path(workdir, "models"),
        "anns": Path(workdir, "anns"),
        "ortho_crops": Path(workdir, "crops", "ortho"),
        "RS_crops": Path(workdir, "crops", "RS"),
        "ortho_preds_base_crops": Path(workdir, "crops", "ortho_preds_base"),
        "ortho_preds_finetuned_crops": Path(workdir, "crops", "ortho_preds_finetuned"),
    }
    # Ensure that they exist
    [os.makedirs(folder, exist_ok=True) for folder in folders.values()]

    ## Resampled files and copy annotation
    resampled_RS_file = Path(folders["inputs"], Path(remote_sensing_file).name)
    resampled_ortho_file = Path(folders["inputs"], Path(ortho_file).name)

    shutil.copyfile(
        training_annotations, Path(folders["inputs"], Path(training_annotations).name)
    )
    training_df = gpd.read_file(training_annotations)
    test_box_df = gpd.read_file(test_box_file)
    xmin, ymin, xmax, ymax = test_box_df.total_bounds
    cropped_training_df = training_df.cx[xmin:xmax, ymin:ymax]
    test_annotations_file = Path(
        folders["inputs"], Path(training_annotations).stem + "_test.shp"
    )
    cropped_training_df.to_file(test_annotations_file)

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
    ortho_files = {
        name: Path(
            folders["inputs"],
            resampled_ortho_file.stem + "_" + name + resampled_ortho_file.suffix,
        )
        for name in (
            "train_drone",
            "train_annotations",
            "test_drone",
            "test_annotations",
        )
    }
    # Concatnate the two
    RS_files = {
        name: Path(
            folders["inputs"],
            resampled_RS_file.stem + "_" + name + resampled_RS_file.suffix,
        )
        for name in (
            "train_drone",
            "train_annotations",
            "test_drone",
            "test_annotations",
        )
    }

    crop_to_annotations(
        resampled_ortho_file, train_box_file, ortho_files["train_drone"]
    )
    crop_to_annotations(resampled_ortho_file, test_box_file, ortho_files["test_drone"])
    crop_to_annotations(
        ortho_files["train_drone"],
        training_annotations,
        ortho_files["train_annotations"],
    )
    crop_to_annotations(
        ortho_files["test_drone"],
        training_annotations,
        ortho_files["test_annotations"],
    )

    crop_to_annotations(resampled_RS_file, train_box_file, RS_files["train_drone"])
    crop_to_annotations(resampled_RS_file, test_box_file, RS_files["test_drone"])
    crop_to_annotations(
        RS_files["train_drone"],
        training_annotations,
        RS_files["train_annotations"],
    )
    crop_to_annotations(
        RS_files["test_drone"],
        training_annotations,
        RS_files["test_annotations"],
    )
    return folders, ortho_files, RS_files, test_annotations_file


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
    parser.add_argument(
        "--train-box-file",
        default="/ofo-share/repos-david/GaiaColabData/data/level_02/annotations/per_collect/stowe_anew/2022_07_14/collect_000/automated_000/train_box.shp",
        help="Geofile. The annotations and drone data within this box will be used for training",
    )
    parser.add_argument(
        "--test-box-file",
        default="/ofo-share/repos-david/GaiaColabData/data/level_02/annotations/per_collect/stowe_anew/2022_07_14/collect_000/automated_000/test_box.shp",
        help="Geofile. The data this box will be used for testing, and evaluated against the annotations",
    )
    parser.add_argument("--GSD", type=float, default=0.1)
    # parser.add_argument("--test-crop", nargs="4", default=(), type=float)
    parser.add_argument("--n-epochs-annotations", default=50, type=int)
    parser.add_argument(
        "--just-vis",
        action="store_true",
        help="Just create visualization form saved files without running experiments",
    )
    args = parser.parse_args()
    return args


def main(
    training_annotations,
    ortho_file,
    remote_sensing_file,
    workdir,
    GSD,
    n_epochs_annotations,
    train_box_file,
    test_box_file,
    just_vis,
):
    (
        folders,
        ortho_files,
        RS_files,
        test_annotations,
    ) = preprocess_files(
        workdir=workdir,
        remote_sensing_file=remote_sensing_file,
        ortho_file=ortho_file,
        training_annotations=training_annotations,
        test_box_file=test_box_file,
        train_box_file=train_box_file,
        GSD=GSD,
    )
    # Create the model
    base_model = create_base_model()

    # Generate predictions with the base model
    ortho_base_preds_file = Path(folders["preds"], "ortho_base_preds.geojson")
    RS_base_preds_file = Path(folders["preds"], "RS_base_preds.geojson")
    base_ortho_eval_dict = predict_and_eval(
        model=base_model,
        input_image_file=ortho_files["test_annotations"],
        output_preds_file=ortho_base_preds_file,
        gt_file=test_annotations,
    )

    base_RS_eval_dict = predict_and_eval(
        model=base_model,
        input_image_file=RS_files["test_annotations"],
        output_preds_file=RS_base_preds_file,
        gt_file=test_annotations,
    )

    # Generated training chips
    train_ortho_annotations_file = create_crops(
        input_annotations_shapefile=training_annotations,
        input_image_file=ortho_files["train_annotations"],
        workdir=workdir,
        annotations_csv=Path(folders["anns"], "ortho_anns.csv"),
        crop_folder=folders["ortho_crops"],
    )

    train_RS_annotations_file = create_crops(
        input_annotations_shapefile=training_annotations,
        input_image_file=RS_files["train_annotations"],
        workdir=workdir,
        annotations_csv=Path(folders["anns"], "RS_anns.csv"),
        crop_folder=folders["RS_crops"],
    )

    # Finetune new models based on small ammount of labeled data
    finetuned_model_ortho_file = Path(folders["models"], "ortho_retrained.pth")
    finetuned_model_RS_file = Path(folders["models"], "RS_retrained.pth")

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
        input_image_file=ortho_files["test_annotations"],
        output_preds_file=Path(folders["preds"], "ortho_finetuned_preds.geojson"),
        gt_file=test_annotations,
    )
    finetune_RS_eval_dict = predict_and_eval(
        model=create_reload_model(finetuned_model_RS_file),
        input_image_file=RS_files["test_annotations"],
        output_preds_file=Path(folders["preds"], "RS_finetuned_preds.geojson"),
        gt_file=test_annotations,
    )

    # Generate predictions on the whole drone region
    ortho_base_preds_whole_region_file = Path(
        folders["preds"], "ortho_base_preds_whole_region.geojson"
    )
    ortho_finetuned_preds_whole_region_file = Path(
        folders["preds"], "ortho_finetuned_preds_whole_region.geojson"
    )

    predict_and_write(
        model=create_base_model(),
        input_file=ortho_files["train_drone"],
        output_file=ortho_base_preds_whole_region_file,
    )
    predict_and_write(
        model=create_reload_model(finetuned_model_ortho_file),
        input_file=ortho_files["train_drone"],
        output_file=ortho_finetuned_preds_whole_region_file,
    )

    # Generated training chips from the drone data predictions
    cropped_ortho_preds_base_annotations_file = create_crops(
        input_annotations_shapefile=ortho_base_preds_whole_region_file,
        input_image_file=RS_files["train_drone"],
        workdir=workdir,
        annotations_csv=Path(folders["anns"], "ortho_preds_base_anns.csv"),
        crop_folder=folders["ortho_preds_base_crops"],
    )
    cropped_ortho_preds_finetuned_annotations_file = create_crops(
        input_annotations_shapefile=ortho_finetuned_preds_whole_region_file,
        input_image_file=RS_files["train_drone"],
        workdir=workdir,
        annotations_csv=Path(folders["anns"], "ortho_preds_finetuned_anns.csv"),
        crop_folder=folders["ortho_preds_finetuned_crops"],
    )
    # Train new model from RS data from predictions from the drone
    finetuned_model_ortho_preds_base_file = Path(
        folders["models"], "ortho_preds_base_retrained_RS.pth"
    )
    finetuned_model_ortho_preds_finetuned_file = Path(
        folders["models"], "ortho_preds_finetuned_retrained_RS.pth"
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
        input_image_file=RS_files["test_annotations"],
        output_preds_file=Path(
            folders["preds"], "RS_finetuned_preds_from_ortho_base_preds.geojson"
        ),
        gt_file=test_annotations,
    )
    fineteuned_RS_on_ortho_finetuned_eval_dict = predict_and_eval(
        model=create_reload_model(finetuned_model_ortho_preds_finetuned_file),
        input_image_file=RS_files["test_annotations"],
        output_preds_file=Path(
            folders["preds"], "RS_finetuned_preds_from_ortho_finetuned_preds.geojson"
        ),
        gt_file=test_annotations,
    )

    metrics = np.array(
        [
            [
                base_ortho_eval_dict["box_recall"],
                base_ortho_eval_dict["box_precision"],
                base_ortho_eval_dict["box_IoU"],
            ],
            [
                finetune_ortho_eval_dict["box_recall"],
                finetune_ortho_eval_dict["box_precision"],
                finetune_ortho_eval_dict["box_IoU"],
            ],
            [
                base_RS_eval_dict["box_recall"],
                base_RS_eval_dict["box_precision"],
                base_RS_eval_dict["box_IoU"],
            ],
            [
                finetune_RS_eval_dict["box_recall"],
                finetune_RS_eval_dict["box_precision"],
                finetune_RS_eval_dict["box_IoU"],
            ],
            [
                fineteuned_RS_on_ortho_base_eval_dict["box_recall"],
                fineteuned_RS_on_ortho_base_eval_dict["box_precision"],
                fineteuned_RS_on_ortho_base_eval_dict["box_IoU"],
            ],
            [
                fineteuned_RS_on_ortho_finetuned_eval_dict["box_recall"],
                fineteuned_RS_on_ortho_finetuned_eval_dict["box_precision"],
                fineteuned_RS_on_ortho_finetuned_eval_dict["box_IoU"],
            ],
        ]
    )
    print(metrics)
    return metrics


def vis_metrics(input_file="data/metrics.npz", output_file="vis/metrics.png", shift=2, title=None, xlabel_tag="Inference resolution"):
    plt.style.use("seaborn-v0_8-paper")
    data = dict(np.load(input_file))
    means = []
    stds = []
    keys = data.keys()
    labels = [x[:5] for x in keys]
    ticks = [float(x) for x in keys]

    for k, v in data.items():
        print(v.shape)
        mean = np.mean(v, axis=2)
        std = np.std(v, axis=2)
        means.append(mean)
        stds.append(std)

    means = np.stack(means, axis=2)
    stds = np.stack(stds, axis=2)

    plt.xticks(size=14)
    plt.xscale("log")
    # plt.xticks(ticks=[], labels=[])
    # plt.xticks(ticks=[], labels=[], minor=True)
    plt.plot(ticks, means[0+shift, 0, :], c="tab:blue", linestyle="--", label="Pretrained recall")
    plt.plot(ticks, means[0+shift, 1, :], c="tab:orange", linestyle="--", label="Pretrained precision")
    plt.plot(ticks, means[0+shift, 2, :], c="tab:green", linestyle="--", label="Pretrained mIoU")
    plt.plot(ticks, means[1+shift, 0, :], c="tab:blue", linestyle="-", label="Finetuned recall")
    plt.plot(ticks, means[1+shift, 1, :], c="tab:orange", linestyle="-", label="Finetuned precision")
    plt.plot(ticks, means[1+shift, 2, :], c="tab:green", linestyle="-", label="Finetuned mIoU")

    plt.xlabel(f"{xlabel_tag} (meters/px, log scale)", size=14)
    plt.ylabel("Test set metrics", size=14)
    plt.yticks(size=14)
    if title is not None:
        plt.title(title, size=14)

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    plt.style.use("default")

def run_inference(args, n_repeats=3, GSD=None):
    first_dict = copy(args.__dict__)
    second_dict = copy(args.__dict__)
    # Swap train and test
    second_dict["test_box_file"] = first_dict["train_box_file"]
    second_dict["train_box_file"] = first_dict["test_box_file"]

    if GSD is not None:
        first_dict["GSD"] = GSD
        second_dict["GSD"] = GSD

    results = []
    for i in range(n_repeats):
        results.extend([main(**first_dict), main(**second_dict)])
    results = np.stack(results, axis=2)
    return results

def run_inference_res_sweep(args):

    first_dict = copy(args.__dict__)
    second_dict = copy(args.__dict__)
    second_dict["test_box_file"] = first_dict["train_box_file"]
    second_dict["train_box_file"] = first_dict["test_box_file"]

    all_metrics = {}
    for GSD in np.geomspace(0.01, 1, num=20):
        first_dict["GSD"] = GSD
        second_dict["GSD"] = GSD
        results = []
        for i in range(3):
            results.extend([main(**first_dict), main(**second_dict)])
        all_metrics[str(GSD)] = np.stack(results, axis=2)
        print(all_metrics)
        np.savez("data/metrics.npz", **all_metrics)
        vis_metrics(input_file="data/metrics.npz", output_file="vis/metrics.png")
        vis_metrics(input_file="data/metrics.npz", output_file="vis/metrics.pdf")

def simulated_RS_experiments(args, n_repeats=3):
    downsampled_folder = Path(Path(args.ortho_file).parent, "downsampled") 
    os.makedirs(downsampled_folder, exist_ok=True)
    all_metrics = {}
    for simulated_RS_res in np.geomspace(0.1,1, 20):
        simulated_RS_file = Path(downsampled_folder, f"downsampled_{simulated_RS_res:03f}_" + Path(args.ortho_file).name)
        # Choose a place to save downsampled version
        align_to_reference(
            reference_file=args.ortho_file,
            working_file=args.ortho_file,
            output_file=simulated_RS_file,
            output_padding=0,
            output_GSD=simulated_RS_res,
        )

        # Create dict as copy
        shared_dict = copy(args.__dict__)
        # Update input file
        shared_dict["remote_sensing_file"] = simulated_RS_file
        shared_dict["workdir"] = str(Path(args.workdir, f"downsampled_{simulated_RS_res:03f}"))
        first_dict = copy(shared_dict)
        second_dict = copy(shared_dict)
        second_dict["test_box_file"] = first_dict["train_box_file"]
        second_dict["train_box_file"] = first_dict["test_box_file"]
        results = []
        for i in range(n_repeats):
            results.extend([main(**first_dict), main(**second_dict)])
        all_metrics[str(simulated_RS_res)] = np.stack(results, axis=2)
        npz_file = "data/simulated_RS_metrics.npz"
        np.savez(npz_file, **all_metrics)
        vis_metrics(input_file=npz_file, output_file="vis/simulated_RS_metrics.png", shift=2)
        vis_metrics(input_file=npz_file, output_file="vis/simulated_RS_metrics.pdf", shift=2)
        vis_metrics(input_file=npz_file, output_file="vis/simulated_RS_metrics_trained_on_drone_preds.png", shift=4)
        vis_metrics(input_file=npz_file, output_file="vis/simulated_RS_metrics_trained_on_drone_preds.pdf", shift=4)


if __name__ == "__main__":
    args = parse_args()
    if args.just_vis:
        vis_metrics(input_file="data/metrics.npz", output_file="vis/drone_metrics.png", shift=0,title="Orthomosaic performance vs. inference resolution", xlabel_tag="Inference resolution")
        vis_metrics(input_file="data/metrics.npz", output_file="vis/drone_metrics.pdf", shift=0,title="Orthomosaic performance vs. inference resolution", xlabel_tag="Inference resolution")
        vis_metrics(input_file="data/metrics.npz", output_file="vis/NAIP_metrics.png", shift=2,title="NAIP performance vs. inference resolution", xlabel_tag="Inference resolution")
        vis_metrics(input_file="data/metrics.npz", output_file="vis/NAIP_metrics.pdf", shift=2,title="NAIP performance vs. inference resolution", xlabel_tag="Inferene resolution")
        vis_metrics(input_file="data/simulated_RS_metrics.npz", output_file="vis/simulated_RS_metrics.pdf", shift=2, title="Performance vs. simulated resolution", xlabel_tag="Data resolution")
        exit()
    run_inference(args)
    run_inference_res_sweep(args)
    simulated_RS_experiments(args)
