from deepforest import main as deepforest_main
from deepforest import get_data
import os
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
from imageio import imwrite
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--resize-factors", type=float, nargs="+", default=[1.0])
    parser.add_argument("--brighten-factors", type=float, nargs="+", default=[1.0])
    parser.add_argument("--output-folder", default="vis")
    args = parser.parse_args()

    return args

def main(input_file, resize_factor, brighten_factor, output_folder):
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

if __name__ == "__main__":
    args = parse_args() 
    for resize_factor in np.geomspace(0.01,1, num=30):#args.resize_factors:
        for brighten_factor in (1.0,): #args.brighten_factors:
            main(args.input_file, resize_factor, brighten_factor=brighten_factor, output_folder=args.output_folder)