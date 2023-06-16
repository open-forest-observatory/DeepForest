from deepforest import main as deepforest_main
from deepforest import get_data
import os
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
from imageio import imwrite
import numpy

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-folder")
    parser.add_argument("--image-extension", default="")
    parser.add_argument("--output-folder")
    args = parser.parse_args()

    return args

def main(folder, extension, output_folder):
    model = deepforest_main.deepforest()
    model.use_release()
    files = Path(folder).glob("*"+extension)
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
    
    for file in files:
        print(file)
        pred = model.predict_image(path=str(file),return_plot=True)
        if pred is None:
            print(f"file {file} is invalid")
            continue

        if output_folder is not None:
            output_file = Path(output_folder, file.name)
            #predict_image returns plot in BlueGreenRed (opencv style), but matplotlib likes RedGreenBlue, switch the channel order.
            imwrite(output_file, pred[:,:,::-1])
        else:
            #predict_image returns plot in BlueGreenRed (opencv style), but matplotlib likes RedGreenBlue, switch the channel order.
            plt.imshow(pred[:,:,::-1])
            plt.show()

if __name__ == "__main__":
    args = parse_args() 
    main(args.image_folder, args.image_extension, output_folder=args.output_folder)