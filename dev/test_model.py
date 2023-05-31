from deepforest import main as deepforest_main
from deepforest import get_data
import os
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
import numpy

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-folder")
    parser.add_argument("--image-extension")
    args = parser.parse_args()

    return args

def main(folder, extension):
    model = deepforest_main.deepforest()
    model.use_release()
    files = Path(folder).glob("*"+extension)
    print(files)
    for file in files:
        img = model.predict_image(path=str(file),return_plot=True)

        #predict_image returns plot in BlueGreenRed (opencv style), but matplotlib likes RedGreenBlue, switch the channel order.
        plt.imshow(img[:,:,::-1])

if __name__ == "__main__":
    args = parse_args() 
    main(args.image_folder, args.image_extension)