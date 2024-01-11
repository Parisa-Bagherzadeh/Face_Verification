import cv2
import numpy as np

from argparse import ArgumentParser

from insightface.app import FaceAnalysis



def verification(image1, image2):
    THRESHOLD = 25

    app = FaceAnalysis(name = "buffalo_S", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id = 0, det_size = (640, 640))

    image_1 = cv2.imread(image1)
    image_2 = cv2.imread(image2)

    result_1 = app.get(image_1)
    embedding_1 = result_1[0]["embedding"]

    result_2 = app.get(image_2)
    embedding_2 = result_2[0]["embedding"]

    dist = np.sqrt(np.sum((embedding_1 - embedding_2)**2))
    
    if dist < THRESHOLD : 
        print("Same Person :)")
    else:
        print("Different persons")    





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image1", type = str, default = "images/obama1.jpg" , help = "path of image1")
    parser.add_argument("--image2", type = str, default = "images/obama2.jpg",  help = "path of image2")

    args = parser.parse_args()

    verification(args.image1, args.image2)
