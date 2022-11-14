
import pickle
from  util import yolof, summary

if __name__ == "__main__":
    y = yolof("config.json")
    images, boxes, classes = y.flow()
    s = summary(boxes)
    with open("_tmp/results.pkl","wb") as f:
        pickle.dump({"images":images,"boxes":s.done}, f)
