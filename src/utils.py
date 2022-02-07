import argparse

import config



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', nargs="?",type=str ,default=config.DATA_PATH)
    parser.add_argument('--save_path',default=config.MODEL_SAVE_PATH)
    parser.add_argument('--load_model',default=None)
    parser.add_argument('--k',type=int,default=5)
    return parser.parse_args()




def accuracy(predictions): 
    cm = predictions.select("label", "prediction")
    acc = cm.filter(cm.label == cm.prediction).count() / cm.count()
    return acc