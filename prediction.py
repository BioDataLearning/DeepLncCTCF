#!/usr/bin/python

from utils import *
from train import *
import os,sys
import argparse

def prediction(infile,outfile):
    fileout = open(outfile, "w")
    X,geneinfo = get_pred_data(infile)
    best = {'batch_size': 4.0, 'dense_unit': 80.0, 'drop_out_cnn': 0.2738070724985381, 'drop_out_lstm': 0.16261503928101084, 'filter': 128.0, 'kernel_initializer': 'random_uniform', 'l2_reg': 1.0960198460047699e-05, 'learning_rate': 0.00028511592517082153, 'lstm_unit': 624.0, 'pool_size': 3.0, 'window_size': 9.0}
    dnn_model = get_model(best)
    dnn_model.load_weights('human.bestmodel.hdf5')
    predictions = dnn_model.predict(X,batch_size=2**int(best['batch_size'])) 
    pred_train_prob = predictions
    rounded = [round(x[0]) for x in predictions]
    fileout.write("Prediction:\n")
    for i in range(len(geneinfo)):
        fileout.write(str(geneinfo[i]))
        fileout.write("\t")
        fileout.write(str(rounded[i]))
        fileout.write("\t")
        fileout.write(str(pred_train_prob[i]))
        fileout.write("\n")

def main():
    parser = argparse.ArgumentParser(description="progrom usage")
    parser.add_argument("-f", "--fasta", type=str, help="prediction instances")
    parser.add_argument("-o", "--out", type=str, help="prediction output")
    args = parser.parse_args()
    infile = args.fasta
    outfile = args.out
    prediction(infile,outfile)

if __name__ == '__main__':
        main()
