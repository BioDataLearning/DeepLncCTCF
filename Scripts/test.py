#!/usr/bin/python

from utils import *
from train import *
import os,sys
import argparse
np.random.seed(12345)

def test(infile,infile2,outfile):    
    fileout = open(outfile, "w")
    X, Y = get_test_data(infile,infile2)
    best = {'batch_size': 7.0, 'dense_unit': 784.0, 'drop_out_cnn': 0.46700349282456455, 'drop_out_lstm': 0.45445161526885014, 'filter': 112.0, 'kernel_initializer': 'glorot_normal', 'l2_reg': 3.5109491470074096e-05, 'learning_rate': 0.0019135917105472034, 'lstm_unit': 256.0, 'pool_size': 5.0, 'window_size': 15.0}
    dnn_model = get_model(best)
    filepa = "human.bestmodel.hdf5"
    dnn_model.load_weights(filepa)
    predictions = dnn_model.predict(X,batch_size=2**int(best['batch_size'])) 
    pred_train_prob = predictions
    rounded = [round(x[0]) for x in predictions]
    metrics(Y, rounded, pred_train_prob, fileout)

def main():
    parser = argparse.ArgumentParser(description="progrom usage")
    parser.add_argument("-f", "--fasta", type=str, help="positive instances")
    parser.add_argument("-n", "--negative", type=str, help="negatve instances")
    parser.add_argument("-o", "--out", type=str, help="prediction output")
    args = parser.parse_args()
    infile = args.fasta
    secondin = args.negative
    outfile = args.out
    test(infile,secondin,outfile)

if __name__ == '__main__':
        main()
