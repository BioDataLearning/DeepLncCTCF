#!/usr/bin/python

from utils import *
from train import *
import os,sys
import argparse
import numpy as np
import subprocess
from keras import backend as K
np.random.seed(12345)

def get_motifs(model,X,Y,output_dir,output_dir2,label):
    layer_names = [l.name for l in model.layers]
    conv_layer_index = layer_names.index('conv1d_1')
    conv_layer = model.layers[conv_layer_index]
    num_motifs = conv_layer.filters
    window = conv_layer.kernel_size[0]
    conv_output = conv_layer.get_output_at(0)
    f = K.function([model.input], [K.max(K.max(conv_output, axis=1), axis=0)])
    f_seq = K.function([model.input], [K.argmax(conv_output, axis=1), K.max(conv_output, axis=1)])
    f_act = K.function([model.input],[conv_output])
    motifs = np.zeros((num_motifs, window, 4))
    nsites = np.zeros(num_motifs)
    nseqs = np.zeros(num_motifs)
    Y_pos = [i for i,e in enumerate(Y) if e ==label]
    X_pos = X[Y_pos]
    nsamples = len(X_pos)
    mean_acts = np.zeros((num_motifs, nsamples))
    z = f([X_pos])
    max_motif = z[0]
    thr_per = 0.5
    z_seq = f_seq([X_pos])
    max_inds = z_seq[0]
    max_acts = z_seq[1]
    z_act = f_act([X_pos])
    acts = z_act[0]
    for m in range(num_motifs):
        for n in range(len(X_pos)):
            if max_acts[n, m] > thr_per*max_motif[m]:
                nseqs[m] +=1

    ##get the filter activity and locations on the input sequence
    act_file = open(output_dir+'motifs'+str(label)+'_act', 'w')
    loc_file = open(output_dir+'motifs'+str(label)+'_loc', 'w')
    for m in range(num_motifs):
        for n in range(len(X_pos)):
            for j in range(acts.shape[1]):
                weight = (100-abs(j-100))/100
                mean_acts[m,n] += acts[n,j,m]*weight
                if acts[n, j, m] > thr_per * max_motif[m]:
                    nsites[m] += 1
                    motifs[m] += X_pos[n, j:j+window, :]
                    loc_file.write("M%i %i %i\n" % (m, j, j+window))    

    for m in range(num_motifs):
        act_file.write("M%i" % (m))
        for n in range(len(X_pos)):
            act_file.write("\t%0.4f" % (mean_acts[m,n]))
        act_file.write("\n")

    for m in range(num_motifs):
        seqfile = open(output_dir2+'motif'+str(m)+'.fasta', 'w')
        for n in range(len(X_pos)):
            for j in range(acts.shape[1]): 
                if acts[n, j, m] > thr_per * max_motif[m]:
                    nsites[m] += 1
                    motifs[m] += X_pos[n, j:j+window, :]
                    kmer = one_hot_to_seq(X_pos[n, j:j+window, :])         
                    seqfile.write('>%d_%d' % (n,j))
                    seqfile.write('\n')
                    seqfile.write(kmer)
                    seqfile.write('\n')
        filename = output_dir2+'motif'+str(m)+'.fasta'
        plot_motif(output_dir2,m,filename)            

    print('Making motifs')
    motifs = motifs[:, :, [0, 3, 2, 1]]
    motifs_file = open(output_dir+'motifs'+str(label)+'.txt', 'w')
    motifs_file.write('MEME version 4.9.0\n\n'
                  'ALPHABET= ACGU\n\n'
                  'strands: + -\n\n'
                  'Background letter frequencies (from uniform background):\n'
                  'A 0.25000 C 0.25000 G 0.25000 U 0.25000\n\n')
    for m in range(num_motifs):
        if nsites[m] == 0:
            continue
        motifs_file.write('MOTIF M%i O%i\n' % (m, m))
        motifs_file.write("letter-probability matrix: alength= 4 w= %i nsites= %i nseqs= %i E= 1337.0e-6\n" % (window, nsites[m], nseqs[m]))
        for j in range(window):
            motifs_file.write("%f %f %f %f\n" % tuple(1.0 * motifs[m, j, 0:4] / np.sum(motifs[m, j, 0:4])))
        motifs_file.write('\n')

def one_hot_to_seq(matrix):
    nts = ['A','U','G','C']
    seqs = []
    index = [np.where(r==1)[0][0] for r in matrix]
    for i in index:
        seqs.append(nts[i])
    seq = ''.join(seqs)
    return seq

def plot_motif(output_dir,m,seqfile):
    weblogo_opts = '-F pdf -X NO --errorbars NO --fineprint ""'
    weblogo_opts += ' -C "#FF0000" A A'
    weblogo_opts += ' -C "#0000FF" C C'
    weblogo_opts += ' -C "#FFD700" G G'
    weblogo_opts += ' -C "#008000" U U'
    logofile = output_dir+'motif'+str(m)+'_logo.pdf'
    weblogo_dir = '/home/skuang/software/weblogo-master'
    weblogo_cmd = '%s/weblogo %s -f %s -o %s' % (weblogo_dir, weblogo_opts, seqfile, logofile)
    print(weblogo_cmd)
    subprocess.call(weblogo_cmd, shell=True)

def main():
    parser = argparse.ArgumentParser(description="progrom usage")
    parser.add_argument("-f", "--fasta", type=str, help="positive instances")
    parser.add_argument("-n", "--negative", type=str, help="negatve instances")
    args = parser.parse_args()
    infile = args.fasta
    secondin = args.negative
    X_train,X_val,X_test,Y_train,Y_val,Y_test = get_data(infile,secondin)
    best = {'batch_size': 7.0, 'dense_unit': 784.0, 'drop_out_cnn': 0.46700349282456455, 'drop_out_lstm': 0.45445161526885014, 'filter': 112.0, 'kernel_initializer': 'glorot_normal', 'l2_reg': 3.5109491470074096e-05, 'learning_rate': 0.0019135917105472034, 'lstm_unit': 256.0, 'pool_size': 5.0, 'window_size': 15.0}
    dnn_model = get_model(best)
    dnn_model.load_weights('human.bestmodel.hdf5')
    output_dir = 'motifs/'
    output_dir2 = 'motifs_logo/'
    get_motifs(dnn_model,X_test,Y_test,output_dir,output_dir2,0)
    get_motifs(dnn_model,X_test,Y_test,output_dir,output_dir2,1)

if __name__ == '__main__':
        main()
