# DeepLncCTCF for identification and analysis of consensus RNA motifs binding to the genome regulator CTCF
DeepLncCTCF is a deep learning model to discover the RNA recognition patterns of CTCF and to identify candidate lncRNAs that may interact with CTCF. It utilized convolutional neural networks (CNNs) and attention-based bi-directional long short-term memory (BLSTM) network. We implemented the DeepLncCTCF model in Python using Keras 2.2.4 on a high performance computing cluster.

## Requirements
- python3
- numpy 
- pandas
- sklearn
- keras >=2.0
- tensorflow
- h5py

## Input Format
The input files are in FASTA format, but the description line (the line with ">" symbol in the begining) should start with class label. An example seqeunce is as follows:
```
>1      chr1    15882   16083   -
CGGCCTCCCCAGCGCAGGGCTCCTCGTTTGAGGGGAGGTGACTTCCCTCCCAGCAGGCTCTTGGACACAGTAAGCTTCCCCAGCCCTGCCTGAGCAGCCTTTCCTCCTTGCCCTGTTCCCCACCTCCCGGCTCCAGTCCAGGGAGCTCCCAGGGAAGTGGTTGACCCCTCCGGTGGCTGGCCACTCTGCTAGAGTCCATCC
```

## Training and Evaluation
Our example data for the model are available in the Data directory. If you want to train your own model with DeepLncCTCF, you can just substitute the input with your own data. The command line to train and evaluate DeepLncCTCF as follows:
```
$ python train.py -f positive.example.seq -n negative.example.seq -o output
```
During the training, the best weights will be automatically stored in a "hdf5" file. Our fully trained model have been uploaded in the Weights directory.

## Testing 
If you want to evaluate the model on a separate test data, first, you can run the following command line:
```
$ python test.py -f test.positive.example.seq -n test.negative.example.seq -o test.output
```
Please make sure to download the "hdf5" file in the Weights directory or generate your own best weights.

## Motif Visualization
If you want to visualize the kernals of the first convolution layer and get its frequency and location information, you can run the following command line：
```
$ python get_motifs.py -f positive.example.seq -n negative.example.seq
```
Same as the Testing process, "hdf5" file with the best weights is needed. 

## Predicting CTCF-binding RNA sites on lncRNAs
We applied the trained DeepLncCTCF model to predict CTCF-binding RNA sites on human lncRNAs, which were further used to select candidate CTCF-binding lncRNAs. To predict the CTCF-binding RNA sites on lncRNAs using trained DeepLncCTCF model, you can run the following command line:
```
$ python prediction.py -f lncRNA.example.seq -o prediction.output
```