#!/usr/bin/env python
# _*_coding:utf-8_*_

import sys
import itertools
import pickle
import os,sys,re
from collections import Counter
import pandas as pd
import numpy as np
import argparse
import csv
import keras
from keras.models import load_model

def binary(sequences):
    AA = 'ACGT'
    binary_feature = []
    for seq in sequences:
        binary = []
        for aa in seq:
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                binary.append(tag)
        binary_feature.append(binary)
    return binary_feature


def read_fasta(inputfile):
    if os.path.exists(inputfile) == False:
        print('Error: file " %s " does not exist.' % inputfile)
        sys.exit(1)
    with open(inputfile) as f:
        record = f.readlines()
    if re.search('>', record[0]) == None:
        print('Error: the input file " %s " must be fasta format!' % inputfile)
        sys.exit(1)

    data = {}
    for line in record:
        if line.startswith('>'):
            name = line.replace('>', '').split('\n')[0]
            data[name] = ''
        else:
            data[name] += line.replace('\n', '')
    return data


def extract_features(data):
    sequences = data
    feature_vector = np.vstack(binary(sequences))
    return feature_vector


def main():
    parser = argparse.ArgumentParser(description='DEPICTER: A multiple deep neural networks learning-based approach for predicting eukaryotic promoters')
    parser.add_argument('--input',dest='inputfile',type=str,required=True,help='query sequences to be predicted in fasta format.')
    parser.add_argument('--output',dest='outputfile',type=str,required=False,help='save the prediction results.')

    parser.add_argument('--species', dest='speciesfile', type=str, required=False,
                        help='--species indicates the specific species, currently we accept \'Human\' or \'Mouse\' or \'Arabidopsis\' or \'Drosophila\'.\n \
                        if --kinds is \'prokaryotic\', you do not enter this item.', default=None)
    parser.add_argument('--type', dest='typefile', type=str, required=False,
                        help='sequences type that to be predicted, and we accept \'TATA+\' or \'TATA-\' or \'TATA+_TATA-\'\n ', default=None)
    parser.add_argument('--select_all', type=str, required=False, help='If select all species or all types.')
    args = parser.parse_args()

    inputfile = args.inputfile
    outputfile = args.outputfile
    speciesfile = args.speciesfile
    typefile = args.typefile
    data = read_fasta(inputfile)
    outputfile_original = outputfile
    if outputfile_original==None:
        outputfile_original = ''
    try:
        if args.select_all=='species_all':
            for speciesfile in ['Human', 'Mouse', 'Arabidopsis', 'Drosophila']:
                if outputfile_original == None:
                    outputfile = 'output'
                outputfile = outputfile_original+'_'+speciesfile
                vector = extract_features(data.values())
                df = pd.DataFrame(vector)
                feature_names = []
                for i in range(0, len(df.columns)):
                    feature_names.append(df.columns[i])
                ppp = df[feature_names]
                ppp = ppp.values.reshape(len(ppp), 40, 30)
                y = np.array([1] * len(ppp))
                y = keras.utils.to_categorical(y, 2)
                if speciesfile == 'Human':
                    if typefile == 'TATA+':
                        model = "nogradientstop"
                        from capsulenet_With import Capsnet_main
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7, lam_recon=0.5, routings=3, modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'HumanWith.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA-':
                        from capsulenet_Non import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'HumanNon.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA+_TATA-':
                        from capsulenet_Com import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'HumanCom.h5')
                        predictions, score = model[1].predict(ppp)
                elif speciesfile == 'Mouse':
                    if typefile == 'TATA+':
                        from capsulenet_With import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=64,lam_recon=0.01, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'MouseWith.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA-':
                        from capsulenet_Non import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=58,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'MouseNon.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA+_TATA-':
                        from capsulenet_Com import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'MouseCom.h5')
                        predictions, score = model[1].predict(ppp)
                elif speciesfile == 'Arabidopsis':
                    if typefile == 'TATA+':
                        from capsulenet_With import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.000216, batch_size=45,lam_recon=0.4993, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'ArabidopsisWith.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA-':
                        from capsulenet_Non import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'ArabidopsisNon.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA+_TATA-':
                        from capsulenet_Com import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'ArabidopsisCon.h5')
                        predictions, score = model[1].predict(ppp)
                elif speciesfile == 'Drosophila':
                    if typefile == 'TATA+':
                        from capsulenet_With import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'DrosophilaWith.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA-':
                        from capsulenet_Non import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'DrosophilaNon.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA+_TATA-':
                        from capsulenet_Com import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,                                         lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'DrosophilaCom.h5')
                        predictions, score = model[1].predict(ppp)
                decision = []
                for i in range(len(predictions)):
                    if predictions[i,0] > predictions[i,1]:
                        decision.append(predictions[i,0])
                    else:
                        decision.append(predictions[i,1])
                predictions = predictions.argmax(axis=1)
                probability = ['%.5f' % float(i) for i in predictions]
                name = list(data.keys())
                seq = list(data.values())
                decisions = ['%.5f' % float(i) for i in decision]
                with open(outputfile, 'w') as f:
                    for i in range(len(data)):
                        if float(probability[i]) > 0.5:
                            f.write(probability[i] + '*'  + '\t')
                            f.write(decisions[i] + '\t')
                            f.write(name[i] + '\t')
                            f.write(seq[i] + '\n')
                        else:
                            f.write(probability[i] + '\t')
                            f.write(name[i] + '\t')
                            f.write(seq[i] + '\n')
                print('output are saved in ' + outputfile + ', and those identified as promoters are marked with *')
        elif args.select_all=='type_all':
            for typefile in ['TATA+', 'TATA-', 'TATA+_TATA-']:
                if outputfile_original == None:
                    outputfile = 'output'
                outputfile = outputfile_original+'_'+typefile
                vector = extract_features(data.values())
                df = pd.DataFrame(vector)
                feature_names = []
                for i in range(0, len(df.columns)):
                    feature_names.append(df.columns[i])
                ppp = df[feature_names]
                ppp = ppp.values.reshape(len(ppp), 40, 30)
                y = np.array([1] * len(ppp))
                y = keras.utils.to_categorical(y, 2)
                if speciesfile == 'Human':
                    if typefile == 'TATA+':
                        model = "nogradientstop"
                        from capsulenet_With import Capsnet_main
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7, lam_recon=0.5, routings=3, modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'HumanWith.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA-':
                        from capsulenet_Non import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'HumanNon.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA+_TATA-':
                        from capsulenet_Com import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'HumanCom.h5')
                        predictions, score = model[1].predict(ppp)
                elif speciesfile == 'Mouse':
                    if typefile == 'TATA+':
                        from capsulenet_With import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=64,lam_recon=0.01, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'MouseWith.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA-':
                        from capsulenet_Non import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=58,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'MouseNon.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA+_TATA-':
                        from capsulenet_Com import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'MouseCom.h5')
                        predictions, score = model[1].predict(ppp)
                elif speciesfile == 'Arabidopsis':
                    if typefile == 'TATA+':
                        from capsulenet_With import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.000216, batch_size=45,lam_recon=0.4993, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'ArabidopsisWith.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA-':
                        from capsulenet_Non import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'ArabidopsisNon.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA+_TATA-':
                        from capsulenet_Com import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'ArabidopsisCon.h5')
                        predictions, score = model[1].predict(ppp)
                elif speciesfile == 'Drosophila':
                    if typefile == 'TATA+':
                        from capsulenet_With import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'DrosophilaWith.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA-':
                        from capsulenet_Non import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'DrosophilaNon.h5')
                        predictions, score = model[1].predict(ppp)
                    elif typefile == 'TATA+_TATA-':
                        from capsulenet_Com import Capsnet_main
                        model = "nogradientstop"
                        model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,                                         lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                        model[1].load_weights(r'DrosophilaCom.h5')
                        predictions, score = model[1].predict(ppp)
                decision = []
                for i in range(len(predictions)):
                    if predictions[i,0] > predictions[i,1]:
                        decision.append(predictions[i,0])
                    else:
                        decision.append(predictions[i,1])
                predictions = predictions.argmax(axis=1)
                probability = ['%.5f' % float(i) for i in predictions]
                name = list(data.keys())
                seq = list(data.values())
                decisions = ['%.5f' % float(i) for i in decision]
                with open(outputfile, 'w') as f:
                    for i in range(len(data)):
                        if float(probability[i]) > 0.5:
                            f.write(probability[i] + '*'  + '\t')
                            f.write(decisions[i] + '\t')
                            f.write(name[i] + '\t')
                            f.write(seq[i] + '\n')
                        else:
                            f.write(probability[i] + '\t')
                            f.write(name[i] + '\t')
                            f.write(seq[i] + '\n')
                print('output are saved in ' + outputfile + ', and those identified as promoters are marked with *')
        else:
            if outputfile == None:
                outputfile = 'output'
            vector = extract_features(data.values())
            df = pd.DataFrame(vector)
            feature_names = []
            for i in range(0, len(df.columns)):
                feature_names.append(df.columns[i])
            ppp = df[feature_names]
            ppp = ppp.values.reshape(len(ppp), 40, 30)
            y = np.array([1] * len(ppp))
            y = keras.utils.to_categorical(y, 2)
            if speciesfile == 'Human':
                if typefile == 'TATA+':
                    model = "nogradientstop"
                    from capsulenet_With import Capsnet_main
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7, lam_recon=0.5, routings=3, modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'HumanWith.h5')
                    predictions, score = model[1].predict(ppp)
                elif typefile == 'TATA-':
                    from capsulenet_Non import Capsnet_main
                    model = "nogradientstop"
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'HumanNon.h5')
                    predictions, score = model[1].predict(ppp)
                elif typefile == 'TATA+_TATA-':
                    from capsulenet_Com import Capsnet_main
                    model = "nogradientstop"
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'HumanCom.h5')
                    predictions, score = model[1].predict(ppp)
            elif speciesfile == 'Mouse':
                if typefile == 'TATA+':
                    from capsulenet_With import Capsnet_main
                    model = "nogradientstop"
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=64,lam_recon=0.01, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'MouseWith.h5')
                    predictions, score = model[1].predict(ppp)
                elif typefile == 'TATA-':
                    from capsulenet_Non import Capsnet_main
                    model = "nogradientstop"
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=58,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'MouseNon.h5')
                    predictions, score = model[1].predict(ppp)
                elif typefile == 'TATA+_TATA-':
                    from capsulenet_Com import Capsnet_main
                    model = "nogradientstop"
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'MouseCom.h5')
                    predictions, score = model[1].predict(ppp)
            elif speciesfile == 'Arabidopsis':
                if typefile == 'TATA+':
                    from capsulenet_With import Capsnet_main
                    model = "nogradientstop"
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.000216, batch_size=45,lam_recon=0.4993, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'ArabidopsisWith.h5')
                    predictions, score = model[1].predict(ppp)
                elif typefile == 'TATA-':
                    from capsulenet_Non import Capsnet_main
                    model = "nogradientstop"
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'ArabidopsisNon.h5')
                    predictions, score = model[1].predict(ppp)
                elif typefile == 'TATA+_TATA-':
                    from capsulenet_Com import Capsnet_main
                    model = "nogradientstop"
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'ArabidopsisCon.h5')
                    predictions, score = model[1].predict(ppp)
            elif speciesfile == 'Drosophila':
                if typefile == 'TATA+':
                    from capsulenet_With import Capsnet_main
                    model = "nogradientstop"
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'DrosophilaWith.h5')
                    predictions, score = model[1].predict(ppp)
                elif typefile == 'TATA-':
                    from capsulenet_Non import Capsnet_main
                    model = "nogradientstop"
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'DrosophilaNon.h5')
                    predictions, score = model[1].predict(ppp)
                elif typefile == 'TATA+_TATA-':
                    from capsulenet_Com import Capsnet_main
                    model = "nogradientstop"
                    model = Capsnet_main(ppp, y, nb_epoch=1, compiletimes=0, lr=0.0001, batch_size=7,                                         lam_recon=0.5, routings=3,modeltype=model, nb_classes=2, predict=True)  # only to get config
                    model[1].load_weights(r'DrosophilaCom.h5')
                    predictions, score = model[1].predict(ppp)
            decision = []
            for i in range(len(predictions)):
                if predictions[i,0] > predictions[i,1]:
                    decision.append(predictions[i,0])
                else:
                    decision.append(predictions[i,1])
            predictions = predictions.argmax(axis=1)
            probability = ['%.5f' % float(i) for i in predictions]
            name = list(data.keys())
            seq = list(data.values())
            decisions = ['%.5f' % float(i) for i in decision]
            with open(outputfile, 'w') as f:
                for i in range(len(data)):
                    if float(probability[i]) > 0.5:
                        f.write(probability[i] + '*'  + '\t')
                        f.write(decisions[i] + '\t')
                        f.write(name[i] + '\t')
                        f.write(seq[i] + '\n')
                    else:
                        f.write(probability[i] + '\t')
                        f.write(name[i] + '\t')
                        f.write(seq[i] + '\n')
            print('output are saved in ' + outputfile + ', and those identified as promoters are marked with *')
    except Exception as e:
        print('Please check the format of your predicting data!')
        sys.exit(1)



if __name__ == "__main__":
    main()


