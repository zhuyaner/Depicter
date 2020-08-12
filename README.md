# Depicter
Requirements

 h5py==2.9.0rc1
 
 Keras==2.1.1
 
 numpy==1.16.5
 
 pandas==0.25.1
 
 scikit-learn==0.21.3
 
 scipy==1.3.1
 
 tensorflow==1.13.1


For advanced users who want to perform prediction by using their own data:
 To get the information the user needs to enter for help, run:
    python prom_pred.py --help
 or
    python prom_pred.py -h
   
as follows:

>python prom_pred.py -h
Using TensorFlow backend.
usage: prom_pred.py [-h] --input INPUTFILE [--output OUTPUTFILE]  --species SPECIESFILE [--type TYPEFILE]

Depicter: a multiple deep neural networks learning-based approach for predicting eukaryotic promoters

optional arguments:
  -h, --help            show this help message and exit
  --input INPUTFILE     query sequences to be predicted in fasta format.
  --output OUTPUTFILE   save the prediction results.
  --species SPECIESFILE
                        --species indicates the specific species, 
                        currently we accept 'Human' or
                        'Mouse' or 'Arabidopsis' or 'Drosophila'.
  --type TYPEFILE       sequences type that to be predicted, only used when
                        --kinds is 'eukaryote'. we accept 'TATA+' or 'TATA-'
                        or 'TATA+_TATA-'.
