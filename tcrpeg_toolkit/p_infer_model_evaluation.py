import os, sys
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import seaborn as sns
from sklearn.cluster import KMeans
import argparse
import logging
from tcrpeg.evaluate import evaluation
from tcrpeg.TCRpeg import TCRpeg


# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'TCRpeg_i3')))

class PinferCalculation:
    def __init__(self, model_file, embedding_file, sequence_file, output_dir, device='cpu'):
        self.model_file = model_file
        self.embedding_file = embedding_file
        self.data_test = sequence_file
        self.output_dir = output_dir
        self.device = device
        self.model = None
        self.data = None
        
    def prepare_directories_and_filenames(self):
    # Create the analysis output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.analysis_dir = os.path.join(self.output_dir, "analysis/Pinfer")
        os.makedirs(self.analysis_dir, exist_ok=True)
    
    # Extract the model file name without extension
        self.input_name = os.path.basename(self.data_test).split('.csv')[0]

    def read_model(self, hidden_size, num_layers):
        # Create an instance of TCRpeg and load the model
        self.model = TCRpeg(hidden_size=hidden_size, num_layers=num_layers, embedding_path=self.embedding_file, device=self.device)
        self.model.create_model(load=True, path=self.model_file)
        logging.info("Model loaded successfully.")
        
    def read_data(self):
        self.data = pd.read_csv(self.data_test)
        # Check if id is present in the data
        self.id = np.arange(len(self.data))
        self.data['id'] = self.id
        
        self.data_test = {
            'seq': np.array(self.data.sequence),
            'count': np.array(self.data['count']),
            'id': np.array(self.data.id)
        }
      
    def calculate_pinfer(self):
        eva = evaluation(model=self.model)
        p_infer = eva.eva_prob(path=self.data_test)[2]
        logging.info("Pinfer calculated successfully.")

        #Create a structured array with sequence, id and p_infer
        structured_array = np.zeros(len(self.data.sequence),
                                    dtype=[('id', 'U50'), ('sequence', 'U100'), ('pinfer', 'f4')])
        
        structured_array['id'] = self.data_test['id']
        structured_array['sequence'] = self.data_test['seq']
        structured_array['embedding'] = p_infer
        # Save the structured array
        np.save(f'{self.analysis_dir}/{self.input_name}_structured_pinfer.npy', structured_array)


    def run(self, **kwargs):
        self.prepare_directories_and_filenames()
        self.read_model()
        self.calculate_pinfer()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TCRpeg classification model.')
    parser.add_argument('-i', '--model', help='Model file', required=True)
    parser.add_argument('-i', '--sequences', help='Sequence file', required=True)
    parser.add_argument('-i', '--embeddings', help='Embedding file', required=True)
    parser.add_argument('-o', '--output', help='Directory to save embeddings, models and p-infer',
                        required=True)
    parser.add_argument('--hidden_size', help='Hidden size in the trained model (default: 128, check log file for verification)',
                        type=int, default=128)
    parser.add_argument('--num_layers', help='Number of layers in the trained model (default: 5, check log file for verification)',
                        type=int, default=5)
    parser.add_argument('-d', '--device', help='Device to use (cpu, cuda:0, mps)', default='cpu',
                        choices=["cpu", "cuda:0", "mps"])

    # Parse the arguments
    args = parser.parse_args()

    # Check if MPS is available and set the device accordingly
    if args.device == 'mps' and not torch.backends.mps.is_available():
        logging.warning("MPS device requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    model_infer = PinferCalculation(model_file=args.model, sequence_file=args.sequences, output_dir=args.output, device=args.device, hidden_size=args.hidden_size, num_layers=args.num_layers)
    model_infer.run()
    
    logging.info("Inference completed successfully.")