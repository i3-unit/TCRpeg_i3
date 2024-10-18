import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# Add the parent directory to the sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '..', 'TCRpeg_i3/tcrpeg')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'TCRpeg_i3')))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.', 'i3_scripts')))

# Debug: Print sys.path to verify the paths
# print("sys.path:", sys.path)

from tcrpeg.TCRpeg import TCRpeg
from tcrpeg.word2vec import word2vec
from tcrpeg.evaluate import evaluation
from utils import load_data

#from utils import load_data

# Function to configure logging
def configure_logging(log_file):
    # Remove all handlers associated with the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Reconfigure logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file, encoding='utf-8'),  # Ensuring encoding is set
                            logging.StreamHandler()
                        ])

np.random.seed(222)

class TCRpegModel:
    def __init__(self, input_file, output_dir, device='cpu'):
        self.input_file = input_file
        self.output_dir = output_dir
        self.device = device
        self.data = None
        self.sequences = None
        self.id = None
        self.count = None
        self.sequences_train = None
        self.sequences_test = None
        self.model = None

    def prepare_directories_and_filenames(self):
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define subdirectories
        self.p_infer_dir = os.path.join(self.output_dir, "p_infer")
        self.embeddings_dir = os.path.join(self.output_dir, "embeddings")
        self.models_dir = os.path.join(self.output_dir, "models")
        
        # Create subdirectories if they don't exist
        os.makedirs(self.p_infer_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Extract the input file name without extension
        self.input_name = os.path.splitext(os.path.basename(self.input_file))[0]
        
    def load_and_preprocess_data(self, seq_col='sequence', id_col='id', count_col='count'):
        self.data = load_data(self.input_file)
        self.sequences = self.data[seq_col].values

        # Lower case all columns names
        self.data.columns = map(str.lower, self.data.columns)

        # Check if id and count are present in the data
        self.id = self.data[id_col].values if id_col in self.data.columns else np.arange(len(self.data))
        self.count = self.data[count_col].values if count_col in self.data.columns else np.arange(len(self.data))

    def split_data(self, test_size=0.2):
        logging.info("Splitting data...")
        # Perform train-test split and get indices
        self.sequences_train, self.sequences_test, train_idx, test_idx = train_test_split(
            self.sequences, range(len(self.sequences)), test_size=test_size, random_state=42)

        # Log info about the split
        logging.info(f"Train size: {len(self.sequences_train)}")
        logging.info(f"Test size: {len(self.sequences_test)}")

        # Map count using test indices
        self.data_test = {
            'seq': np.array(self.sequences_test),
            'count': np.array(self.count)[test_idx]
        }

        self.data_test['id'] = self.id[test_idx]

    def train_word2vec(self, epochs=10, batch_size=100, learning_rate=1e-4):
        logging.info("Training Word2Vec model...")
        self.aa_emb = word2vec(path=self.sequences_train, epochs=epochs,
                               batch_size=batch_size,
                               device=self.device, lr=learning_rate,
                               window_size=3,
                               record_path=f'{self.embeddings_dir}/{self.input_name}_aa.txt')
        logging.info("Word2Vec model trained successfully.")

    def train_model(self, hidden_size=128, num_layers=5, epochs=20, batch_size=100, learning_rate=1e-4):
        logging.info("Training TCRpeg model...")
        self.model = TCRpeg(hidden_size=hidden_size, num_layers=num_layers, load_data=True, max_length=50,
                            embedding_path=f'{self.embeddings_dir}/{self.input_name}_aa.txt',
                            path_train=self.sequences_train, device=self.device)
        
        self.model.create_model()
        self.model.train_tcrpeg(epochs=epochs, batch_size=batch_size, lr=learning_rate)
        self.model.save(f'{self.models_dir}/{self.input_name}.pth')
        logging.info("TCRpeg model trained successfully.")

    def probability_inference(self):
        logging.info("Performing probability inference...")
        eva = evaluation(model=self.model)

        r,p_data,p_infer = eva.eva_prob(path=self.data_test)

        logging.info(f"Pearson correlation coefficient are : {r}")

        np.save(f'{self.p_infer_dir}/{self.input_name}_p_infer.npy', p_infer)    

        logging.info("Probability inference completed successfully.")    

         # Create a structured array with sequence, id and p_infer
        # structured_array = np.zeros(len(self.sequences_test),
        #                             dtype=[('sequence', 'U50'), ('id', 'U50'), ('p_infer', 'f4')])
        # structured_array['sequence'] = self.sequences_test
        # structured_array['id'] = self.data_test['id']
        # structured_array['p_infer'] = p_infer

        # Save the structured array
        # np.save(f'{self.output_dir}/{self.input_name}_p_infer_structured.npy', structured_array)
     
    def calculate_embeddings(self):
        logging.info("Calculating embeddings...")
        # Calculate embeddings for each sequence
        embeddings = self.model.get_embedding(self.sequences)
        reduced_embeddings = np.mean(embeddings, axis=1)
        np.save(f'{self.embeddings_dir}/{self.input_name}_raw_embeddings.npy', embeddings)        

        #Create a structured array with sequence, id and p_infer
        structured_array = np.zeros(len(self.sequences),
                                    dtype=[('id', 'U50'), ('sequence', 'U100'), ('embedding', 'f4', (640,))])
        
        structured_array['id'] = np.arange(len(self.sequences))
        structured_array['sequence'] = self.sequences
        structured_array['embedding'] = embeddings
        # Save the structured array
        np.save(f'{self.embeddings_dir}/{self.input_name}_structured_embeddings.npy', structured_array)
        logging.info("Embeddings calculated successfully.")

    def run(self, seq_col='sequence', id_col='id', count_col='count', test_size=0.2,
            word2vec_epochs=10, word2vec_batch_size=100, word2vec_learning_rate=1e-4,
            hidden_size=128, num_layers=5, epochs=20, batch_size=100, learning_rate=1e-4):
        self.prepare_directories_and_filenames()
        self.load_and_preprocess_data(seq_col=seq_col, id_col=id_col, count_col=count_col)
        self.split_data(test_size=test_size)
        self.train_word2vec(epochs=word2vec_epochs, batch_size=word2vec_batch_size,
                            learning_rate=word2vec_learning_rate)
        self.train_model(hidden_size=hidden_size, num_layers=num_layers, epochs=epochs, batch_size=batch_size,
                        learning_rate=learning_rate)
        self.probability_inference()
        self.calculate_embeddings()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TCRpeg classification model.')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Directory to save embeddings, models and p-infer',
                        required=True)
    parser.add_argument('-d', '--device', help='Device to use (cpu, cuda:0, mps)', default='cpu',
                        choices=["cpu", "cuda:0", "mps"])
    parser.add_argument('-s', '--seq_col', help='Sequence column required (default: sequence)',
                        default='sequence')
    parser.add_argument('-c', '--count_col', help='Count column optional (default: count)',
                        default='label')
    parser.add_argument('--id', '--id', help='ID column optional (default:id)',
                        default='id')
    parser.add_argument('--word2vec_epochs', help='Number of epochs for word2vec training (default: 10)',
                        type=int, default=10)
    parser.add_argument('--epochs', help='Number of epochs for model training (default: 20)',
                        type=int, default=20)
    parser.add_argument('--word2vec_batch_size', help='Batch size for word2vec training (default: 100)',
                        type=int, default=100)
    parser.add_argument('--word2vec_learning_rate', help='Learning rate for word2vec training (default: 1e-4)',
                        type=float, default=1e-4)
    parser.add_argument('--hidden_size', help='Hidden size for model training (default: 128)',
                        type=int, default=128)
    parser.add_argument('--num_layers', help='Number of layers for model training (default: 5)',
                        type=int, default=5)
    parser.add_argument('--batch_size', help='Batch size for model training (default: 100)',
                        type=int, default=100)
    parser.add_argument('--learning_rate', help='Learning rate for model training (default: 1e-4)',
                        type=float, default=1e-4)
    parser.add_argument('--test_size', help='Test size for train-test split (default: 0.2)',
                        type=float, default=0.2)
    parser.add_argument('--log', help='Log file to save logs (default: tcrpeg_p-infer.log)',
                        default='tcrpeg_p-infer.log')

    # Parse the arguments
    args = parser.parse_args()

    # Configure logging with the specified log file name
    print(f"Writing logs to {args.log}")
    # Remove log file 
    os.remove(args.log) if os.path.exists(args.log) else None
    configure_logging(args.log)

    logging.info("Starting TCRpegModel probability inference")
    # Log the arguments
    logging.info(f"Running parameters: {vars(args)}")

    # Check if MPS is available and set the device accordingly
    if args.device == 'mps' and not torch.backends.mps.is_available():
        logging.warning("MPS device requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    model_infer = TCRpegModel(input_file=args.input, output_dir=args.output, device=args.device)
    model_infer.run(test_size=args.test_size, word2vec_epochs=args.word2vec_epochs,
                    word2vec_batch_size=args.word2vec_batch_size, word2vec_learning_rate=args.word2vec_learning_rate,
                    hidden_size=args.hidden_size, num_layers=args.num_layers, epochs=args.epochs,
                    batch_size=args.batch_size, learning_rate=args.learning_rate)
    
    logging.info("Model training and inference completed successfully.")
