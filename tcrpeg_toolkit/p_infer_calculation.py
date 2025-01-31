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
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'TCRpeg_i3')))

from tcrpeg.TCRpeg import TCRpeg
from tcrpeg.word2vec import word2vec
from tcrpeg.evaluate import evaluation

from tcrpeg_toolkit.utils import load_data

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

#improve word2vec path as option
#improve return without output dir


#todo check if better to separate model and p_infer in two classes
class TCRpegModel:
    def __init__(self, input_file, output_dir, device='cpu', name=None):
        self.input_file = input_file
        self.output_dir = output_dir
        self.embedding_file = None
        self.device = device
        self.data = None
        self.sequences = None
        self.id = None
        self.count = None
        self.sequences_train = None
        self.sequences_test = None
        self.model = None
        self.input_name = name
        self.vj = None

    def prepare_directories_and_filenames(self):
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
       
        # Define subdirectories
        self.p_infer_dir = os.path.join(self.output_dir, "p_infer")
        self.embeddings_dir = os.path.join(self.output_dir, "embeddings")
        self.models_dir = os.path.join(self.output_dir, "models")
        
        # Create subdirectories if they don't exist
        os.makedirs(self.p_infer_dir, exist_ok=True)
        os.makedirs(f"{self.p_infer_dir}/raw", exist_ok=True)
        os.makedirs(f"{self.p_infer_dir}/structured", exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(f"{self.embeddings_dir}/word2vec_aa", exist_ok=True)
        os.makedirs(f"{self.embeddings_dir}/raw", exist_ok=True)
        os.makedirs(f"{self.embeddings_dir}/structured", exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        if self.input_name is None:
            # Extract the input file name without extension
            self.input_name = os.path.splitext(os.path.basename(self.input_file))[0]

    #improve this to a utils or a base class and unique vs not in train here
    
    def load_and_preprocess_data(self, seq_col='sequence', id_col='id', count_col='count', v_col='v', j_col='j', vj=False):
        self.data = load_data(self.input_file)
        
        # Lower case all columns names
        self.data.columns = map(str.lower, self.data.columns)
        
        self.sequences = self.data[seq_col].values
        # Check if V and J genes are present in the data
        self.vj = vj
        if self.vj:
            self.v_genes = self.data[v_col.lower()].values
            self.j_genes = self.data[j_col.lower()].values
            self.sequences_vj = [list(item) for item in zip(self.sequences, self.v_genes, self.j_genes)]
        
        # Check if id and count are present in the data
        self.ids = self.data[id_col].values if id_col in self.data.columns else np.arange(len(self.data))
        self.count = self.data[count_col].values if count_col in self.data.columns else np.arange(len(self.data))

        # Create dictionary using original sequences
        if self.vj:
            # Create dictionary using sequence, v, j as key
            self.id_seq_dict = dict(zip(zip(self.sequences, self.v_genes, self.j_genes), self.ids))
        else:
            self.id_seq_dict = dict(zip(self.sequences, self.ids))

    def split_data(self, test_size=0.2):
        logging.info("Splitting data...")

        if self.vj:
            # Split data
            train_data, test_data, train_idx, test_idx = train_test_split(
                self.sequences_vj, range(len(self.sequences_vj)),
                test_size=test_size, random_state=42)

            # Convert train and test data to lists
            train_data = [list(item) for item in train_data]
            test_data = [list(item) for item in test_data]

            # Unzip the data
            self.sequences_train, v_train, j_train = zip(*train_data)
            self.sequences_test, v_test, j_test = zip(*test_data)

            # Convert to lists
            self.sequences_train = list(self.sequences_train)
            self.sequences_test = list(self.sequences_test)

            # Store VJ data
            self.sequences_train_vj = train_data
            self.sequences_test_vj = test_data
                        
            # self.data_test = {
            #     'seq': np.array(self.sequences_test),
            #     'count': np.array(self.count)[test_idx],
            #     'id': self.ids[test_idx],
            #     'v': np.array(v_test),
            #     'j': np.array(j_test)
            # }

            self.data_test = {
                # 'seq': [self.sequences_test, v_test, j_test],
                'seq': test_data,
                'count': np.array(self.count)[test_idx],
                'id': self.ids[test_idx],
            }

        else:
            self.sequences_train, self.sequences_test, train_idx, test_idx = train_test_split(
                self.sequences, range(len(self.sequences)), test_size=test_size, random_state=42)
                
            # Map count using test indices
            self.data_test = {
                'seq': np.array(self.sequences_test),
                'count': np.array(self.count)[test_idx],
                'id': self.ids[test_idx]
            }
            
        # Log info about the split
            logging.info(f"Train size: {len(self.sequences_train)}")
            logging.info(f"Test size: {len(self.sequences_test)}")
            

    def train_word2vec(self, epochs=10, batch_size=100, learning_rate=1e-4):
        logging.info("Training Word2Vec model...")
        batch_size = min(batch_size, len(self.sequences_train))
        self.aa_emb = word2vec(path=self.sequences_train, epochs=epochs,
                               batch_size=batch_size,
                               device=self.device, lr=learning_rate,
                               window_size=3,
                               record_path=f'{self.embeddings_dir}/word2vec_aa/{self.input_name}_aa.txt')
        logging.info("Word2Vec model trained successfully.")

    def train_model(self, embedding_file, hidden_size=128, num_layers=5, epochs=20, batch_size=100, learning_rate=1e-4, vj=False, load=False, path=None):
        logging.info("Training TCRpeg model...")        
        if embedding_file is None:
            self.embedding_file = f'{self.embeddings_dir}/word2vec_aa/{self.input_name}_aa.txt'
        else:
            self.embedding_file = embedding_file
        
        logging.info(f"Embedding file : {self.embedding_file}")

        if self.vj:
            # Update the V and J lists dynamically based on the dataset
            unique_vs = list(set(self.v_genes))
            unique_js = list(set(self.j_genes))

            # Calculate the batch size
            batch_size = min(batch_size, len(self.sequences_train))
            
            self.model = TCRpeg(hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            load_data=True, 
                            max_length=50,
                            embedding_path=self.embedding_file,
                            path_train=self.sequences_train_vj,  # Use VJ-specific data structure
                            device=self.device,
                            vs_list=unique_vs,
                            js_list=unique_js,
                            vj=True)
            self.model.create_model(vj=True)
            self.model.train_tcrpeg_vj(epochs=epochs, batch_size=batch_size, lr=learning_rate)
        else:
            self.model = TCRpeg(hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            load_data=True, 
                            max_length=50,
                            embedding_path=self.embedding_file,
                            path_train=self.sequences_train,
                            device=self.device)
            self.model.create_model()
            self.model.train_tcrpeg(epochs=epochs, batch_size=batch_size, lr=learning_rate)
            

        #                     embedding_path='tcrpeg/data/embedding_32.txt', vj=vj,
        #                     #fix testing with the provided word2vec

        # Save the model
        self.model.save(f'{self.models_dir}/{self.input_name}.pth')
        logging.info("TCRpeg model trained successfully.")

    def probability_inference(self, min_count=1):
        logging.info("Performing probability inference...")

        eva = evaluation(model=self.model, vj_model=self.vj)

        r, p_data, p_infer, p_infer_annotated = eva.eva_prob(path=self.data_test, min_occurrence=min_count)

        logging.info(f"Pearson correlation coefficient are : {r}")

        np.save(f'{self.p_infer_dir}/raw/{self.input_name}_p_infer.npy', p_infer)    

        logging.info("Probability inference completed successfully.")

        if self.vj:   
            # Map IDs using all three components
            p_infer_annotated_ids = list(map(lambda x: (
                self.id_seq_dict.get((x[0][0], x[0][1], x[0][2]), None), 
                x[0][0], x[0][1], x[0][2], x[1]
            ), p_infer_annotated))
            
            # Create a structured array with sequence, v, j and p_infer
            structured_array = np.array(p_infer_annotated_ids, 
                                    dtype=[('id', 'U50'), 
                                            ('sequence', 'U100'),
                                            ('v', 'U50'),
                                            ('j', 'U50'),
                                            ('p_infer', 'f8')])

        else:
            p_infer_annotated_ids = list(map(lambda x: (self.id_seq_dict.get(x[0], None), x[0], x[1]), p_infer_annotated))

            # Create a structured array with sequence and p_infer
            structured_array = np.array(p_infer_annotated_ids, dtype=[('id', 'U50'), ('sequence', 'U100'), ('p_infer', 'f8')])

        np.save(f'{self.p_infer_dir}/structured/{self.input_name}_structured_p_infer.npy', structured_array)

    def calculate_embeddings(self):
        logging.info("Calculating embeddings...")
        # Calculate embeddings for each sequence
        embeddings = self.model.get_embedding(self.sequences)
        reduced_embeddings = np.mean(embeddings, axis=1)
        np.save(f'{self.embeddings_dir}/raw/{self.input_name}_raw_embeddings.npy', embeddings)        

        #Create a structured array with sequence, id and p_infer
        structured_array = np.zeros(len(self.sequences),
                                    dtype=[('id', 'U50'), ('sequence', 'U100'), ('embedding', 'f4', (640,))])
        
        structured_array['id'] = self.ids
        structured_array['sequence'] = self.sequences
        structured_array['embedding'] = embeddings
        # Save the structured array
        np.save(f'{self.embeddings_dir}/structured/{self.input_name}_structured_embeddings.npy', structured_array)
        logging.info("Embeddings calculated successfully.")

    def run(self, seq_col='sequence', id_col='id', count_col='count', test_size=0.2, embedding_file = None,
            word2vec_epochs=10, word2vec_batch_size=100, word2vec_learning_rate=1e-4,
            hidden_size=128, num_layers=5, epochs=20, batch_size=100, learning_rate=1e-4,
            min_count=1, vj=False):
        self.prepare_directories_and_filenames()
        self.load_and_preprocess_data(seq_col=seq_col, id_col=id_col, count_col=count_col, vj=vj)
        self.split_data(test_size=test_size)
        # self.train_word2vec(epochs=word2vec_epochs, batch_size=word2vec_batch_size,
                            # learning_rate=word2vec_learning_rate)
        self.train_word2vec(epochs=word2vec_epochs, batch_size=word2vec_batch_size,
                            learning_rate=word2vec_learning_rate)
        self.train_model(embedding_file=embedding_file, hidden_size=hidden_size, num_layers=num_layers, epochs=epochs, batch_size=batch_size,
                        learning_rate=learning_rate, vj=vj)
        self.probability_inference(min_count=min_count)
        # self.calculate_embeddings()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TCRpeg classification model.')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Directory to save embeddings, models and p-infer',
                        required=True)
    parser.add_argument('-e', '--embedding', help='Embedding file to use for model training', default=None),
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
    parser.add_argument('--min_count', help='Minimum count for probability inference (default: 1)',
                        type=float, default=1)
    parser.add_argument('--vj', help='Use VJ for probability inference (default: False)',
                        action='store_true')
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

    #todo fix problem with mps and float
    # Check if MPS is available and set the device accordingly
    if args.device == 'mps' and not torch.backends.mps.is_available():
        logging.warning("MPS device requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    model_infer = TCRpegModel(input_file=args.input, output_dir=args.output,  device=args.device)
    model_infer.run(seq_col=args.seq_col, id_col=args.id, count_col=args.count_col, embedding_file = args.embedding,
                    test_size=args.test_size, word2vec_epochs=args.word2vec_epochs,
                    word2vec_batch_size=args.word2vec_batch_size, word2vec_learning_rate=args.word2vec_learning_rate,
                    hidden_size=args.hidden_size, num_layers=args.num_layers, epochs=args.epochs,
                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                    min_count=args.min_count, vj=args.vj)
    
    logging.info("Model training and inference completed successfully.")
