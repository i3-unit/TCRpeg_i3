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
from tcrpeg_toolkit.utils import load_data


# Add the parent directory to the sys.path
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'TCRpeg_i3')))

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

def identify_signature_sequences(structured_arrays_dict, reference_sample):
    """
    Identify signature sequences for a reference sample based on maximum probability inference.
    
    Parameters:
    - structured_arrays_dict (dict): Dictionary with sample names as keys and structured arrays as values.
    - reference_sample (str): The sample name to identify signatures for.
    
    Returns:
    - np.array: An array of signature sequences for the reference sample.
    """
    # Gather all sequences and their pinfers across samples
    sequence_data = {}
    for sample, data in structured_arrays_dict.items():
        for row in data:
            seq_id, sequence, pinfer = row
            if sequence not in sequence_data:
                sequence_data[sequence] = {}
            sequence_data[sequence][sample] = pinfer

    # Determine the maximum pinfer for each sequence across samples
    max_pinfer_by_seq = {
        seq: max(details.values()) for seq, details in sequence_data.items()
    }

    # Identify sequences where the reference sample has a pinfer greater than any other sample
    signature_sequences = []
    for row in structured_arrays_dict[reference_sample]:
        seq_id, sequence, pinfer = row
        # Check if the pinfer for this sequence in the reference sample is the highest
        if pinfer == max_pinfer_by_seq[sequence] and all(pinfer >= other_pinfer for other_pinfer in sequence_data[sequence].values()):
            signature_sequences.append((seq_id, sequence, pinfer))

    return np.array(signature_sequences, dtype=[('id', '<U50'), ('sequence', '<U100'), ('pinfer', '<f4')])


def identify_signature_sequences_and_update(structured_arrays_dict, reference_sample):
    # Concatenate all data into a single DataFrame
    frames = []
    for sample, data in structured_arrays_dict.items():
        df = pd.DataFrame(data)
        df['sample'] = sample
        frames.append(df)
    all_data = pd.concat(frames)

    # Compute the max pinfer across samples for each sequence
    max_pinfer = all_data.groupby('sequence')['pinfer'].max().rename('max_pinfer')

    # Merge max pinfer back to the original data
    all_data = all_data.merge(max_pinfer, left_on='sequence', right_index=True)

    # Add 'signature' column, True if the sequence's pinfer equals the max_pinfer and is from the reference sample
    all_data['signature'] = (all_data['sample'] == reference_sample) & (all_data['pinfer'] == all_data['max_pinfer'])

    # Filter to find signatures
    signatures = all_data[(all_data['sample'] == reference_sample) & (all_data['pinfer'] == all_data['max_pinfer'])]

    return all_data[['sample', 'id', 'sequence', 'pinfer', 'signature']], signatures[['id', 'sequence', 'pinfer']].to_records(index=False)



class PinferCalculation:
    def __init__(self, model_file, embedding_file, sequence_file, output_dir, device='cpu'):
        self.model_file = model_file
        self.embedding_file = embedding_file
        self.input_file = sequence_file
        self.output_dir = output_dir
        self.device = device
        self.model = None
        self.data = None
        #improve self.vj defined here
        
    def prepare_directories_and_filenames(self):
        # Extract the model file name without extension
        self.model_name = os.path.basename(self.model_file).split('.pth')[0]
        
        self.data_name = os.path.basename(self.input_file).split('.csv')[0]
        
        # Create the analysis output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        self.analysis_dir = os.path.join(self.output_dir, "model_evaluation/", self.model_name)

        os.makedirs(self.analysis_dir, exist_ok=True)

    #improve all classes option without output to return and use directly

    def load_and_preprocess_data(self, seq_col='sequence', id_col='id', count_col='count', v_col='v', j_col='j', vj=False):
        self.data = load_data(self.input_file)
        self.sequences = self.data[seq_col].values
        # Lower case all columns names
        self.data.columns = map(str.lower, self.data.columns)

        # Check if V and J genes are present in the data
        self.vj = vj
        if self.vj:
            self.v_genes = self.data[v_col.lower()].values
            self.j_genes = self.data[j_col.lower()].values
            self.sequences_vj = [list(item) for item in zip(self.sequences, self.v_genes, self.j_genes)]
        
        # Check if id and count are present in the data
        self.ids = self.data[id_col].values if id_col in self.data.columns else np.arange(len(self.data))
        self.counts = self.data[count_col].values if count_col in self.data.columns else np.arange(len(self.data))

        # Create dictionary using original sequences
        if self.vj:
            # Create dictionary using sequence, v, j as key
            self.id_seq_dict = dict(zip(zip(self.sequences, self.v_genes, self.j_genes), self.ids))
            self.data_dict = {
                # Seq needs to be like: [sequence, v, j]
                'seq': [list(item) for item in zip(self.sequences, self.v_genes, self.j_genes)],
                'count': np.array(self.counts),
                'id': np.array(self.ids)
            }
            self.unique_vs = list(set(self.v_genes))
            self.unique_js = list(set(self.j_genes))
        else:
            self.id_seq_dict = dict(zip(self.sequences, self.ids))

            self.data_dict = {
                'seq': np.array(self.sequences),
                'count': np.array(self.counts),
                'id': np.array(self.ids)
            }
            
    def load_model(self, hidden_size=128, num_layers=5):
        # Create an instance of TCRpeg and load the model
        if self.vj:
            self.model = TCRpeg(hidden_size=hidden_size,
                                num_layers=num_layers,
                                embedding_path=self.embedding_file,
                                device=self.device,
                                vj=True,
                                vs_list=self.unique_vs,
                                js_list=self.unique_js)
        else:
            self.model = TCRpeg(hidden_size=hidden_size,
                                num_layers=num_layers,
                                embedding_path=self.embedding_file,
                                device=self.device)
    
        self.model.create_model(load=True, path=self.model_file, vj=self.vj)
        logging.info(f"Model loaded successfully with VJ: {self.vj}")
      
    def calculate_pinfer(self, sample_name=None):
        eva = evaluation(model=self.model, vj_model=self.vj)
        p_infer = eva.eva_prob(path=self.data_dict)[2]
        logging.info("Probability inference calculated successfully.")

        if self.vj:
            structured_array = np.zeros(len(self.data_dict['seq']),
                                        dtype=[('id', 'U50'), 
                                               ('sequence', 'U100'),
                                               ('v', 'U50'),
                                               ('j', 'U50'),
                                               ('pinfer', 'f4')])
            structured_array['id'] = self.data_dict['id']

            # Unpack the data
            sequences, v_genes, j_genes = zip(*self.data_dict['seq'])
            structured_array['sequence'] = sequences
            structured_array['v'] = v_genes
            structured_array['j'] = j_genes
            
            structured_array['pinfer'] = p_infer
        else:
            #Create a structured array with sequence, id and p_infer
            structured_array = np.zeros(len(self.data_dict['seq']),
                                        dtype=[('id', 'U50'), ('sequence', 'U100'), ('pinfer', 'f4')])
            
            structured_array['id'] = self.data_dict['id']
            structured_array['sequence'] = self.data_dict['seq']
            structured_array['pinfer'] = p_infer

        #improve structure in raw and embeddings
        # Save the structured array
        np.save(f'{self.analysis_dir}/{self.data_name}_structured_pinfer.npy', structured_array)
        logging.info(f"Probability inference saved successfully.")
        return structured_array

        #improve can use third slot of eva return for a seq, prob dictionnary -> will avoid problem if not calculated for one

    def run(self, seq_col='sequence',
            id_col='id', count_col='count',
            v_col='v', j_col='j',
            hidden_size=128,  num_layers=5, vj=False, sample_name=None):
        self.prepare_directories_and_filenames()
        self.load_and_preprocess_data(seq_col=seq_col,
                                      id_col=id_col,
                                      count_col=count_col,
                                      v_col=v_col,
                                      j_col=j_col,
                                      vj=vj)
        self.load_model(hidden_size=hidden_size, num_layers=num_layers)
        structured_pinfer = self.calculate_pinfer(sample_name=sample_name)
        return structured_pinfer

    @staticmethod
    def run_for_folder(data_dir, model_dir, output_dir, default_embedding_file=None, embedding_dir=None, vj=False, signature=True, device='cpu'):
        # List all data files in the data directory
        data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')])

        # Create a dictionary with sample names as keys and paths to model and data files as values
        sample_info = {
            os.path.basename(data_file).split('.csv')[0]: {
                'model': os.path.join(model_dir, f"{os.path.basename(data_file).split('.csv')[0]}.pth"),
                'data': data_file
            }
            for data_file in data_files
            if os.path.exists(os.path.join(model_dir, f"{os.path.basename(data_file).split('.csv')[0]}.pth"))
        }
        
        # Probability inference results
        p_infer_results = {}
        
        # Iterate over the dictionary and pass each entry to PinferCalculation
        for sample_name, paths in sample_info.items():
            model_file = paths['model']
            data_file = paths['data']
            
            # Determine the embedding file
            if default_embedding_file is not None:
                embedding_file = default_embedding_file
            elif embedding_dir is not None:
                embedding_file = os.path.join(embedding_dir, f'{model_name}_embedding.txt')
            else:
                logging.error("Please provide either a default embedding file for word2vec model or a specific one for the data.")

            # Create and run the PinferCalculation instance
            pinfer_calc = PinferCalculation(
                model_file=model_file,
                embedding_file=embedding_file,
                sequence_file=data_file,
                output_dir=output_dir,
                device=device
            )
            # Store the structured array containg p_infer, seq, id, v and j gene
            p_infer_model = pinfer_calc.run(vj=vj)

            # Store the structured array in the dictionary
            p_infer_results[sample_name] = p_infer_model

        if signature:
            signatures_df_and_array = [identify_signature_sequences_and_update(p_infer_results, x) for x in sample_info.keys()]
            return p_infer_results, signatures_df_and_array

            # Create a new dataframe with sample_name, sequence, p_infer, id, v and j genes
        return p_infer_results
            

    
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