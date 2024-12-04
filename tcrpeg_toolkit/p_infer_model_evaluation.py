import os
import sys
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
from tcrpeg_toolkit.utils import apply_grouping_and_filtering

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
                            # Ensuring encoding is set
                            logging.FileHandler(log_file, encoding='utf-8'),
                            logging.StreamHandler()
                        ])


np.random.seed(222)


def old_identify_signature_sequences(structured_arrays_dict, reference_sample):
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


def identify_signature_sequences(structured_arrays_dict, reference_sample, groupby_vj=True):
    # Concatenate all data into a single DataFrame, considering optional v and j gene information
    frames = []
    for sample, data in structured_arrays_dict.items():
        df = pd.DataFrame(data)
        df['sample'] = sample
        frames.append(df)
    all_data = pd.concat(frames)

    # Define the grouping columns based on the available data
    group_columns = ['sequence']
    if 'v' in all_data.columns and 'j' in all_data.columns and groupby_vj:
        group_columns += ['v', 'j']

    # Compute the max pinfer across samples for each sequence combination
    max_pinfer = all_data.groupby(group_columns)[
        'pinfer'].max().rename('max_pinfer')

    # Merge max pinfer back to the original data
    all_data = all_data.merge(max_pinfer, on=group_columns)

    # Add 'signature' column, True if the sequence's pinfer equals the max_pinfer and is from the reference sample
    all_data['signature'] = (all_data['sample'] == reference_sample) & (
        all_data['pinfer'] == all_data['max_pinfer'])

    # Filter to find signatures, ensuring to select additional gene information if present
    signature_columns = ['sample', 'id', 'sequence', 'pinfer',
                         'signature'] + group_columns[1:]  # Avoid duplicating sequence

    # Return the full data and a structured array of signatures
    signatures = all_data[(all_data['sample'] ==
                           reference_sample) & all_data['signature']]

    # Convert to records then change dtype
    signatures_records = signatures[[
        'id', 'sequence', 'pinfer'] + group_columns[1:]].to_records(index=False)
    signatures_records = signatures_records.astype(
        [('id', 'U50'), ('sequence', 'U100'), ('pinfer', 'f4')] + [(col, 'U50') for col in group_columns[1:]])

    return all_data[signature_columns], signatures_records


def calculate_signatures(p_infer_results, sample_info, output_dir=None):
    """
    Calculate signature sequences for the provided samples.

    Args:
        p_infer_results (dict): Dictionary with sample names as keys and p_infer results as values.
        sample_info (dict): Dictionary with sample names as keys and data paths as values.
        output_dir (str): Directory to save the signature results.

    Returns:
        tuple: A tuple containing:
            - sample_signature_non_signature_df (pd.DataFrame): DataFrame with all sequences and their signature status.
            - sample_signature_array (np.ndarray): Array of signature sequences.
    """
    logging.info(f"Calculating signature sequences...")

    # Ensure output directories for signatures
    if output_dir is not None:
        os.makedirs(os.path.join(output_dir, 'signatures'), exist_ok=True)
        os.makedirs(os.path.join(
            output_dir, 'sequences_analysis'), exist_ok=True)

    # Iterate over samples and calculate signatures
    for sample in sample_info.keys():
        sample_signature_non_signature_df, sample_signature_array = identify_signature_sequences(
            p_infer_results, sample
        )
        if output_dir is not None:
            # Save the results
            sample_signature_non_signature_df.to_csv(
                os.path.join(output_dir, 'sequences_analysis',
                             f'{sample}_all_sequences_with_signatures.csv'),
                index=False
            )
            np.save(
                os.path.join(output_dir, 'signatures',
                             f'{sample}_signature.npy'),
                sample_signature_array
            )

    logging.info(
        f"Signature calculation completed for {len(sample_info.keys())} samples.")
    return sample_signature_non_signature_df, sample_signature_array


class PinferCalculation:
    def __init__(self, model_file, embedding_file, sequence_file, output_dir=None, device='cpu'):
        self.model_file = model_file
        self.embedding_file = embedding_file
        self.input_file = sequence_file
        self.output_dir = output_dir
        self.device = device
        self.model = None
        self.data = None
        self.analysis_dir = None
        # improve self.vj defined here

    def prepare_directories_and_filenames(self):
        # Extract the model file name without extension
        self.model_name = os.path.basename(self.model_file).split('.pth')[0]

        self.data_name = os.path.basename(self.input_file).split('.csv')[0]

        # Create the analysis output directory if it doesn't exist
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

            self.analysis_dir = os.path.join(
                self.output_dir, "model_evaluation/", self.model_name)

            os.makedirs(self.analysis_dir, exist_ok=True)

    # improve all classes option without output to return and use directly

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
            self.sequences_vj = [list(item) for item in zip(
                self.sequences, self.v_genes, self.j_genes)]

        # Check if id and count are present in the data
        self.ids = self.data[id_col].values if id_col in self.data.columns else np.arange(
            len(self.data))
        self.counts = self.data[count_col].values if count_col in self.data.columns else np.arange(
            len(self.data))

        # Create dictionary using original sequences
        if self.vj:
            # Create dictionary using sequence, v, j as key
            self.id_seq_dict = dict(
                zip(zip(self.sequences, self.v_genes, self.j_genes), self.ids))
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
            # Create a structured array with sequence, id and p_infer
            structured_array = np.zeros(len(self.data_dict['seq']),
                                        dtype=[('id', 'U50'), ('sequence', 'U100'), ('pinfer', 'f4')])

            structured_array['id'] = self.data_dict['id']
            structured_array['sequence'] = self.data_dict['seq']
            structured_array['pinfer'] = p_infer

        # improve structure in raw and embeddings
        # Save the structured array
        if self.analysis_dir is not None:
            np.save(
                f'{self.analysis_dir}/{self.data_name}_structured_pinfer.npy', structured_array)
            logging.info(f"Probability inference saved successfully.")
        return structured_array

        # improve can use third slot of eva return for a seq, prob dictionnary -> will avoid problem if not calculated for one

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
# fix needs to compare for one parameters (same cell subset, same tissue)

    @staticmethod
    def run_for_folder(data_dir, model_dir, output_dir=None,
                       default_embedding_file=None, embedding_dir=None,
                       seq_col='sequence', vj=False, groupby=None,
                       filter_values=None, filters=None, metadata=None,
                       key_metadata='id', device='cpu', signature=True):
        """
        Run p_infer calculation for multiple files with optional metadata grouping, filtering, and signature calculation.

        Args:
            data_dir (str): Directory containing data files (.csv).
            model_dir (str): Directory containing model files (.pth).
            output_dir (str, optional): Directory to save results.
            default_embedding_file (str, optional): Path to a default embedding file.
            embedding_dir (str, optional): Directory containing embedding files.
            seq_col (str, optional): Name of the sequence column in data. Default: 'sequence'.
            vj (bool, optional): Whether to include V and J gene information.
            groupby (str or list, optional): Column(s) in metadata to group by.
            filter_values (dict, optional): Filters for metadata. Keys are column names, 
                                                and values are lists of allowed values.
            filters (dict, optional): Combined argument (groupby and filter_values) where keys are columns and values:
                                    - Value to filter by.
                                    - None to indicate grouping.                                    
            metadata (str, optional): Path to a metadata file.
            key_metadata (str, optional): Key column in metadata for sample matching. Default: 'id'.
            device (str, optional): Device to use ('cpu', 'cuda:0', etc.). Default: 'cpu'.
            signature (bool, optional): Whether to calculate signature sequences. Default: False.

        Returns:
            dict: Dictionary with sample names as keys and p_infer results as values.
        """
        import os
        import logging
        import numpy as np

        # Ensure output directories exist
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # List all data files
        data_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')
        ])

        # Create sample info dictionary
        sample_info = {
            os.path.basename(data_file).split('.csv')[0]: {
                'model': os.path.join(model_dir, f"{os.path.basename(data_file).split('.csv')[0]}.pth"),
                'data': data_file
            }
            for data_file in data_files
            if os.path.exists(os.path.join(model_dir, f"{os.path.basename(data_file).split('.csv')[0]}.pth"))
        }

        # Initialize results dictionary
        p_infer_results = {}

        if metadata:
            metadata = load_data(metadata)
            grouped_metadata = apply_grouping_and_filtering(
                metadata, groupby=groupby, filter_values=filter_values, filters=filters)
        else:
            # No metadata: Treat all samples as one group
            grouped_metadata = [('all', None)]

        # Process each group
        for group_name, group_df in grouped_metadata.items():
            logging.info(f"Processing group: {group_name}")

            # Filter sample_info based on group metadata if applicable
            if group_df is not None:
                group_ids = group_df[key_metadata].values
                filtered_sample_info = {
                    k: v for k, v in sample_info.items() if k in group_ids}
            else:
                filtered_sample_info = sample_info

            # Process each sample in the group
            for sample_name, paths in filtered_sample_info.items():
                model_file = paths['model']
                data_file = paths['data']

                # Determine the embedding file
                if default_embedding_file:
                    embedding_file = default_embedding_file
                elif embedding_dir:
                    embedding_file = os.path.join(
                        embedding_dir, f'{sample_name}_embedding.txt')
                    if not os.path.exists(embedding_file):
                        logging.warning(
                            f"Embedding file not found for {sample_name}. Skipping.")
                        continue
                else:
                    raise ValueError(
                        "Embedding file must be provided either as default or in embedding_dir.")

                # Run PinferCalculation for the sample
                pinfer_calc = PinferCalculation(
                    model_file=model_file,
                    embedding_file=embedding_file,
                    sequence_file=data_file,
                    output_dir=output_dir,
                    device=device,
                )
                p_infer_results[sample_name] = pinfer_calc.run(
                    vj=vj, seq_col=seq_col)

        # Optional: Calculate signature sequences for each group
        if signature:
            signature_results = {}
            sequences_annotated_df, signature_array = calculate_signatures(
                p_infer_results, sample_info, output_dir)
            return sequences_annotated_df, signature_array

        return p_infer_results

# improve create base class for any comparison, add snakemake


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='TCRpeg classification model.')
    parser.add_argument('-m', '--model', help='Model file', required=True)
    parser.add_argument('-s', '--sequences',
                        help='Sequence file', required=True)
    parser.add_argument('-e', '--embeddings',
                        help='Embedding file', required=True)
    parser.add_argument(
        '-o', '--output', help='Directory to save embeddings, models and p-infer', required=True)
    parser.add_argument(
        '--hidden_size', help='Hidden size in the trained model (default: 128, check log file for verification)', type=int, default=128)
    parser.add_argument(
        '--num_layers', help='Number of layers in the trained model (default: 5, check log file for verification)', type=int, default=5)
    parser.add_argument('-d', '--device', help='Device to use (cpu, cuda:0, mps)',
                        default='cpu', choices=["cpu", "cuda:0", "mps"])
    parser.add_argument('--vj', help='Use V and J genes', action='store_true')
    parser.add_argument(
        '--signature', help='Calculate signature sequences', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Check if MPS is available and set the device accordingly
    if args.device == 'mps' and not torch.backends.mps.is_available():
        logging.warning(
            "MPS device requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    # Configure logging
    configure_logging(os.path.join(args.output, 'model_evaluation.log'))

    # Run the PinferCalculation
    pinfer_calc = PinferCalculation(
        model_file=args.model,
        embedding_file=args.embeddings,
        sequence_file=args.sequences,
        output_dir=args.output,
        device=args.device
    )
    structured_pinfer = pinfer_calc.run(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        vj=args.vj,
    )

    # if args.signature:
    #     sample_name = os.path.basename(args.sequences).split('.csv')[0]
    #     _, sample_signature_array = identify_signature_sequences_and_update({sample_name: structured_pinfer}, sample_name)
    #     np.save(os.path.join(args.output, 'signatures', f'{sample_name}_signature.npy'), sample_signature_array)

    logging.info("Inference completed successfully.")
