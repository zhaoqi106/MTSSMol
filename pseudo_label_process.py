import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import MACCSkeys


class MultiGranularityPseudoLabeler:
    def __init__(self, k_values=[100, 1000, 10000]):
        """
        Initialize with different granularity levels (k_values for K-means)

        Args:
            k_values (list): List of cluster numbers for different granularities
        """
        self.k_values = sorted(k_values)  # Sort from coarse to fine
        self.cluster_models = {}
        self.fingerprint_size = 166  # MACCS fingerprints are 166 bits

    def compute_fingerprints(self, smiles_list):
        """
        Compute MACCS fingerprints from SMILES strings

        Args:
            smiles_list (list): List of SMILES strings

        Returns:
            np.array: Numpy array of fingerprints (n_molecules x 166)
        """
        fingerprints = []
        valid_indices = []  # To keep track of valid SMILES
        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp = MACCSkeys.GenMACCSKeys(mol)
            fingerprints.append(np.array(fp))
            valid_indices.append(idx)

        return np.array(fingerprints), valid_indices

    def fit_clusters(self, fingerprints):
        """
        Fit K-means models at different granularities

        Args:
            fingerprints (np.array): Molecular fingerprints
        """
        for k in self.k_values:
            print(f"Clustering with k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(fingerprints)
            self.cluster_models[k] = kmeans

    def assign_pseudo_labels(self, fingerprints):
        """
        Assign multi-granularity pseudo-labels to molecules

        Args:
            fingerprints (np.array): Molecular fingerprints

        Returns:
            dict: Dictionary containing pseudo-labels at each granularity
        """
        pseudo_labels = {}
        for k in self.k_values:
            labels = self.cluster_models[k].predict(fingerprints)
            pseudo_labels[k] = labels

        return pseudo_labels


def process_csv(input_csv, smiles_col='Smiles', output_csv=None):
    """
    Process a CSV file containing SMILES strings and add pseudo-labels

    Args:
        input_csv (str): Path to input CSV file
        smiles_col (str): Name of column containing SMILES strings
        output_csv (str): Path to output CSV file (None to overwrite input)
    """
    # Read input CSV
    df = pd.read_csv(input_csv)

    # Initialize processor
    processor = MultiGranularityPseudoLabeler()

    # Compute fingerprints
    smiles_list = df[smiles_col].tolist()
    fingerprints, valid_indices = processor.compute_fingerprints(smiles_list)

    # Fit clusters
    processor.fit_clusters(fingerprints)

    # Assign pseudo-labels
    pseudo_labels = processor.assign_pseudo_labels(fingerprints)

    # Initialize new columns with NaN
    for k in processor.k_values:
        df[f'k={k}'] = np.nan

    # Fill in labels for valid SMILES
    for idx, valid_idx in enumerate(valid_indices):
        for k in processor.k_values:
            df.at[valid_idx, f'k={k}'] = pseudo_labels[k][idx]

    # Save results
    output_path = output_csv if output_csv else input_csv
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"Added columns: {', '.join([f'k={k}' for k in processor.k_values])}")


# Example usage
if __name__ == "__main__":
    # Configure these parameters
    input_file = "data10m.csv"  # Replace with your input file
    smiles_column = "smiles"  # Column name containing SMILES
    output_file = None  # Set to None to overwrite input, or specify new path

    # Run the processing
    process_csv(input_file, smiles_col=smiles_column, output_csv=output_file)