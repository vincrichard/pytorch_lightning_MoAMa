from typing import List

from torch.utils.data import Dataset, Subset
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold


def scaffold_split(
    dataset: Dataset,
    smiles_list: List[str],
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split

    :param dataset: A torch Dataset
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param frac_train:
    :param frac_valid:
    :param frac_test:

    :return: Subset of the original dataset with the correct splits
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for scaffold_set in sorted(all_scaffolds.values(), key=lambda x: (len(x), x[0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = Subset(dataset, indices=train_idx)
    valid_dataset = Subset(dataset, indices=valid_idx)
    test_dataset = Subset(dataset, indices=test_idx)

    return train_dataset, valid_dataset, test_dataset
