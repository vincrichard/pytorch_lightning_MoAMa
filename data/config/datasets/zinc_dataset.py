from src.featurizer import MotifMaskAtom
from src.featurizer.motif_extractor import MoaMAMotifExtractor

from src.data.datasets.zinc import Zinc

dataset_path = "dataset/zinc_standard_agent/processed/smiles.csv"

motif_extractor = dict(type=MoaMAMotifExtractor, motif_depth=5)

# Note: we are loading the smiles in pretrained and adding the smiles_list
# param dynamically.
masking_strategy = dict(
    type=MotifMaskAtom,
    num_atom_type=119,
    mask_rate=0.15,
    motif_extractor=motif_extractor,
)

zinc_dataset = dict(type=Zinc, dataset_path=dataset_path, masking_strategy=masking_strategy)
