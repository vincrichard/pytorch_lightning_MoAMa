from typing import List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from rdkit import Chem
from rdkit.Chem import BRICS


@dataclass
class Motif:
    atom_ids: List[int]
    neighbors_id: List[int]
    size_largest_parth: int


class AbstractMotifExtractor(ABC):
    def __init__(self, motif_depth: int = 5):
        self.motif_depth = motif_depth

    @abstractmethod
    def get_motifs(self, mol: Chem.Mol) -> List[Motif]:
        pass


class BricsRingMotifExtractor(AbstractMotifExtractor):
    """
    Extract BRICS motifs as well as rings. If the motif is too large > 5 hop
    from the nearest neighbor, it is filter out
    """

    def get_motifs(self, mol: Chem.Mol) -> List[Motif]:
        motifs = self.get_brics_motifs(mol)
        motifs = self.add_ring_motif_and_delete_dupplicates(mol, motifs)
        return self.filter_large_motifs(mol, motifs, self.motif_depth)

    def get_brics_motifs(self, mol: Chem.Mol):
        fragmented = BRICS.BreakBRICSBonds(mol)
        pieces = Chem.GetMolFrags(fragmented, asMols=True)
        motifs = []
        for piece in pieces:
            motifs.extend(mol.GetSubstructMatches(self.remove_dummy_atoms(piece)))
        return list(set(motifs))

    def add_ring_motif_and_delete_dupplicates(self, mol: Chem.Mol, motifs: List[List[int]]) -> List[List[int]]:
        """Add all single ring as motif, some BRICS motif which are subset of ring are deleted."""
        ring_groups = mol.GetRingInfo().AtomRings()
        motifs = [motif for motif in motifs if self.to_keep(motif, ring_groups)]
        motifs.extend(ring_groups)
        return motifs

    def to_keep(self, atom_group, ring_groups):
        is_ring_subset = not all([len(set(atom_group) - set(ring)) for ring in ring_groups])
        is_single_atom = len(atom_group) == 1
        return not (is_single_atom or is_ring_subset)

    def remove_dummy_atoms(self, mol: Chem.Mol) -> Chem.Mol:
        mol = Chem.rdchem.RWMol(mol)
        atoms = list(mol.GetAtoms())
        for atom in atoms:
            if atom.GetAtomicNum() == 0:
                mol.RemoveAtom(atom.GetIdx())
        return Chem.Mol(mol)

    def filter_large_motifs(self, mol: Chem.Mol, motifs: List[List[int]], k_hop: int = 5) -> List[Motif]:
        """Each node within the motif must be within a k-hop neighborhood
        (k equals number of GNN layers) of an inter-motif node
        """
        valid_motifs = []
        for motif in motifs:
            if len(motif) > k_hop:
                motif_neighbors_ids = self.get_motif_neigboring_atoms(mol, motif)
                size_largest_path = self.get_largest_path(mol, motif, motif_neighbors_ids)
                if size_largest_path <= k_hop:
                    valid_motifs.append(Motif(motif, motif_neighbors_ids, size_largest_path))
            else:
                valid_motifs.append(Motif(motif, [], -1))

        return valid_motifs

    def get_shortest_path(self, mol: Chem.Mol, start_atom: int, target_atoms: int) -> int:
        """Get size of shortest path between start_atom and target_atoms, correspond
        to the number of edge between a atom outside the motif and an atom in the motif.
        """
        visited = [False] * mol.GetNumAtoms()
        queue = [(start_atom, [start_atom])]
        visited[start_atom] = True

        while queue:
            current_atom, path = queue.pop()

            if current_atom in target_atoms:
                return len(path) - 1

            neighbors = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(current_atom).GetNeighbors()]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, path + [neighbor]))

        return -1

    def get_largest_path(self, molecule: Chem.Mol, motif_atom_ids: List[int], motif_neighbors_ids: List[int]) -> bool:
        """Return the largest number of edge necessary to reach all atom from a neighbor of the motif"""
        motif_largest_path = 0
        for motif_atom in motif_atom_ids:

            atom_largest_path = 1e5
            for neighbor_atom in motif_neighbors_ids:
                shortest_path = self.get_shortest_path(molecule, motif_atom, [neighbor_atom])
                atom_largest_path = min(atom_largest_path, shortest_path)

            motif_largest_path = max(motif_largest_path, atom_largest_path)

        return motif_largest_path

    def get_motif_neigboring_atoms(self, mol: Chem.Mol, motif: List[int]) -> List[int]:
        motif_neigbors = []
        for atom_id in motif:
            atom = mol.GetAtomWithIdx(atom_id)
            neighbors = set([nb.GetIdx() for nb in atom.GetNeighbors()]) - set(motif)
            if len(neighbors):
                # Only one atom is enough
                motif_neigbors.append(list(neighbors)[0])

        return motif_neigbors


class MoaMAMotifExtractor(AbstractMotifExtractor):

    def get_motifs(self, mol: Chem.Mol) -> List[Motif]:
        Chem.SanitizeMol(mol)
        motifs = self.brics_decomp(mol)
        return self.filter_large_motif(mol, motifs)

    def brics_decomp(self, mol: Chem.Mol) -> List[List[int]]:
        """
        Not literally a BRICS decomposition. Most of the motif are removed only
        chain with no close R-group or Ring are kept intact. The rest are standalone atoms.
        """

        def break_brics_bond_from_cliques(brics_bonds: List[Tuple[int, int]], cliques: List[List[int]]) -> None:
            """In this step they remove the brics bond present in cliques.
            Note that bonds means tuple in clique and that cliques can have single atoms.
            In this case all the atom forming BRICS bond becomes single atoms
            """
            for src, dst in brics_bonds:
                if [src, dst] in cliques:
                    cliques.remove([src, dst])
                else:
                    cliques.remove([dst, src])
                cliques.append([src])
                cliques.append([dst])

        def break_bonds_connected_to_rings_from_clique(
            mol: Chem.Mol, cliques: List[List[int]], breaks: List[List[int]]
        ) -> None:
            """Break bonds between rings and non-ring atoms. Keep the atom outside the ring."""
            for c in cliques:
                if len(c) > 1:
                    if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                        cliques.remove(c)
                        cliques.append([c[1]])
                        breaks.append(c)
                    if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                        cliques.remove(c)
                        cliques.append([c[0]])
                        breaks.append(c)

        def break_bonds_outside_of_rings_from_clique(
            mol: Chem.Mol, cliques: List[List[int]], breaks: List[List[int]]
        ) -> None:
            """
            Original comment: select atoms at intersections as motif
            Break all bonds which have more than 2 neigbors outside of rings.
            Only chain bonds away from anyother R-group are kept.
            Basically most of the BRICS motif if remove outside the Ring
            see https://github.com/einae-nd/MoAMa-dev/issues/3
            """
            for atom in mol.GetAtoms():
                if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
                    cliques.append([atom.GetIdx()])
                    for nei in atom.GetNeighbors():
                        if [nei.GetIdx(), atom.GetIdx()] in cliques:
                            cliques.remove([nei.GetIdx(), atom.GetIdx()])
                            breaks.append([nei.GetIdx(), atom.GetIdx()])
                        elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                            cliques.remove([atom.GetIdx(), nei.GetIdx()])
                            breaks.append([atom.GetIdx(), nei.GetIdx()])
                        cliques.append([nei.GetIdx()])

        def merge_partial_motif_together(cliques: List[List[int]]) -> List[List[int]]:
            """Merge various motif together if they share an atom."""
            for c in range(len(cliques) - 1):
                if c >= len(cliques):
                    break
                for k in range(c + 1, len(cliques)):
                    if k >= len(cliques):
                        break
                    if len(set(cliques[c]) & set(cliques[k])) > 0:
                        cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                        cliques[k] = []
                cliques = [c for c in cliques if len(c) > 0]
            cliques = [c for c in cliques if len(c) > 0]
            return cliques

        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]]

        cliques = [[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()] for bond in mol.GetBonds()]
        breaks = []

        brics_bonds = [bond_ids for bond_ids, _ in BRICS.FindBRICSBonds(mol)]
        if len(brics_bonds) == 0:
            return [list(range(n_atoms))]
        else:
            break_brics_bond_from_cliques(brics_bonds, cliques)

        break_bonds_connected_to_rings_from_clique(mol, cliques, breaks)
        break_bonds_outside_of_rings_from_clique(mol, cliques, breaks)

        return merge_partial_motif_together(cliques)

    def filter_large_motif(self, mol: Chem.Mol, motifs: List[List[int]], k_hop: int = 5) -> List[Motif]:
        """Each node within the motif must be within a k-hop neighborhood
        (k equals number of GNN layers) of an inter-motif node"""
        valid_motifs = []
        if len(motifs) != 1:
            for motif in motifs:
                size_largest_parth = 0
                for atom in mol.GetAtoms():
                    if atom.GetIdx() in motif:
                        proximity = self.inter_motif_proximity(motif, [atom], [])
                        if proximity > k_hop:
                            break
                        else:
                            size_largest_parth = max(size_largest_parth, proximity)
                valid_motifs.append(Motif(motif, neighbors_id=[], size_largest_parth=size_largest_parth))
        return valid_motifs

    def inter_motif_proximity(self, target_motif: List[int], neighbors: List[int], checked: List[int]) -> bool:
        """Recursively check if the atom is in the motif, and find the proximity of the first atom outside the motif"""
        new_neighbors = []
        for atom in neighbors:
            for nei in atom.GetNeighbors():
                if nei.GetIdx() in checked:
                    continue
                new_neighbors.append(nei)
                if nei.GetIdx() not in target_motif:
                    return 1
            checked.append(atom.GetIdx())
        return self.inter_motif_proximity(target_motif, new_neighbors, checked) + 1
