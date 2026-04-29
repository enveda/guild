"""
Protein sequence tools
"""

import copy
import os
import random

from Bio.PDB import PDBParser

from guild.constants.general import RANDOM_SEED
from guild.constants.proteins import (
    AMINO_ACID_CONVERTER,
    AMINO_ACID_GROUPS,
    POSSIBLE_AMINO_ACIDS_THREE_LETTER,
)

random.seed(RANDOM_SEED)


def get_original_sequence_dictionary(input_file):
    """
    Use BioPython to get the original sequence of the protein. This is avoids mismatches in the residue numbering
    :param input_file: The input pdb file.
    """
    parser = PDBParser()
    structure = parser.get_structure("PHA-L", input_file)
    pdb_dictionary = {}
    for residue in structure.get_residues():
        res_name = residue.get_resname()
        res_number = residue.get_id()[1]
        chain_name = residue.get_full_id()[2]
        if res_name not in POSSIBLE_AMINO_ACIDS_THREE_LETTER:
            continue
        if chain_name not in pdb_dictionary.keys():
            pdb_dictionary[chain_name] = {}

        pdb_dictionary[chain_name][res_number] = res_name
    return pdb_dictionary


def mutate_position(target_residue, protein_dictionary):
    """
    Takes in a target residue and a protein dictionary and returns a mutated residue.
    :param target_residue: Target residue to be mutated
    :param protein_dictionary: Dictionary with the protein's original amino acids
    """
    amino_acid_groups = copy.deepcopy(AMINO_ACID_GROUPS)

    original_amino_acid_group = None
    for current_group in amino_acid_groups:
        if target_residue["res_name"] in amino_acid_groups[current_group]:
            original_amino_acid_group = current_group
    remaining_groups = list(amino_acid_groups.keys())
    remaining_groups.remove(original_amino_acid_group)

    substitution_dictionary = {}

    same_group_list = amino_acid_groups[original_amino_acid_group]
    same_group_list.remove(target_residue["res_name"])
    substitution_dictionary[original_amino_acid_group] = random.choice(same_group_list)

    for current_group in remaining_groups:
        substitution_dictionary[current_group] = random.choice(amino_acid_groups[current_group])

    mutations = {}
    for current_key in substitution_dictionary.keys():
        mirror_dictionary = copy.deepcopy(protein_dictionary)
        mirror_dictionary[target_residue["chain_name"]][target_residue["res_number"]] = (
            substitution_dictionary[current_key]
        )
        mutations[current_key] = mirror_dictionary
    return mutations


def process_into_fasta_string(input_dictionary):
    """
    Convert a dictionary of the form: {1: "MET", 2: "LYS"} to a string of the form: "MK"
    :param input_dictionary: Dictionary with the protein's original amino acids
    :return: String of the protein's original sequence
    """
    integer_keys = [int(x) for x in input_dictionary.keys()]
    sorted_keys = sorted(integer_keys)
    chain_string = ""
    for current_key in sorted_keys:
        chain_string += AMINO_ACID_CONVERTER[input_dictionary[current_key]]
    return chain_string


def dictionary_to_sequence(
    input_dictionary,
    current_mutation,
    write_ali_mode=False,
    output_folder=None,
):
    """
    Convert a dictionary to a fasta file.
    :param input_dictionary: Dictionary with the protein's original amino acids
    :param current_mutation: Current mutation to be written
    :param write_ali_mode: Whether to write the ali file
    :param output_folder: Folder to save the ali file
    """
    mutated_pdb_id = current_mutation["pdb_id"]
    mutated_chain = current_mutation["chain_name"]
    mutated_residue_number = str(current_mutation["res_number"])

    for current_amino_acid_group in input_dictionary.keys():
        ali_string = ""
        mutation_id = f"{mutated_pdb_id}_mutated_at_{mutated_chain}_{str(mutated_residue_number)}_{current_amino_acid_group}"
        for current_chain in input_dictionary[current_amino_acid_group].keys():
            chain_residues = input_dictionary[current_amino_acid_group][current_chain]
            string_aminoacids = process_into_fasta_string(chain_residues)

            ali_string += (
                f">P1;{mutation_id}\n"
                + f"sequence:{mutation_id}:{current_chain}::::::0.00: 0.00\n"
                + f"{string_aminoacids}*\n"
            )

        if write_ali_mode is True:
            if output_folder is None:
                output_folder = "mutations"
            os.makedirs(output_folder, exist_ok=True)
            with open(
                f"{output_folder}/{mutation_id}.ali",
                "w",
            ) as outfile:
                outfile.write(ali_string)
