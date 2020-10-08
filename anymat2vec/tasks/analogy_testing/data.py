import pprint
import os
import json

import tqdm
from pymatgen.core.composition import Composition, CompositionError
from mat2vec.processing import MaterialsTextProcessor

this_dir = os.path.dirname(os.path.abspath(__file__))
full_analogy_path = os.path.join(this_dir, "analogies.txt")


def load_excluded_materials_list(filename="excluded_materials_compounds_only.json"):
    """
    Loads an excluded materials list.

    Args:
        filename (str): Local filename of the json materials list within the
            analogy testing folder. E.g., excluded_materials_compounds_only.json.

    Returns:
        materials_list ([str]): List of the materials stoichiometries as strings.

    """
    with open(os.path.join(this_dir, filename), "r") as f:
        materials_list = json.load(f)
    return materials_list


def get_analogies_filename(filename):
    """
    Get the full path of an analogies file based on its name in the analogy
    testing folder. For use with gensim analogy tester.

    Args:
        filename (str): Filename of the analogies file within the analogy
            teting folder, e.g., analogies.txt

    Returns:
        str: the full path of the analogies file.

    """
    return os.path.join(this_dir, filename)


def create_restricted_analogies(
        filename_src=full_analogy_path,
        output_suffix="compounds_only",
        quiet=True
):
    """
    Rules:
    1. Only keep analogy sections relevant to compounds
    2. Remove analogies not containing a single compound (i.e., only elements)
    3. Remove analogies requiring a composition not in processed_abstracts


    Args:
        filename_src: Filename of the analogy sources.
        output_suffix (str): Suffix to use for output files.
        quiet (bool): If false, prints removed analogies to screen.

    Returns:
        analogies_by_section ({str: [str]}): Mapping of abbreviated section header
            to list of strig analogies for that section.

    """
    with open(filename_src, "r") as f:
        analogy_list = f.readlines()

    section_mapping = {
        ": crystal structures (zincblende, wurtzite, rutile, rocksalt, etc.)": "compounds_structures",
        ": crystal symmetry (cubic, hexagonal, tetragonal, etc.)": "compounds_symmetry",
        ": magnetic properties": "compounds_magnetic",
        ": metals and their oxides (most common)": "oxides",
    }


    analogies_by_section = {sh: [] for sh in section_mapping.values()}
    compounds_list = []

    current_section = None
    for line in tqdm.tqdm(analogy_list):
        line_clean = line.replace("\n", "")
        if ":" in line_clean:
            current_section = section_mapping.get(line_clean, None)
        elif current_section:

            line_split = line_clean.split(" ")
            if len(line_split) != 4:
                raise ValueError

            # remove unusable analogies as per this function doc rules
            comps_org_strs = []
            comps_pmg_objs = []
            for word in line_split:
                try:
                    c = Composition(word)
                    comps_pmg_objs.append(c)
                    comps_org_strs.append(word)
                except (CompositionError, ValueError):
                    pass

            if not all([len(comp.elements) == 1 for comp in comps_pmg_objs]) and comps_pmg_objs:
                for i, comp in enumerate(comps_pmg_objs):
                    if len(comp.elements) > 1:
                        compounds_list.append(comps_org_strs[i])
            else:
                if not quiet:
                    print(f"Analogy '{line_clean}' not included as per rules!")

            analogies_by_section[current_section].append(line_clean)
        else:
            continue

    compounds_list = list(set(compounds_list))

    fname_analogies_restricted = full_analogy_path.replace(".txt", "_") + output_suffix + ".txt"
    with open(fname_analogies_restricted, "w") as f:
        for section, analogy_set in analogies_by_section.items():
            f.write(":" + section + "\n")
            for analogy in analogy_set:
                f.write(analogy + "\n")

    fname_compounds_list = full_analogy_path.replace("analogies.txt", "excluded_materials_") + output_suffix + ".json"
    with open(fname_compounds_list, "w") as f:
        json.dump(compounds_list, f)

    if not quiet:
        n_analogies = sum([len(aset) for aset in analogies_by_section.values()])
        print(f"Extracted {n_analogies} analogies, containing {len(compounds_list)} unique compounds.")
        print(f"Write files {fname_analogies_restricted} and {fname_compounds_list}")

    return analogies_by_section, compounds_list


if __name__ == "__main__":
    create_restricted_analogies(quiet=False)