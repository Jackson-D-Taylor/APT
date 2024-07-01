import pint.models as pm
import pint.logging
import numpy as np
import sys
import os
from pathlib import Path
from pint.models.model_builder import parse_parfile
from copy import deepcopy

pint.logging.setup(level='WARNING')
original = Path("/data1/people/jdtaylor/binary_fake_data6")
alg_saves = original / Path("alg_saves")
successes = np.zeros(100, dtype = bool)
for i in range(1,101):
    os.chdir(original)

    m_cor, t_cor = pm.get_model_and_toas(f"fake_{i}sol.par", f"fake_{i}.tim")
    t_cor_copy = deepcopy(t_cor)
    t_cor_copy.compute_pulse_numbers(m_cor)
    pn_cor = t_cor_copy.table["pulse_number"]

    top_path = alg_saves / Path(f"fake_{i}")
    os.chdir(top_path)
    print(f"cwd1 = {os.getcwd()}")
    print(f"listdir = {os.listdir()}", end = "\n\n")
    for dir in os.listdir():
        if "old" in dir:
            continue
        dir = top_path / Path(dir)
        print(f"cwd2 = {os.getcwd()}")
        # print(f"listdir = {os.listdir()}")
        os.chdir(dir)
        print(f"cwd3 = {os.getcwd()}", end = "\n\n")

        if f"fake_{i}.fin" not in os.listdir():
            os.chdir("..")
            continue
        m_fin = pm.get_model(f"fake_{i}.fin")
        t_fin = deepcopy(t_cor)
        t_fin.compute_pulse_numbers(m_fin)
        pn_fin = t_fin.table["pulse_number"]

        identical = np.array_equal(pn_fin, pn_cor)
        print(f"{dir} is {identical}")

        os.chdir("..")
        if identical:
            successes[i-1] = 1
            break

        
    # os.chdir(original)

print(successes)
    