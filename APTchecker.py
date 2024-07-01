import pint.models as pm
import numpy as np
import sys
import os
from pathlib import Path
from pint.models.model_builder import parse_parfile
from concurrent.futures import ProcessPoolExecutor

"""
This script is intended to determine how many soltuions APT succesfully solved.
"""

original_path = Path.cwd()


def model_checker(number: int = None, path: Path = Path("/users/jdtaylor/Jackson/APT/binary_test_data/binary6_1")):
    """
    Checks if two parameter files give indetical timing solutions
    # FIXME This can be easily implented in the general case if desired.
    """
    os.chdir(path)
    # print(sys.argv[1])
    whole_list = False
    F1_list = []
    F0_list = []

    if number is None:
        number = int(sys.argv[1])
        numbers = [number]
        if number == -1:
            whole_list = True
            numbers = np.arange(1, 103)
    else:
        numbers = [number]
    identical_bool_array = [0 for i in range(len(numbers))]

    for number in numbers:
        try:
            m_cor, t_cor = pm.get_model_and_toas(
                f"fake_{number}sol.par", f"fake_{number}.tim"
            )

            t_cor.compute_pulse_numbers(m_cor)
            pn_cor = t_cor.table["pulse_number"]

            F1_info = parse_parfile(f"fake_{number}sol.par")["F1"][0]
            F0_info = parse_parfile(f"fake_{number}sol.par")["F0"][0]

            stop_index = F1_info.index(" 1 ")
            F1 = float(F1_info[:stop_index])
            F1_list.append(F1)

            stop_index = F0_info.index(" 1 ")
            F0 = float(F0_info[:stop_index])
            F0_list.append(F0)

            ###

            m_fin, t_fin = pm.get_model_and_toas(
                f"solved_{number}.par", f"fake_{number}.tim"
            )

            t_fin.compute_pulse_numbers(m_fin)
            pn_fin = t_fin.table["pulse_number"]

            value = np.array_equal(pn_cor, pn_fin)

            if not value:
                print("\n" * 6)
            print(f"number {number} is {value}")
            if not value:
                print("\n" * 6)

            if whole_list:
                identical_bool_array[number - 1] = [value, number, F0, F1]
            else:
                identical_bool_array[0] = [value, number, F0, F1]
        except Exception as error:
            print("\n" * 6)
            print("#"*80)
            print(os.listdir())
            print(f"{number} had an error. It is being skipped and marked False.")
            print(error)
            print("\n" * 6)
            if whole_list:
                identical_bool_array[number - 1] = [False, number, F0, F1]
            else:
                identical_bool_array[0] = [False, number, F0, F1]

            continue

    print(f"{identical_bool_array}, length = {len(identical_bool_array)}")
    return identical_bool_array


def main():
    model_checker()


if __name__ == "__main__":
    main()
