#!/usr/bin/env python

import numpy as np
import numpy.random as r

"""
This file provides quick support for including binary models into simdata without cluttering the original file. 
To include another binary model, simply follow TODO s 1 and 2.
"""


def binary_pars_setter(binary_model: str, *args) -> dict:
    if binary_model.upper() == "ELL1":
        return ELL1(*args)
    # TODO 1: add another if statement for another binary model
    else:
        raise Exception(f"The '{binary_model}' binary model is not available yet")


def binary_blur(binary_model: str, *args) -> dict:
    if binary_model.upper() == "ELL1":
        return ELL1_blur(*args)
    # TODO 1: add another if statement for another binary model
    else:
        raise Exception(f"The '{binary_model}' binary model is not available yet")


def ELL1(A1, EPS1, EPS2, PB, TASC, span) -> dict:
    """
    Sets the parameters for a binary pulsar with a low eccentricity.
    See pint.models.binary_ell1.py and pint.models.pulsar_binary.py for more info.
    """
    e = r.uniform(1e-7, 7e-4)  # eccentricity
    w = r.uniform(0, 2 * np.pi)  # longitude of periastron
    if A1 is None or A1 == "None":
        # cosi = r.uniform(0, 1)
        # i = np.arccos(cosi)  # inclination of orbit
        # ap = r.uniform(1e-3, 5)  # semi-major axis of pulsar, units of lightseconds
        # A1 = (ap * np.sin(i), 0.001)
        if r.uniform() > 0.1:
            log10A1 = r.uniform(np.log10(0.2), 1.61)
            A1 = (10**log10A1, 0.001)
        else:
            log10A1 = r.uniform(-2, np.log10(0.2))
            A1 = (10**log10A1, 0.001)
    else:
        A1 = A1
    if EPS1 is None or EPS1 == "None":
        EPS1 = (e * np.sin(w), 0.001)  # definition of eps1
    else:
        EPS1 = EPS1

    if EPS2 is None or EPS2 == "None":
        EPS2 = (e * np.cos(w), 0.001)  # definition of eps2
    else:
        EPS2 = EPS2

    if PB is None or PB == "None":
        if A1[0] > 0.2:
            PB = (10**1.243 + r.uniform(-0.5, 0.5), 0.001)
        else:
            PB = (10**0.5 + r.uniform(-0.25, 0.25), 0.001)
    else:
        PB = PB

    if TASC is None or TASC == "None":
        duration = int(r.uniform(span[0], span[1]))
        TASC = (r.uniform(0, duration) + 56000, 0.01)  # epoch of ascending node
    else:
        TASC = TASC

    return {"A1": A1, "EPS1": EPS1, "EPS2": EPS2, "PB": PB, "TASC": TASC}


def ELL1_blur(a1, eps1, eps2, pb, tasc):

    # a1, eps1, eps2, pb, tasc = (0, 0), (0,0), (0, 0), (0, 0), (0, 0)
    a1 = (a1[0] + r.uniform(0, 0.001) * a1[0], 0.1)
    eps1 = (0, 0)
    eps2 = (0, 0)
    # eps1 = (eps1[0] + r.uniform(0, 0.001) * eps1[0], 0.1)
    # eps2 = (eps2[0] + r.uniform(0, 0.001) * eps2[0], 0.1)
    pb = (pb[0] + r.uniform(0, 0.0001) * pb[0], 0.1)
    tasc = (tasc[0] + r.uniform(-0.1, 0.1), 0.1)

    binary_pars = {
        "A1": a1,
        "EPS1": eps1,
        "EPS2": eps2,
        "PB": pb,
        "TASC": tasc,
    }
    return binary_pars


# TODO 2: add a function here for the appropriate parameters. Keep in mind, that in simdata,
# binary_pars_setter is given 7 parameters..


def main():
    pass


if __name__ == "__main__":
    main()
