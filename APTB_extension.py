#!/usr/bin/env python
# -W ignore::FutureWarning -W ignore::UserWarning -W ignore:DeprecationWarning
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import pint.utils
import pint.models.model_builder as mb
import pint.random_models
from pint.phase import Phase
from pint.fitter import WLSFitter
from copy import deepcopy
from astropy import log
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import operator
import time
from pathlib import Path
import socket
import APTB
import treelib
import matplotlib as mpl


"""
The intention of this file is to store functions to make adding additional binary models to APTB
seamless and standardized. For example, APTB will, instead of checking for the type of binary model 
and its features (like not fitting for EPS1 and EPS2 immediately), APTB will call a function from this
file that will do the equivalent process. This also serves to prevent clutter in APTB.
"""


def set_binary_pars_lim(m, args):
    if args.binary_model.lower() == "ell1" and args.EPS_lim is None:
        if args.EPS_lim == "inf":
            args.EPS_lim = np.inf
        else:
            args.EPS_lim = m.PB.value * 5
            args.EPS_lim = m.PB.value * 5

    elif args.binary_model.lower() == "bt":
        if args.ECC_lim:
            if args.ECC_lim == "inf":
                args.ECC_lim = np.inf
        else:
            args.ECC_lim = 0

        if args.OM_lim:
            if args.OM_lim == "inf":
                args.OM_lim = np.inf
        else:
            args.OM_lim = 0
    elif args.binary_model.lower() == "dd":
        if args.ECC_lim:
            if args.ECC_lim == "inf":
                args.ECC_lim = np.inf
        else:
            args.ECC_lim = 0

        if args.OM_lim:
            if args.OM_lim == "inf":
                args.OM_lim = np.inf
        else:
            args.OM_lim = 0

        if args.OMDOT_lim:
            if args.OMDOT_lim == "inf":
                args.OMDOT_lim = np.inf
        else:
            args.OMDOT_lim = 0
        pass

    return args


def do_Ftests_binary(m, t, f, f_params, span, Ftests, args):
    """
    Helper function for APTB.
    """

    if args.binary_model.lower() == "ell1":
        # want to add eps1 and eps2 at the same time
        if "EPS1" not in f_params and span > args.EPS_lim * u.d:
            Ftest_F, m_plus_p = APTB.Ftest_param(m, f, "EPS1&2", args)
            Ftests[Ftest_F] = "EPS1&2"
        # if "EPS2" not in f_params and span > args.EPS2_lim * u.d:
        #     Ftest_F = APTB.Ftest_param(m, f, "EPS2", args)
        #     Ftests[Ftest_F] = "EPS2"

    elif args.binary_model.lower() == "bt":
        for param in ["ECC", "OM"]:
            if (
                param not in f_params and span > getattr(args, f"{param}_lim") * u.d
            ):  # args.F0_lim * u.d:
                Ftest_R, m_plus_p = APTB.Ftest_param(m, f, param, args)
                Ftests[Ftest_R] = param
    elif args.binary_model.lower() == "dd":
        for param in ["ECC", "OM", "OMDOT"]:
            if (
                param not in f_params and span > getattr(args, f"{param}_lim") * u.d
            ):  # args.F0_lim * u.d:
                Ftest_R, m_plus_p = APTB.Ftest_param(m, f, param, args)
                Ftests[Ftest_R] = param

    return m, t, f, f_params, span, Ftests, args


def skeleton_tree_creator(
    blueprint, iteration_dict=None, blueprint_string=False, format="tuple"
):
    """
    This creates what the tree looks like, without any of the data attributes.

    Parameters
    ----------
    blueprint : blueprint in the form [(parent, child) for node in tree]

    Returns
    -------
    tree : treelib.Tree
    """
    tree = treelib.Tree()
    tree.create_node("Root", "Root")
    U_counter = 0
    bp_string = ""
    if iteration_dict and format == "tuple":
        for parent, child in blueprint:
            if parent != "Root":
                i_index = parent.index("i")
                d_index = parent.index("d")
                parent = f"i{iteration_dict.get(parent, f'U{(U_counter:=U_counter+1)}')}_{parent[d_index:i_index-1]}"
            i_index = child.index("i")
            d_index = child.index("d")
            child = f"i{iteration_dict.get(child, f'U{(U_counter:=U_counter+1)}')}_{child[d_index:i_index-1]}"
            tree.create_node(child, child, parent=parent)
            if blueprint_string:
                bp_string += f"{parent},{child};"
    elif format == "tuple":
        for parent, child in blueprint:
            tree.create_node(child, child, parent=parent)
            if blueprint_string:
                bp_string += f"{parent},{child};"
    elif format == "string":
        for parent_child in blueprint:
            parent, child = parent_child.split(",")
            tree.create_node(child, child, parent=parent)
    else:
        print(f"format = {format}")
        raise Exception("Unsupported blueprint format")
    if blueprint_string:
        return tree, bp_string[:-1]
    return tree


def linear_interp(a, b, n):
    x = np.linspace(a[0], b[0], n)
    y = np.linspace(a[1], b[1], n)
    return x, y


def solution_tree_grapher(tree, blueprint):
    font = {"weight": "bold", "size": 18}
    mpl.rc("font", **font)
    # tree = skeleton_tree_creator(blueprint)

    node_data = {"Root": [0, 0, 0]}
    l0 = 50
    eps = 0.18
    d = {0: 50, 1: 3, 2: l0}
    depth_numbers = {0: 1}
    for i in range(1, tree.depth() + 1):
        nodes_at_depth = list(tree.filter_nodes(lambda x: tree.depth(x) == i))
        depth_numbers[i] = len(nodes_at_depth)

    for i, parent_child in enumerate(blueprint):
        parent, child = parent_child
        depth = tree.depth(parent)
        l_default = l0 * eps ** (depth)
        l = l_default
        children_number = len(tree.children(parent))
        parent_x, parent_y, children_assigned = node_data[parent]
        node_data[parent][2] = children_assigned + 1
        if children_number == 1:
            w = 0
        else:
            w = 2 * l / (children_number - 1)

        if child == "1":
            print(child)
            print(children_assigned)

        p_left = (children_number - 1) / 2 * -w
        child_x = parent_x + p_left + w * children_assigned
        node_data[child] = [child_x, parent_y - 1, 0]

    x_value = np.zeros(len(node_data))
    y_value = np.zeros(len(node_data))
    for i, node_list in enumerate(node_data.values()):
        x_value[i], y_value[i], _ = node_list

    x = []
    y = []
    for i, total in enumerate(blueprint):
        parent, child = total
        x_temp, y_temp = linear_interp(node_data[parent], node_data[child], 1000)
        x.append(list(x_temp))
        y.append(list(y_temp))

    x = np.array(x).flatten()
    y = np.array(y).flatten()

    fig, ax = plt.subplots(figsize=(15, 12))
    ax.plot(x_value, y_value, "ro", alpha=0.4, label="Nodes")
    im = ax.scatter(x, y, c=np.linspace(0, 1, len(x)))
    ax.set_yticks([-i for i in range(tree.depth() + 1)])
    ax.set_xticks([])
    ax.set_yticklabels([i for i in range(tree.depth() + 1)])
    ax.set_ylabel("Depth")
    ax.set_title(f"Solution Tree")
    cbar = fig.colorbar(im, ticks=[0, 1])
    cbar.ax.set_yticklabels(["First", "Last"])

    return fig
    # fig.savefig(f"solution_tree.jpg", bbox_inches="tight")
