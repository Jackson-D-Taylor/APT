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
from pint.models import parameter, PhaseJump
from pint.fitter import WLSFitter
from copy import deepcopy
import astropy.units as u
from astropy.coordinates import Angle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from pathlib import Path
import socket
from APT.APT import get_closest_cluster, solution_compare, bad_points
from APT import APTB_extension
from loguru import logger as log
import treelib
from dataclasses import dataclass
import colorama


class CustomTree(treelib.Tree):
    """
    A treelib.Tree function with additional methods and attributes.

    """

    def __init__(
        self,
        tree=None,
        deep=False,
        node_class=None,
        identifier=None,
        blueprint=None,
        explored_blueprint=None,
        save_location=Path.cwd(),
        node_iteration_dict={"Root": 0},
        g=list(),
        G=None,
    ):
        super().__init__(tree, deep, node_class, identifier)

        # will store this in a list so that when a branch is pruned
        # I can reconstruct the skeleton of the tree later
        self.blueprint = blueprint
        # The python docs recommends doing default empty lists like this
        if blueprint is None:
            self.blueprint = []
        if explored_blueprint is None:
            self.explored_blueprint = []
        self.save_location = save_location
        self.node_iteration_dict = node_iteration_dict

    def branch_creator(
        self,
        f,
        m_wrap_dict,
        min_wrap_vertex,
        iteration,
        chisq_wrap,
        min_wrap_number_total,
        mask_with_closest,
        closest_cluster_mask,
        maxiter_while,
        args,
        unJUMPed_clusters,
        cluster_distances,
        cluster_to_JUMPs,
    ):
        """
        Explores neighboring phase wraps and produced branches on those that give an acceptable reduced chisq.

        Parameters
        ----------
        f : fitter
        m_wrap_dict : dictionary with the format {wrap: model}
        min_wrap_vertex : the vertex of the parabola defined by the sampling points
        iteration : the global iteration
        chisq_wrap : dictionary with the format {reduced chisq: wrap}
        min_wrap_number_total : the wrap number predicted by quad_phase_wrap_checker
        mask_with_closest : the mask of every unJUMPed cluster
        closest_cluster_mask : the mask of the most recently unJUMPed cluster
        maxiter_while : the maxiter for f.fit_toas
        args : command line arguments
        unJUMPed_clusters : an array of the number every serially unJUMPed cluster
        """
        current_parent_id = self.current_parent_id
        parent_depth = self.depth(current_parent_id)
        depth = parent_depth + 1  # depth of the models about to be created
        m_copy = deepcopy(f.model)
        t = f.toas

        def branch_node_creator(self, node_name: str, number: int):
            """
            More or less a wrapper function of tree.create_node. Only has meaning in branch_creator.

            Parameters
            ----------
            node_name : str
                name of node to be created
            number : int
                wrap_number

            Returns
            -------
            bool
                validator result
            """

            if args.debug_mode:
                print(
                    f"{colorama.Fore.GREEN}Branch created: parent = {current_parent_id}, name = {node_name}{colorama.Style.RESET_ALL}"
                )
            self.blueprint.append((current_parent_id, node_name))
            data = NodeData(
                m_wrap_dict[number],
                deepcopy(unJUMPed_clusters),
                deepcopy(cluster_distances),
                deepcopy(cluster_to_JUMPs),
            )
            if data.data_is_valid(args):
                self.create_node(
                    node_name,
                    node_name,
                    parent=current_parent_id,
                    data=data,
                )
                return True
            elif args.debug_mode:
                # the branch was never really created to begin with
                print(
                    f"{colorama.Fore.RED}Validation failed, branch creation stopped: parent = {current_parent_id}, name = {node_name}{colorama.Style.RESET_ALL}"
                )
                return False

        node_name = f"d{depth}_w{min_wrap_number_total}_i{iteration}"
        chisq_wrap_reversed = {i: j for j, i in chisq_wrap.items()}
        if branch_node_creator(
            self,
            node_name,
            min_wrap_number_total,
        ):
            chisq_accepted = {
                chisq_wrap_reversed[min_wrap_number_total]: min_wrap_number_total
            }
            branches = {chisq_wrap_reversed[min_wrap_number_total]: node_name}
        else:
            chisq_accepted = {}
            branches = {}

        # if min_wrap_number_total is also the wrap predicted by the vertex, then its
        # immediate neighbors have already been calculated. Thus, this can save a little
        # bit of time, especially if its immediate neighbors already preclude
        # the need for any branches
        if min_wrap_vertex == min_wrap_number_total:
            increment_factor_list = []
            for i in [-1, 1]:
                wrap_i = min_wrap_number_total + i
                if chisq_wrap_reversed[wrap_i] < args.prune_condition:
                    node_name = f"d{depth}_w{wrap_i}_i{iteration}"
                    if branch_node_creator(
                        self,
                        node_name,
                        wrap_i,
                    ):
                        chisq_accepted[chisq_wrap_reversed[wrap_i]] = wrap_i
                        branches[chisq_wrap_reversed[wrap_i]] = node_name
                        increment_factor_list.append(i)

            increment = 2
        else:
            increment_factor_list = [-1, 1]
            increment = 1
        while increment_factor_list:
            # need list() to make a copy to not corrupt the iterator
            for increment_factor in list(increment_factor_list):
                wrap = min_wrap_number_total + increment_factor * increment
                # print(f"increment_factor_list = {increment_factor_list}")
                # print(f"min_wrap_number_total = {min_wrap_number_total}")
                # print(f"wrap = {wrap}")
                t.table["delta_pulse_number"][closest_cluster_mask] = wrap
                f.fit_toas(maxiter=maxiter_while)

                # t_plus_minus["delta_pulse_number"] = 0
                # t_plus_minus.compute_pulse_numbers(f_plus_minus.model)
                f_resids_chi2_reduced = f.resids.chi2_reduced
                m_wrap_dict[wrap] = deepcopy(f.model)
                chisq_wrap[f_resids_chi2_reduced] = wrap
                f.model = deepcopy(m_copy)

                if f_resids_chi2_reduced < args.prune_condition:
                    node_name = f"d{depth}_w{wrap}_i{iteration}"
                    if branch_node_creator(self, node_name, wrap):
                        chisq_accepted[f_resids_chi2_reduced] = wrap
                        branches[f_resids_chi2_reduced] = node_name
                else:
                    increment_factor_list.remove(increment_factor)
                    # print("removed")
            #     print("right after")
            # print("increment += 1")
            increment += 1

        chisq_list = list(chisq_accepted.keys())
        chisq_list.sort()
        # the branch order that APTB will follow
        if args.debug_mode:
            print("In branch_creator:")
            print(f"\tchisq_wrap = {chisq_wrap}")
            print(f"\tbranches = {branches}")
            print(
                f"\tchisq_accepted = {chisq_accepted}"
            )  # TODO use chisq_accepted to populate solution_tree.g and solution_tree.G
        growth = len(chisq_accepted)
        self.g.append(growth)
        self.G[parent_depth] = self.G.get(parent_depth, list())
        self.G[parent_depth].append(growth)

        order = [branches[chisq] for chisq in chisq_list]
        self[current_parent_id].order = order

    def node_selector(self, f, args, iteration):
        """
        Selects the next branch to take based on the order instance of the parent. Also prunes dead branches.

        Parameters
        ----------
        f : fitter
        args : command line arguments

        Returns
        -------
        m, unJUMPed_clusters
            These are for the selected branch
        """
        current_parent_id = self.current_parent_id
        current_parent_node = self[current_parent_id]

        while True:
            if current_parent_node.order:
                break
            elif current_parent_id == "Root":
                # program is done
                return (None, None, None, None), " "

            new_parent = self.parent(current_parent_id)
            self.prune(current_parent_id, args)

            current_parent_id = new_parent.identifier
            current_parent_node = self[current_parent_id]
        # selects the desired branch and also removes it from the order list
        selected_node_id = current_parent_node.order.pop(0)
        selected_node = self[selected_node_id]
        # if
        # explored_name = f"i{iteration}_{selected_node_id[selected_node_id.index('d'):]}"
        # self.explored_blueprint.append(
        #     (
        #         self.current_parent_id,
        #         explored_name,
        #     )
        # )
        self.current_parent_id = selected_node_id
        self.node_iteration_dict[selected_node_id] = iteration

        i_index = selected_node_id.index("i")
        d_index = selected_node_id.index("d")
        explored_name = f"i{iteration}_{selected_node_id[d_index:i_index-1]}"
        # explored_name = f"I{iteration}_{selected_node_id}"
        # self.explored_blueprint.append((, explored_name))

        return selected_node.data, explored_name

    def prune(self, id_to_prune, args):
        """
        Wrapper function for Tree.remove_node.

        Parameters
        ----------
        id_to_prune : node.identifier
        args : command line arguments
        """
        if args.debug_mode:
            print(
                f"{colorama.Fore.RED}Pruning in progress (id = {id_to_prune}){colorama.Style.RESET_ALL}"
            )
            self.show()
        self.remove_node(id_to_prune)
        # perhaps more stuff here:

    def current_depth(self):
        """
        Wrapper function for depth. Meant to declutter.
        """
        return self.depth(self.current_parent_id)

    def pwss_size_estimate(self):
        """
        This function estimates the size of the solution tree. It uses the information on the
        number of branches per parent (g) and all of the g's at the same depth (stored in G).
        Then, applying Equation 3 and assuming Equation 4 of the APTB paper, it calculates
        the PWSS size estimate based on the current available tree, as yet explored.

        Returns
        -------
        int
            The estimated size of the PWSS
        """
        g = np.array(self.g)
        G = self.G

        g2_mask = g >= 2
        g2 = g[g2_mask]
        r1 = np.average(g2)
        r1_max = np.max(g)
        # print(f"r1 = {r1}")
        # print(f"r1_max = {r1_max}")

        g1 = g[~g2_mask]
        r2 = np.average(g1)
        # print(f"r2 = {r2}")
        D = max(G.keys())
        d0 = D + 1
        for k in range(D + 1):
            G_k = G[k]
            G_k_avg = np.average(G_k)
            if np.abs(G_k_avg - r2) < np.abs(G_k_avg - r1):
                d0 = k
                break
        # print(f"d0 = {d0}")

        W = [
            r1_max**d if d <= d0 else r1_max**d0 * r2 ** (d - d0) for d in range(D + 1)
        ]

        ### TODO figure out what to do with these
        self.r1 = r1
        self.r1_max = r1_max
        self.d0 = d0
        self.r2 = r2
        self.W = W
        ###

        return sum(W)


class CustomNode(treelib.Node):
    """
    Allows for additional functions
    """

    def __init__(self, tag=None, identifier=None, expanded=True, data=None):
        super().__init__(tag, identifier, expanded, data)
        self.order = None


@dataclass
class NodeData:
    m: pint.models.timing_model.TimingModel = None
    unJUMPed_clusters: np.ndarray = None
    cluster_distances: list = None
    cluster_to_JUMPs: np.ndarray = None

    def data_is_valid(self, args):
        """_summary_

        Parameters
        ----------
        args : command line arguments

        Returns
        -------
        bool
            whether the data is valid
        """

        # if already validated, no need to validate again
        validated = getattr(self, "validated", False)
        if validated:
            return validated

        # can easily add validator functions to this list
        validator_funcs = [
            self.F1_validator,
            self.A1_validator,
            self.RAJ_validator,
            self.DECJ_validator,
        ]
        self.validated = np.all([valid_func(args) for valid_func in validator_funcs])
        return self.validated

    def A1_validator(self, args):
        if args.binary_model is None:
            return True
        A1 = self.m.A1.value
        return_value = A1 >= 0

        if not return_value:
            log.warning(f"Negative A1! ({A1=}))")

        return return_value

    def F1_validator(self, args):
        if args.F1_sign_always is None:
            return True
        F1 = self.m.F1.value
        if args.F1_sign_always == "-":
            return_value = F1 <= 0
        elif args.F1_sign_always == "+":
            return_value = F1 >= 0

        if not return_value:
            log.warning(f"F1 wrong sign! ({F1=})")

        return return_value

    def RAJ_validator(self, args):
        if args.RAJ_prior is None:
            return True
        RAJ = self.m.RAJ.value
        return_value = True
        if args.RAJ_prior[0]:
            return_value = RAJ >= args.RAJ_prior[0]
        if return_value and args.RAJ_prior[1]:
            return_value = RAJ <= args.RAJ_prior[1]

        if not return_value:
            log.warning(f"The RAJ has left the boundary! ({RAJ=})")

        return return_value

    def DECJ_validator(self, args):
        if args.DECJ_prior is None:
            return True
        DECJ = self.m.DECJ.value
        return_value = True
        if args.DECJ_prior[0]:
            return_value = DECJ >= args.DECJ_prior[0]
        if return_value and args.DECJ_prior[1]:
            return_value = DECJ <= args.DECJ_prior[1]

        if not return_value:
            log.warning(f"The DECJ has left the boundary! ({DECJ=})")

        return return_value

    def __iter__(self):
        return iter(
            (
                self.m,
                self.unJUMPed_clusters,
                self.cluster_distances,
                self.cluster_to_JUMPs,
            )
        )


def starting_points(
    toas, args=None, score_function="original", start_type=None, start=[0], **kwargs
):
    """
    Choose which cluster to NOT jump, i.e. where to start.

    Parameters
    ----------
    toas : TOAs object
    args : command line arguments
    score_function : which function to use to rank TOAs

    Returns
    -------
    tuple : (mask_list[:max_starts], starting_cluster_list[:max_starts])
    """
    mask_list = []
    starting_cluster_list = []

    if start_type != None:
        if start_type == "clusters":
            for cluster_number in start:
                mask_list.append(toas.table["clusters"] == cluster_number)
            return mask_list, start
        else:
            raise Exception("Other start_types not supported")

    t = deepcopy(toas)
    if "clusters" not in t.table.columns:
        t.table["clusters"] = t.get_clusters(gap_limit=args.cluster_gap_limit * u.h)
    mjd_values = t.get_mjds().value
    dts = np.fabs(mjd_values - mjd_values[:, np.newaxis]) + np.eye(len(mjd_values))

    if score_function == "original":
        score_list = (1.0 / dts).sum(axis=1)
    # different powers give different starting masks
    elif score_function == "original_different_power":
        score_list = (1.0 / dts**args.score_power).sum(axis=1)
    else:
        raise TypeError(f"score function '{score_function}' not understood")

    # f = pint.fitter.WLSFitter(t, m)
    # f.fit_toas()
    i = -1
    while score_list.any():
        i += 1
        # print(i)
        hsi = np.argmax(score_list)
        score_list[hsi] = 0
        cluster = t.table["clusters"][hsi]
        # mask = np.zeros(len(mjd_values), dtype=bool)
        mask = t.table["clusters"] == cluster

        if i == 0 or not np.any(
            np.all(mask == mask_list, axis=1)
        ):  # equivalent to the intended effect of checking if not mask in mask_list
            mask_list.append(mask)
            starting_cluster_list.append(cluster)
    if args is not None:
        max_starts = args.max_starts
    else:
        max_starts = 5
    return (mask_list[:max_starts], starting_cluster_list[:max_starts])


def JUMP_adder_begginning_cluster(
    t, m, cluster_gap_limit, starting_cluster, cluster_max, mjds, clusters
):
    """
    Adds JUMPs to a timfile as the begginning of analysis.
    This differs from JUMP_adder_begginning in that the jump flags
    are named based on the cluster number, not sequenitally from 0.

    Parameters
    ----------
    mask : a mask to select which toas will not be jumped
    t : TOAs object
    m : model object
    output_parfile : name for par file to be written
    output_timfile : name for the tim file to be written

    Returns
    -------
    model, t
    """
    if "clusters" not in t.table.columns:
        t.table["clusters"] = t.get_clusters(gap_limit=cluster_gap_limit * u.h)

    # this adds the JUMPs based on the MJD range in which they apply
    m.add_component(PhaseJump(), validate=False)
    for c in range(cluster_max + 1):
        # if c == starting_cluster:
        #     continue
        mjds_cluster = mjds[clusters == c]
        # here we will use a range of MJDs
        par = parameter.maskParameter(
            "JUMP",
            key="mjd",
            value=0.0,
            key_value=[
                mjds_cluster.min(),
                mjds_cluster.max(),
            ],
            units=u.s,
            frozen=True if c == 0 else False,
        )
        m.components["PhaseJump"].add_param(par, setup=True)

    return m, t


def JUMP_remover_decider(depth, starting_cluster, smallest_distance, serial_depth):
    """Decides which JUMP to remove

    Parameters
    ----------
    depth : int
        The depth of the current model
    starting_cluster : int
        The starting clustern umber
    smallest_distance : int
        The smallest distance between clusters
    serial_depth : int
        The depth at which APTB will no longer serially phase connect

    Returns
    -------
    Boolean
        Whether APTB will serially connect or not
    """
    # returning True means do serial
    # TODO incorporate some level of decision making here

    # This is a good start for now
    return True if depth < serial_depth else False


def JUMP_remover(
    left_cluster, cluster_distances, distance, cluster_to_JUMPs, unJUMPed_clusters, m
):
    """Removed JUMP

    Parameters
    ----------
    left_cluster : int
        The cluster to the left of the gap being mapped
    cluster_distances : np.ndarray
        An array of the distances between clusters
    distance : float
        The smallest time between clusters
    cluster_to_JUMPs : list
        The list that gives the JUMP number of the cluster being indexed
    unJUMPed_clusters : list
        All clusters that have unJUMPed relative to some anchor cluster
    m : model object

    Returns
    -------
    tuple
        left_cluster, cluster_distances, cluster_to_JUMPs, unJUMPed_clusters, m, anchor_jump_numb, right_cluster, removed_JUMP_numb,
    """

    right_cluster = left_cluster + 1
    cluster_distances.remove(distance)
    anchor_jump_numb = cluster_to_JUMPs[left_cluster]

    removed_JUMP_numb = cluster_to_JUMPs[right_cluster]
    removed_JUMP_attr = f"JUMP{removed_JUMP_numb}"
    getattr(m, removed_JUMP_attr).frozen = True
    getattr(m, removed_JUMP_attr).value = 0
    getattr(m, removed_JUMP_attr).uncertainty = 0
    cluster_to_JUMPs[cluster_to_JUMPs == removed_JUMP_numb] = cluster_to_JUMPs[
        left_cluster
    ]

    right_mjd = getattr(m, removed_JUMP_attr).key_value[1]
    getattr(m, f"JUMP{anchor_jump_numb}").key_value[1] = right_mjd

    unJUMPed_clusters = np.append(unJUMPed_clusters, [left_cluster, right_cluster])

    return (
        left_cluster,
        cluster_distances,
        cluster_to_JUMPs,
        unJUMPed_clusters,
        m,
        anchor_jump_numb,
        right_cluster,
        removed_JUMP_numb,
    )


def JUMP_remover_total(
    f,
    t,
    unJUMPed_clusters,
    cluster_distances,
    cluster_distances_dict,
    starting_cluster,
    cluster_to_JUMPs,
    depth,
    mjds,
    clusters,
    cluster_max,
    serial_depth,
):
    """Takes care of all JUMP removing and decision making

    Parameters
    ----------
    f : fitter object
    t : TOA object
    unJUMPed_clusters : _type_
        _description_
    cluster_distances : np.array
        distances between successive clusters
    cluster_distances_dict : _type_
        _description_
    starting_cluster : int
        The starting clustern umber
    cluster_to_JUMPs : list
        The list that gives the JUMP number of the cluster being indexed
    depth : int
        The depth of the current model
    mjds : np.ndarray
        Array of the TOAs' MJDs
    clusters : np.ndarray
        Array of clusters that directly corresponds to each TOA
    cluster_max : int
        The highest cluster number
    serial_depth : int
        The depth at which APTB will no longer serially phase connect

    Returns
    -------
    tuple
        f, t, closest_cluster, closest_cluster_group, unJUMPed_clusters, cluster_distances, cluster_to_JUMPs, adder,
    """
    cluster_to_JUMPs_old = cluster_to_JUMPs.copy()
    smallest_distance = np.min(cluster_distances)
    serial = JUMP_remover_decider(
        depth, starting_cluster, smallest_distance, serial_depth
    )
    m = f.model

    if not serial:
        left_cluster = cluster_distances_dict[smallest_distance]
        (
            left_cluster,
            cluster_distances,
            cluster_to_JUMPs,
            unJUMPed_clusters,
            m,
            anchor_jump_numb,
            right_cluster,
            removed_JUMP_numb,
        ) = JUMP_remover(
            left_cluster,
            cluster_distances,
            smallest_distance,
            cluster_to_JUMPs,
            unJUMPed_clusters,
            m,
        )

    else:
        starting_cluster_jump_numb = cluster_to_JUMPs[starting_cluster]
        starting_cluster_group = np.where(
            cluster_to_JUMPs == starting_cluster_jump_numb
        )[0]
        group_left = starting_cluster_group.min()
        group_right = starting_cluster_group.max()

        if group_left == 0:
            left = False
            dist_right = cluster_distances_dict[group_right]
        elif group_right == cluster_max:
            left = True
            dist_left = cluster_distances_dict[group_left - 1]
        else:
            dist_left = cluster_distances_dict[group_left - 1]
            dist_right = cluster_distances_dict[group_right]

            left = True if dist_left < dist_right else False

        if left:
            (
                left_cluster,
                cluster_distances,
                cluster_to_JUMPs,
                unJUMPed_clusters,
                m,
                anchor_jump_numb,
                right_cluster,
                removed_JUMP_numb,
            ) = JUMP_remover(
                group_left - 1,
                cluster_distances,
                dist_left,
                cluster_to_JUMPs,
                unJUMPed_clusters,
                m,
            )
        else:
            (
                left_cluster,
                cluster_distances,
                cluster_to_JUMPs,
                unJUMPed_clusters,
                m,
                anchor_jump_numb,
                right_cluster,
                removed_JUMP_numb,
            ) = JUMP_remover(
                group_right,
                cluster_distances,
                dist_right,
                cluster_to_JUMPs,
                unJUMPed_clusters,
                m,
            )

    unJUMPed_clusters = np.unique(unJUMPed_clusters)  # this is probably unneccesary

    left_cluster_left_mjd = mjds[clusters == left_cluster].min()
    if left_cluster_left_mjd == getattr(m, f"JUMP{anchor_jump_numb}").key_value[0]:
        closest_cluster = left_cluster
        adder = 1
    else:
        closest_cluster = right_cluster
        adder = -1

    # cluster_to_JUMPs[cluster_to_JUMPs == removed_JUMP_numb] = cluster_to_JUMPs[
    #     left_cluster
    # ]
    closest_cluster_group = np.where(cluster_to_JUMPs_old == removed_JUMP_numb)[0]
    # left_mjd = getattr(m, f"JUMP{anchor_jump_numb}").key_value[0]
    # if left_mjd ==

    f.model = m
    return (
        f,
        t,
        closest_cluster,
        closest_cluster_group,
        unJUMPed_clusters,
        cluster_distances,
        cluster_to_JUMPs,
        adder,
    )


def serial_closest(unJUMPed_clusters, cluster_distances, cluster_max):
    """_summary_

    Parameters
    ----------
    unJUMPed_clusters : np.array
        list of the cluster numbers of the unJUMPed clusters which have only been unJUMPed serially
    cluster_distances : np.array
        distances between successive clusters
    cluster_max : int
        highest cluster number

    Returns
    -------
    int
        cluster number of the closest cluster to the unJUMPed clusters
    """
    left_cluster = np.min(unJUMPed_clusters)
    right_cluster = np.max(unJUMPed_clusters)

    if left_cluster == 0 and right_cluster == cluster_max:
        return None

    if left_cluster == 0:
        return right_cluster + 1

    if right_cluster == cluster_max:
        return left_cluster - 1

    left_distance = cluster_distances[left_cluster - 1]
    right_distance = cluster_distances[right_cluster]

    return left_cluster - 1 if left_distance < right_distance else right_cluster + 1


def phase_connector(
    toas: pint.toa.TOAs,
    model: pint.models.timing_model.TimingModel,
    connection_filter: str = "linear",
    cluster: int = "all",
    mjds_total: np.ndarray = None,
    residuals=None,
    cluster_gap_limit=2 * u.h,
    **kwargs,
):
    """
    Makes sure each cluster is phase connected with itself.

    Parameters
    ----------
    toas : TOAs object
    model : model object
    connection_filter : the basic filter for determing what is and what is not phase connected
        options: 'linear', 'polynomial'
    kwargs : an additional constraint on phase connection, can use any number of these
        options: 'wrap', 'degree'
    mjds_total : all mjds of TOAs, optional (may decrease runtime to include)

    Returns
    -------
    None
    """

    # these need to be reset before unwrapping occurs
    toas.table["delta_pulse_number"] = np.zeros(len(toas.get_mjds()))
    toas.compute_pulse_numbers(model)

    residuals_unwrapped = np.unwrap(np.array(residuals), period=1)
    toas.table["delta_pulse_number"] = residuals_unwrapped - residuals


def save_state(
    f,
    m,
    t,
    mjds,
    pulsar_name,
    iteration,
    folder,
    args,
    save_plot=False,
    show_plot=False,
    mask_with_closest=None,
    explored_name=None,
    **kwargs,
):
    """
    Records the par and tim files of the current state and graphs a figure.
    It also checks if A1 is negative and if it is, it will ask the user if APTB
    should attempt to fix it. When other binary models are implemented, other
    types of checks for their respective models would need to be implemented here,
    provided easy fixes are available.

    Parameters
    ----------
    f : fitter object
    m : model object
    t : TOAs object
    mjds : all mjds of TOAS
    pulsar_name : name of pulsar ('fake_#' in the test cases)
    iteration : how many times the while True loop logic has repeated + 1
    folder : path to directory for saving the state
    args : command line arguments
    save_plot : whether to save the plot or not (defaults to False)
    show_plot : whether to display the figure immediately after generation, iterupting the program (defaults to False)
    mask_with_closest : the mask with the non-JUMPed TOAs
    kwargs : additional keyword arguments

    Returns
    -------
    bool
        whether or not a managed error occured (True if no issues). Nonmanaged errors will completely hault the program
    """
    if not args.save_state:
        return True
    if explored_name:
        iteration = explored_name
    m_copy = deepcopy(m)
    t = deepcopy(t)

    t.write_TOA_file(folder / Path(f"{pulsar_name}_{iteration}.tim"))
    with open(folder / Path(f"{pulsar_name}_{iteration}.par"), "w") as file:
        file.write(m_copy.as_parfile())

    if save_plot or show_plot:
        fig, ax = plt.subplots(figsize=(12, 7))

        fig, ax = plot_plain(
            f,
            mjds,
            t,
            m,
            iteration,
            fig,
            ax,
            args.JUMPs_in_fit_params_list,
            mask_with_closest=mask_with_closest,
        )

        if show_plot:
            plt.show()
        if save_plot:
            fig.savefig(folder / Path(f"{pulsar_name}_{iteration}.png"))
        plt.close()
    # if the function has gotten to this point, (likely) no issues have occured
    return True


def plot_plain(
    f, mjds, t, m, iteration, fig, ax, JUMPs_in_fit_params_list, mask_with_closest=None
):
    """
    A helper function for save_state. Graphs the time & phase residuals.
    Including mask_with_closests colors red the TOAs not JUMPed.

    This function is largely inherited from APT with some small, but crucial, changes

    Parameters
    ----------
    f, mjds, t, m, iteration, mask)wtih_closest : identical to save_state
    fig : matplotlib.pyplot.subplots figure object
    ax : matplotlib.pyplot.subplots axis object

    Returns
    -------
    fig, ax
        These then will be handled, and likely saved, by the rest of the save_state function
    """
    # plot post fit residuals with error bars
    model0 = deepcopy(m)
    r = pint.residuals.Residuals(t, model0).time_resids.to(u.us).value

    xt = mjds
    ax.errorbar(
        mjds,
        r,
        t.get_errors().to(u.us).value,
        fmt=".b",
        label="post-fit",
    )

    if mask_with_closest is not None:
        ax.errorbar(
            mjds[mask_with_closest],
            r[mask_with_closest],
            t.get_errors().to(u.us).value[mask_with_closest],
            fmt=".r",
        )

    # string of fit parameters for plot title
    fitparams = ""
    if f:
        for param in f.get_fitparams().keys():
            if "JUMP" in str(param):
                if JUMPs_in_fit_params_list:
                    fitparams += f"J{str(param)[4:]} "
            else:
                fitparams += str(param) + " "

    # notate the pulsar name, iteration, and fit parameters
    plt.title(f"{m.PSR.value} Post-Fit Residuals {iteration} | fit params: {fitparams}")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Residual (us)")

    # set the y limit to be just above and below the max and min points
    yrange = abs(max(r) - min(r))
    ax.set_ylim(min(r) - 0.1 * yrange, max(r) + 0.1 * yrange)

    # scale to the edges of the points or the edges of the random models, whichever is smaller
    width = max(mjds) - min(mjds)
    if (min(mjds) - 0.1 * width) < (min(xt) - 20) or (max(mjds) + 0.1 * width) > (
        max(xt) + 20
    ):
        ax.set_xlim(min(xt) - 20, max(xt) + 20)

    else:
        ax.set_xlim(min(mjds) - 0.1 * width, max(mjds) + 0.1 * width)

    plt.grid()

    def us_to_phase(x):
        return (x / (10**6)) * m.F0.value

    def phase_to_us(y):
        return (y / m.F0.value) * (10**6)

    # include secondary axis to show phase
    secaxy = ax.secondary_yaxis("right", functions=(us_to_phase, phase_to_us))
    secaxy.set_ylabel("residuals (phase)")

    return fig, ax


def Ftest_param(r_model, fitter, param_name, args):
    """
    do an F-test comparing a model with and without a particular parameter added

    Note: this is NOT a general use function - it is specific to this code and cannot be easily adapted to other scripts

    Parameters
    ----------
    r_model : timing model to be compared
    fitter : fitter object containing the toas to compare on
    param_name : name of the timing model parameter to be compared

    Returns
    -------
    float
        the value of the F-test
    """
    # read in model and toas
    m_plus_p = deepcopy(r_model)
    toas = deepcopy(fitter.toas)

    # set given parameter to unfrozen
    if param_name == "EPS1&2":
        getattr(m_plus_p, "EPS1").frozen = False
        getattr(m_plus_p, "EPS2").frozen = False
    else:
        getattr(m_plus_p, param_name).frozen = False

    # make a fitter object with the chosen parameter unfrozen and fit the toas using the model with the extra parameter
    f_plus_p = pint.fitter.WLSFitter(toas, m_plus_p)
    f_plus_p.fit_toas()

    # calculate the residuals for the fit with (m_plus_p_rs) and without (m_rs) the extra parameter
    m_rs = pint.residuals.Residuals(toas, fitter.model)
    m_plus_p_rs = pint.residuals.Residuals(toas, f_plus_p.model)

    # calculate the Ftest, comparing the chi2 and degrees of freedom of the two models
    Ftest_p = pint.utils.FTest(
        float(m_rs.chi2), m_rs.dof, float(m_plus_p_rs.chi2), m_plus_p_rs.dof
    )
    # The Ftest determines how likely (from 0. to 1.) that improvement due to the new parameter is due to chance and not necessity
    # Ftests close to zero mean the parameter addition is necessary, close to 1 the addition is unnecessary,
    # and NaN means the fit got worse when the parameter was added

    # if the Ftest returns NaN (fit got worse), iterate the fit until it improves to a max of 3 iterations.
    # It may have gotten stuck in a local minima
    counter = 0

    while not Ftest_p and counter < 3:
        counter += 1

        f_plus_p.fit_toas()
        m_plus_p_rs = pint.residuals.Residuals(toas, f_plus_p.model)

        # recalculate the Ftest
        Ftest_p = pint.utils.FTest(
            float(m_rs.chi2), m_rs.dof, float(m_plus_p_rs.chi2), m_plus_p_rs.dof
        )

    # print the Ftest for the parameter and return the value of the Ftest
    print("Ftest" + param_name + ":", Ftest_p)
    return Ftest_p, f_plus_p.model


def do_Ftests(f, mask_with_closest, args):
    """
    Does the Ftest on the neccesarry parameters

    Parameters
    ----------
    t : TOAs object
    m : model object
    mask_with_closest : the clusters
    args : command line arguments

    Returns
    m (with particular parameters now potentially unfrozen)
    """

    # fit toas with new model
    # f.fit_toas()
    t = f.toas
    m = f.model

    t_copy = deepcopy(t)
    t_copy.select(mask_with_closest)  # TODO update this and the span variable

    # calculate the span of fit toas for comparison to minimum parameter spans
    span = t_copy.get_mjds().max() - t_copy.get_mjds().min()
    print("Current fit TOAs span:", span)

    Ftests = dict()
    f_params = []

    # make list of already fit parameters
    for param in m.params:
        if getattr(m, param).frozen == False:
            f_params.append(param)

    # if span is longer than minimum parameter span and parameter hasn't been added yet, do Ftest to see if parameter should be added
    if "F0" not in f_params and span > args.F0_lim * u.d:
        Ftest_F0, m_plus_p = Ftest_param(m, f, "F0", args)
        Ftests[Ftest_F0] = "F0"

    if "RAJ" not in f_params and span > args.RAJ_lim * u.d:
        Ftest_R, m_plus_p = Ftest_param(m, f, "RAJ", args)
        Ftests[Ftest_R] = "RAJ"

    if "DECJ" not in f_params and span > args.DECJ_lim * u.d:
        Ftest_D, m_plus_p = Ftest_param(m, f, "DECJ", args)
        Ftests[Ftest_D] = "DECJ"

    if "F1" not in f_params and span > args.F1_lim * u.d:
        Ftest_F, m_plus_p = Ftest_param(m, f, "F1", args)
        if args.F1_sign_always:
            # print(1)
            allow_F1 = True
            if args.F1_sign_always == "+":
                if m_plus_p.F1.value < 0:
                    Ftests[1.0] = "F1"
                    print(f"Disallowing negative F1! ({m_plus_p.F1.value})")
                    allow_F1 = False
            elif args.F1_sign_always == "-":
                # print(2)
                if m_plus_p.F1.value > 0:
                    # print(3)
                    Ftests[1.0] = "F1"
                    log.warning(f"Disallowing positive F1! ({m_plus_p.F1.value})")
                    allow_F1 = False
            if allow_F1:
                Ftests[Ftest_F] = "F1"
        else:
            Ftests[Ftest_F] = "F1"

    if "F2" not in f_params and span > args.F2_lim * u.d:
        Ftest_D, m_plus_p = Ftest_param(m, f, "F2", args)
        Ftests[Ftest_D] = "F2"

    if args.binary_model is not None:
        m, t, f, f_params, span, Ftests, args = APTB_extension.do_Ftests_binary(
            m, t, f, f_params, span, Ftests, args
        )

    # remove possible boolean elements from Ftest returning False if chi2 increases
    Ftests_reversed = {i: j for j, i in Ftests.items()}
    Ftests_keys = [key for key in Ftests.keys() if type(key) != bool]

    # if no Ftests performed, continue on without change
    if not Ftests_keys:
        # if span > 100 * u.d:
        log.debug(
            f"No F-tests conducted. Parameters fit for so far, not including jumps, are {[param for param in f_params if not 'JUMP' in param]}"
        )
        # print("F0, RAJ, DECJ, and F1 have all been added")

    # if smallest Ftest of those calculated is less than the given limit, add that parameter to the model. Otherwise add no parameters
    elif min(Ftests_keys) < args.Ftest_lim:
        add_param = Ftests[min(Ftests_keys)]
        print(
            f"{colorama.Fore.LIGHTGREEN_EX}adding param {add_param} with Ftest {min(Ftests_keys)}{colorama.Style.RESET_ALL}"
        )
        if add_param == "EPS1&2":
            getattr(m, "EPS1").frozen = False
            getattr(m, "EPS2").frozen = False
        else:
            getattr(m, add_param).frozen = False

        # sometimes it's neccesary/benefitial to add more
        # if add_param == "RAJ" or add_param == "DECJ":
        #     if "DECJ" in Ftests_reversed and Ftests_reversed["DECJ"] < 1e-7:
        #         print("Adding DECJ as well.")
        #         getattr(m, "DECJ").frozen = False
        #     if "RAJ" in Ftests_reversed and Ftests_reversed["RAJ"] < 1e-7:
        #         print("Adding RAJ as well")
        #         getattr(m, "RAJ").frozen = False

    if args.debug_mode:
        print(f"Ftests = {Ftests}")

    return m


def set_F0_lim(args, m):
    if args.F0_lim is not None:
        m.F0.frozen = True
    else:
        m.F0.frozen = False


def set_F1_lim(args, parfile):
    """
    if F1_lim not specified in command line, calculate the minimum span based on general F0-F1 relations from P-Pdot diagram

    Parameters
    ----------
    args : command line arguments
    parfile : parfile

    Returns
    -------
    None
    """

    if args.F1_lim is None:
        # for slow pulsars, allow F1 to be up to 1e-12 Hz/s, for medium pulsars, 1e-13 Hz/s, otherwise, 1e-14 Hz/s (recycled pulsars)
        F0 = mb.get_model(parfile).F0.value

        if F0 < 10:
            F1 = 10**-12

        elif 10 < F0 < 100:
            F1 = 10**-13

        else:
            F1 = 10**-14

        # rearranged equation [delta-phase = (F1*span^2)/2], span in seconds.
        # calculates span (in days) for delta-phase to reach 0.35 due to F1
        args.F1_lim = np.sqrt(0.35 * 2 / F1) / 86400.0
    elif args.F1_lim == "inf":
        args.F1_lim = np.inf


def set_F2_lim(args, parfile):
    if args.F2_lim is None or args.F2_lim == "inf":
        args.F2_lim = np.inf
    # elif:
    #     pass


def set_RAJ_lim(args, parfile):
    if args.RAJ_lim is None:
        F0 = mb.get_model(parfile).F0.value
        args.RAJ_lim = -30 / 700 * F0 + 40


def set_DECJ_lim(args):
    if args.DECJ_lim is None:
        args.DECJ_lim = 1.3 * args.RAJ_lim


# TODO make sure the JUMP changes keep this intact
def quadratic_phase_wrap_checker(
    f,
    mask_with_closest,
    closest_cluster_mask,
    b,
    maxiter_while,
    closest_cluster,
    args,
    solution_tree,
    unJUMPed_clusters,
    cluster_distances,
    cluster_to_JUMPs,
    folder=None,
    wrap_checker_iteration=1,
    iteration=1,
    pulsar_name="fake",
):
    """
    Checks for phase wraps using the Freire and Ridolfi method.
    Their method assumes a quadratic depence of the reduced chi sq on the phase wrap number.

    Parameters
    ----------
    m : model
    t : TOAs object
    mask_with_closest : the mask with the non-JUMPed TOAs
    closest_cluster_mask : the mask with only the closest cluster
    b : which phase wraps to use as the sample
    maxiter_while : maxiter for WLS fitting
    closest_cluster : the cluster number of the closest cluster
    args : command line arguments
    folder : where to save any data
    wrap_checker_iteration : the number of times quadratic_phase_wrap_checker has
        done recursion
    iteration : the iteration of the while loop of the main algorithm

    Returns
    -------
    m, mask_with_closest
    """
    # run from highest to lowest b until no error is raised.
    # ideally, only b_i = b is run (so only once)
    t = f.toas
    t_copy = deepcopy(f.toas)
    m_copy = deepcopy(f.model)
    for b_i in range(b, -1, -1):
        # f = WLSFitter(t, m)
        chisq_samples = {}
        try:
            for wrap in [-b_i, 0, b_i]:
                t.table["delta_pulse_number"][closest_cluster_mask] = wrap
                f.fit_toas(maxiter=maxiter_while)
                chisq_samples[wrap] = f.resids.chi2_reduced
                f.model = deepcopy(m_copy)

            # if the loop did not encounter an error, then reassign b and end the loop
            b = b_i
            break

        except Exception as e:
            # try running it again with a lower b
            # sometimes if b is too high, APTB wants to fit the samples phase
            # wraps by allowing A1 to be negative
            print(f"(in handler) b_i is {b_i}")
            log.debug(f"b_i is {b_i}")
            # just in case an error prevented this
            f.model = deepcopy(m_copy)
            if b_i < 1:
                log.error(f"QuadraticPhaseWrapCheckerError: b_i is {b_i}")
                print(e)
                response = input(
                    "Should APTB continue anyway, assuming no phase wraps for the next cluster? (y/n)"
                )
                if response.lower() == "y":
                    return WLSFitter(t, m), t
                else:
                    raise e

    print(f"chisq_samples = {chisq_samples}")
    if len(chisq_samples) < 3:
        log.debug(f"chisq sampling failed, setting min_wrap_vertex = 0")
        min_wrap_vertex = 0
    else:
        min_wrap_vertex = round(
            (b / 2)
            * (chisq_samples[-b] - chisq_samples[b])
            / (chisq_samples[b] + chisq_samples[-b] - 2 * chisq_samples[0])
        )
    m_wrap_dict = {}
    chisq_wrap = {}
    # check +1, 0, and -1 wrap from min_wrap_vertex just to be safe
    # default is range(-1, 2) (i.e. [-1, 0, 1])
    for wrap in range(-args.vertex_wrap, args.vertex_wrap + 1):
        min_wrap_plusminus = min_wrap_vertex + wrap
        t.table["delta_pulse_number"][closest_cluster_mask] = min_wrap_plusminus
        f.fit_toas(maxiter=maxiter_while)

        # t_plus_minus["delta_pulse_number"] = 0
        # t_plus_minus.compute_pulse_numbers(f_plus_minus.model)

        chisq_wrap[f.resids.chi2_reduced] = min_wrap_plusminus
        ###f.reset_model()
        m_wrap_dict[min_wrap_plusminus] = deepcopy(f.model)
        f.model = deepcopy(m_copy)
        # t_wrap_dict[min_wrap_vertex + wrap] = deepcopy(t)
        # f_wrap_dict[min_wrap_vertex + wrap] = deepcopy(f)

    min_chisq = min(chisq_wrap.keys())
    print(f"chisq_wrap = {chisq_wrap}")

    # This likely means a new parameter needs to be added, in which case
    # the phase wrapper should be ran AFTER the F-test call:
    if min_chisq > args.prune_condition:
        # if the chisq is still above args.prune_condition (default = 2), then prune this branch
        if wrap_checker_iteration >= 2:
            print(
                f"quadratic_phase_wrap_checker ran twice without any difference, pruning branch"
            )
            # do not prune here, but by not giving the parent an order instance,
            # it will be pruned shortly after this return
            # solution_tree.branch_pruner(current_parent_id)
            # raise RecursionError(
            #     "In quadratic_phase_wrap_checker: maximum recursion depth exceeded (2)"
            # )
            parent_depth = solution_tree.depth(solution_tree.current_parent_id)
            growth = 0
            solution_tree.g.append(growth)
            solution_tree.G[parent_depth] = solution_tree.G.get(parent_depth, list())
            solution_tree.G[parent_depth].append(growth)

            # a tuple must be returned
            return None, None

        f.toas = t = deepcopy(t_copy)
        f.model = m = deepcopy(m_copy)
        log.warning(
            f"min_chisq = {min_chisq} > {args.prune_condition}, attempting F-test, then rerunning quadratic_phase_wrap_checker (iteration = {iteration})"
        )
        f.model = do_Ftests(f, mask_with_closest, args)
        # print(f"{f.model.F1.value=}")
        f.fit_toas(maxiter=maxiter_while)
        # print(f"{f.model.F1.value=}")
        print(f"reduced chisq is {f.resids.chi2_reduced}")

        phase_connector(
            t,
            f.model,
            "np.unwrap",
            cluster="all",
            residuals=pint.residuals.Residuals(t, m).calc_phase_resids(),
            mask_with_closest=mask_with_closest,
            wraps=True,
            cluster_gap_limit=args.cluster_gap_limit,
        )
        if args.pre_save_state:
            save_state(
                f,
                f.model,
                t,
                t.get_mjds().value,
                pulsar_name,
                args=args,
                folder=folder,
                iteration=f"{iteration}_wrap_checker{wrap_checker_iteration}",
                save_plot=True,
                mask_with_closest=mask_with_closest,
            )
        return quadratic_phase_wrap_checker(
            f,
            mask_with_closest,
            closest_cluster_mask,
            b,
            maxiter_while,
            closest_cluster,
            args,
            solution_tree,
            unJUMPed_clusters,
            cluster_distances,
            cluster_to_JUMPs,
            folder,
            wrap_checker_iteration + 1,
            iteration,
            pulsar_name,
        )

    min_wrap_number_total = chisq_wrap[min_chisq]

    print(
        f"Attemping a phase wrap of {min_wrap_number_total} on closest cluster (cluster {closest_cluster}).\n"
        + f"\tMin reduced chisq = {min_chisq}"
    )

    if args.branches:
        solution_tree.branch_creator(
            f,
            m_wrap_dict,
            min_wrap_vertex,
            iteration,
            chisq_wrap,
            min_wrap_number_total,
            mask_with_closest,
            closest_cluster_mask,
            maxiter_while,
            args,
            unJUMPed_clusters,
            cluster_distances,
            cluster_to_JUMPs,
        )
        return None, None

    else:
        m = m_wrap_dict[min_wrap_number_total]

        return m, mask_with_closest

    # t_closest_cluster.table["delta_pulse_number"] = min_wrap_number_total
    # t.table[closest_cluster_mask] = t_closest_cluster.table


def main_for_loop(
    parfile,
    timfile,
    args,
    mask_number,
    mask,
    starting_cluster,
    alg_saves_Path,
    toas,
    pulsar_name,
    mjds_total,
    maxiter_while,
    for_loop_start,
    solution_tree,
    start_time,
):

    # starting_cluster = starting_cluster_list[mask_number]

    # for multiprocessing, the mask_selector tells each iteration of main to skip
    # all but one of the masks
    # if mask_selector is not None and mask_number != mask_selector:
    #     return "continue"
    # if starting_cluster != 3:
    #     return "continue"
    print(
        f"\nMask number {mask_number} has started. Starting cluster: {starting_cluster}\n"
    )

    alg_saves_mask_Path = alg_saves_Path / Path(
        f"mask{mask_number}_cluster{starting_cluster}"
    )
    if not alg_saves_mask_Path.exists():
        alg_saves_mask_Path.mkdir()

    iterations_Path = alg_saves_mask_Path / Path("Iterations")
    if not iterations_Path.exists():
        iterations_Path.mkdir()

    solution_tree.save_location = alg_saves_mask_Path
    solution_tree.g = list()
    solution_tree.G = dict()

    m = mb.get_model(parfile)
    clusters = toas.table["clusters"]
    cluster_max = max_depth = np.max(clusters)
    m, t = JUMP_adder_begginning_cluster(
        toas,
        m,
        args.cluster_gap_limit,
        starting_cluster,
        cluster_max,
        mjds_total,
        clusters,
    )
    t.compute_pulse_numbers(m)
    args.binary_model = m.BINARY.value
    if args.binary_model is not None:
        args = APTB_extension.set_binary_pars_lim(m, args)

    # start fitting for main binary parameters immediately
    if args.binary_model is not None:
        if args.binary_model.lower() == "ell1":
            for param in ["PB", "TASC", "A1"]:
                getattr(m, param).frozen = False
        elif args.binary_model.lower() == "bt":
            for param in ["PB", "T0", "A1"]:
                getattr(m, param).frozen = False
            for param in ["ECC", "OM"]:
                if getattr(args, f"{param}_lim") is None:
                    getattr(m, param).frozen = False
        elif args.binary_model.lower() == "dd":
            for param in ["PB", "T0", "A1"]:
                getattr(m, param).frozen = False
            for param in ["ECC", "OM", "OMDOT"]:
                if getattr(args, f"{param}_lim") is None:
                    getattr(m, param).frozen = False

    set_F0_lim(args, m)

    # a copy, with the flags included
    base_toas = deepcopy(t)

    # this should be one of the few instantiations of f (the 'global' f)
    f = WLSFitter(t, m)

    # the following before the bigger while loop is the very first fit with only one cluster not JUMPed.
    # ideally, the following only runs once
    start_iter = 0
    while True:
        start_iter += 1
        if start_iter > 3:
            log.error(
                "StartingJumpError: Reduced chisq abnormally high, quitting program (use --no-start_warning to proceed anyway)."
            )
            raise RecursionError("In start: maximum recursion depth exceeded (3)")
        residuals_start = pint.residuals.Residuals(t, m).calc_phase_resids()

        # want to phase connect toas within a cluster first:
        phase_connector(
            t,
            m,
            "np.unwrap",
            cluster="all",
            mjds_total=mjds_total,
            residuals=residuals_start,
            wraps=True,
            cluster_gap_limit=args.cluster_gap_limit,
        )

        if not save_state(
            m=m,
            t=t,
            mjds=mjds_total,
            pulsar_name=pulsar_name,
            f=None,
            args=args,
            folder=iterations_Path,
            iteration=f"start_right_after_phase_connector{start_iter}",
            save_plot=True,
        ):
            # try next mask
            return "continue"

        # print("Fitting...")
        ####f = WLSFitter(t, m)
        ## f.model = m
        # print("BEFORE:", f.get_fitparams())
        # changing maxiter here may have some effects
        print(
            f.fit_toas(maxiter=4)
        )  # NOTE: need to investigate this ... this could make the base reduced chisq different than it should be

        if f.resids.dof < 0:
            log.error(
                f"With {len(t)} TOAs and {len(f.model.free_params)} free parameters (including JUMPs), DOF is {f.resids.dof}"
            )
            raise ValueError(f"DOF < 0")

        print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
        print("RMS in phase is", f.resids.phase_resids.std())
        print("RMS in time is", f.resids.time_resids.std().to(u.us))
        # print("\n Best model is:")
        # print(f.model.as_parfile())

        # new model so need to update table
        t.table["delta_pulse_number"] = 0
        t.compute_pulse_numbers(f.model)

        # update the model
        ########## m = f.model

        if not save_state(
            f,
            f.model,
            t,
            mjds_total,
            pulsar_name,
            args=args,
            folder=iterations_Path,
            iteration=f"start{start_iter}",
            save_plot=True,
            mask_with_closest=mask,
        ):
            return "continue"

        # something is certaintly wrong if the reduced chisq is greater than 3 at this stage
        chisq_start = f.resids.chi2_reduced
        # pint.residuals.Residuals(t, m).chi2_reduced
        log.info(f"The reduced chisq after the initial fit is {round(chisq_start, 3)}")
        if chisq_start > 3 and args.start_warning:
            log.warning(
                f"The reduced chisq after the initial fit is {round(chisq_start, 3)}"
            )
            # plt.plot(mjds_total, pint.residuals.Residuals(t, m).calc_phase_resids(), "o")
            # plt.show()
            print(
                f"The reduced chisq is {pint.residuals.Residuals(t, m).chi2_reduced}.\n"
                + "This is adnormally high. APTB will try phase connecting and fitting again.",
                end="\n\n",
            )

            # raise StartingJumpError("Reduced chisq adnormally high, quitting program.")

        else:
            break
    if args.prune_condition is None:
        args.prune_condition = 1 + chisq_start
    log.info(f"args.prune_condition = {args.prune_condition}")

    # TODO test for bad MJDs
    bad_mjds = []

    # mask_with_closest will be everything where the JUMPs between them and a neighbor have been removed
    mask_with_closest = deepcopy(clusters == 0)
    # mask_with_closest = deepcopy(mask)

    # unJUMPed_clusters = np.array([starting_cluster])
    unJUMPed_clusters = np.array([0])
    cluster_to_JUMPs = np.arange(1, cluster_max + 2)
    # tim_jump = deepcopy(clusters)
    cluster_distances = []
    cluster_distances_dict = {}
    for c in range(cluster_max):
        c0_mjds = mjds_total[clusters == c]
        c1_mjds = mjds_total[clusters == c + 1]
        cluster_distances.append(np.min(c1_mjds) - np.max(c0_mjds))
        cluster_distances_dict[cluster_distances[-1]] = c
        cluster_distances_dict[c] = cluster_distances[-1]

    # this starts the solution tree
    solution_tree.current_parent_id = "Root"
    iteration = 0
    while iteration < args.iteration_limit:
        # the main while True loop of the algorithm:
        iteration += 1

        if cluster_distances:
            (
                f,
                t,
                closest_cluster,
                closest_cluster_group,
                unJUMPed_clusters,
                cluster_distances,
                cluster_to_JUMPs,
                adder,
            ) = JUMP_remover_total(
                f,
                t,
                unJUMPed_clusters,
                cluster_distances,
                cluster_distances_dict,
                starting_cluster,
                cluster_to_JUMPs,
                solution_tree.current_depth(),
                mjds_total,
                clusters,
                cluster_max,
                args.serial_depth,
            )

            print(
                f"\n{colorama.Fore.LIGHTCYAN_EX}Removing the JUMP between clusters {closest_cluster} and {closest_cluster+adder}{colorama.Style.RESET_ALL}"
            )
        else:
            # save this file
            correct_solution_procedure(
                deepcopy(f),
                args,
                for_loop_start,
                mask,
                alg_saves_mask_Path / Path("Solutions"),
                iteration,
                timfile,
                pulsar_name,
                bad_mjds,
            )

            if args.find_all_solutions and args.branches:
                # need to load the next best branch, but need to do what happens after quad_phase_wrap_checker first

                data, explored_name = solution_tree.node_selector(f, args, iteration)
                f.model, unJUMPed_clusters, cluster_distances, cluster_to_JUMPs = data
                if f.model is None:
                    break
                mask_with_closest = np.isin(f.toas.table["clusters"], unJUMPed_clusters)
                f.toas = t = t_original
                t.table["delta_pulse_number"] = 0
                t.compute_pulse_numbers(f.model)
                f.model = do_Ftests(f, mask_with_closest, args)
                f.fit_toas(maxiter=maxiter_while)
                print(
                    f"{colorama.Fore.MAGENTA}reduced chisq at botttom of while loop is {f.resids.chi2_reduced}{colorama.Style.RESET_ALL}"
                )

                if not save_state(
                    f,
                    f.model,
                    t,
                    mjds_total,
                    pulsar_name,
                    args=args,
                    folder=iterations_Path,
                    iteration=f"i{iteration}_d{solution_tree.current_depth()}_c{closest_cluster}",
                    save_plot=True,
                    mask_with_closest=mask_with_closest,
                    explored_name=explored_name,
                ):
                    skip_mask = True
                    break

                # go to start of while True loop
                continue
            else:
                # end the program
                break

        if iteration % args.pwss_estimate == 0:  # TODO make this more accurare
            S = solution_tree.pwss_size_estimate()
            T = time.monotonic() - start_time
            tau = T / iteration
            terminal_message = ""
            terminal_message += (
                f"Phase wrap search space (PWSS) size estimator (beta):\n"
            )
            terminal_message += f"Iterations completed = {iteration}\n"
            terminal_message += f"Time elapsed (T) = {round(T)} s ({round(T/60)} m)\n"
            terminal_message += f"Time per iteration (tau) = {round(tau,2)} s\n\n"
            terminal_message += f"Estimated size of the total PWSS = {round(S)}\n"
            terminal_message += f"Estimated total time (S * tau) = {round(S*tau)} s ({round((S*tau)/60)} m)\n"
            terminal_message += f"Estimated time remaining (S * tau - T) = {round(S * tau - T)} s ({round((S * tau - T)/60)} m)\n"
            terminal_message += f"\ng = {solution_tree.g}\n"
            terminal_message += f"G = {solution_tree.G}\n\n"
            terminal_message += f"r1 = {solution_tree.r1}\n"
            terminal_message += f"r1_max = {solution_tree.r1_max}\n"
            terminal_message += f"r2 = {solution_tree.r2}\n"
            terminal_message += f"d0 = {solution_tree.d0}\n"
            terminal_message += f"W = {solution_tree.W}\n"
            print("\n" * 3 + "#" * 100)
            print(terminal_message)
            print("#" * 100 + "\n" * 3)
            terminal_message += "#" * 100 + "\n\n"

            with open(
                alg_saves_mask_Path / Path("time_estimate_file.txt"), "a"
            ) as file:
                file.write(terminal_message)

        # closest_cluster_mask = clusters == closest_cluster
        closest_cluster_mask = np.isin(clusters, closest_cluster_group)

        # TODO add polyfit here
        # random models can cover this instead
        # do slopes match from next few clusters, or does a quadratic/cubic fit

        mask_with_closest = np.logical_or(mask_with_closest, closest_cluster_mask)
        # unJUMPed_clusters = np.append(unJUMPed_clusters, closest_cluster) # I already do this in JUMP_remover

        t.table["delta_pulse_number"] = 0
        t.compute_pulse_numbers(f.model)
        residuals = pint.residuals.Residuals(t, f.model).calc_phase_resids()

        phase_connector(
            t,
            f.model,
            "np.unwrap",
            cluster="all",
            mjds_total=mjds_total,
            residuals=residuals,
            mask_with_closest=mask_with_closest,
            wraps=True,
            cluster_gap_limit=args.cluster_gap_limit,
        )
        # If args.pre_save_state is False (default) then the save will not be saved
        if args.pre_save_state and not save_state(
            f,
            f.model,
            t,
            mjds_total,
            pulsar_name,
            args=args,
            folder=iterations_Path,
            iteration=f"prefit_i{iteration}_d{solution_tree.current_depth()}_c{closest_cluster}",
            save_plot=True,
            mask_with_closest=mask_with_closest,
        ):
            skip_mask = True
            break

        if args.check_phase_wraps:
            t_original = deepcopy(t)
            f.model, mask_with_closest = quadratic_phase_wrap_checker(
                f,
                mask_with_closest,
                closest_cluster_mask,
                cluster_distances=cluster_distances,
                cluster_to_JUMPs=cluster_to_JUMPs,
                b=5,
                maxiter_while=maxiter_while,
                closest_cluster=closest_cluster,
                args=args,
                solution_tree=solution_tree,
                folder=iterations_Path,
                iteration=iteration,
                pulsar_name=pulsar_name,
                unJUMPed_clusters=unJUMPed_clusters,
            )
            if args.branches:
                data, explored_name = solution_tree.node_selector(f, args, iteration)
                if data is None:
                    solution_tree.show()
                    break
                try:
                    (
                        f.model,
                        unJUMPed_clusters,
                        cluster_distances,
                        cluster_to_JUMPs,
                    ) = data
                except ValueError as e:
                    print(f"data = {data}")
                    raise e
                if f.model is None:
                    break
                mask_with_closest = np.isin(f.toas.table["clusters"], unJUMPed_clusters)
            f.toas = t = t_original

        else:
            f.fit_toas(maxiter=maxiter_while)
        # print(f"{f.model=}")
        t.table["delta_pulse_number"] = 0
        t.compute_pulse_numbers(f.model)

        # use random models, or design matrix method to determine if next
        # cluster is within the error space. If the next cluster is not
        # within the error space, check for phase wraps.

        # TODO use random models or design matrix here

        # TODO add check_bad_points here-ish

        # use the F-test to determine if another parameter should be fit
        f.model = do_Ftests(f, mask_with_closest, args)

        # fit
        #########f = WLSFitter(t, m)
        #########f.model = m
        f.fit_toas(maxiter=maxiter_while)
        print(
            f"{colorama.Fore.MAGENTA}reduced chisq at botttom of while loop is {f.resids.chi2_reduced}{colorama.Style.RESET_ALL}"
        )

        #######m = f.model

        if not save_state(
            f,
            f.model,
            t,
            mjds_total,
            pulsar_name,
            args=args,
            folder=iterations_Path,
            iteration=f"i{iteration}_d{solution_tree.current_depth()}_c{closest_cluster}",
            save_plot=True,
            mask_with_closest=mask_with_closest,
            explored_name=explored_name,
        ):
            skip_mask = True
            break

        # repeat while True loop

    # end of main_for_loop
    print(f"End of mask {mask_number}")


def correct_solution_procedure(
    f,
    args,
    for_loop_start,
    mask,
    alg_saves_mask_solutions_Path,
    iteration,
    timfile,
    pulsar_name,
    bad_mjds,
):
    # skip this solution
    if args.F1_sign_solution == "+" and f.model.F1.value < 0:
        print(f"Solution discarded due to negative F1 ({f.model.F1.value=})!")
        return
    elif args.F1_sign_solution == "-" and f.model.F1.value > 0:
        print(f"Solution discarded due to positive F1 ({f.model.F1.value=})!")
        return

    if not alg_saves_mask_solutions_Path.exists():
        alg_saves_mask_solutions_Path.mkdir()
    # fit again with maxiter=4 for good measure
    f.fit_toas(maxiter=4)
    m = f.model
    t = f.toas
    # if skip_mask was set to true in the while loop, then move onto the next mask

    # try fitting with any remaining unfit parameters included and see if the fit is better for it
    m_plus = deepcopy(m)

    if args.final_fit_everything:
        getattr(m_plus, "RAJ").frozen = False
        getattr(m_plus, "DECJ").frozen = False
        getattr(m_plus, "F1").frozen = False

        if args.binary_model is not None:
            if args.binary_model.lower() == "ell1":
                getattr(m_plus, "EPS1").frozen = False
                getattr(m_plus, "EPS2").frozen = False

            elif args.binary_model.lower() == "bt":
                getattr(m_plus, "ECC").frozen = False
                getattr(m_plus, "OM").frozen = False

            elif args.binary_model.lower() == "dd":
                getattr(m_plus, "ECC").frozen = False
                getattr(m_plus, "OM").frozen = False
                getattr(m_plus, "OMDOT").frozen = False

    f_plus = pint.fitter.WLSFitter(t, m_plus)
    # if this is truly the correct solution, fitting up to 4 times should be fine
    f_plus.fit_toas(maxiter=4)

    # residuals
    r = pint.residuals.Residuals(t, f.model)
    r_plus = pint.residuals.Residuals(t, f_plus.model)
    if r_plus.chi2 <= r.chi2:
        f = deepcopy(f_plus)

    # print("Final Model:\n", f.model.as_parfile())

    fin_Path = alg_saves_mask_solutions_Path / Path(
        f"{f.model.PSR.value}_i{iteration}.par"
    )
    with open(fin_Path, "w") as finfile:
        finfile.write(f.model.as_parfile())
    tim_fin_name = Path(f.model.PSR.value + f"_i{iteration}.tim")
    f.toas.write_TOA_file(alg_saves_mask_solutions_Path / tim_fin_name)

    # plot final residuals if plot_final True
    xt = t.get_mjds()
    plt.clf()
    plt.close()
    fig, ax = plt.subplots()
    twinx = ax.twinx()
    ax.errorbar(
        xt.value,
        pint.residuals.Residuals(t, f.model).time_resids.to(u.us).value,
        t.get_errors().to(u.us).value,
        fmt=".b",
        label="post-fit (time)",
    )
    twinx.errorbar(
        xt.value,
        pint.residuals.Residuals(t, f.model).phase_resids,
        t.get_errors().to(u.us).value * float(f.model.F0.value) / 1e6,
        fmt=".b",
        label="post-fit (phase)",
    )
    chi2_reduced = pint.residuals.Residuals(t, f.model).chi2_reduced
    if f.model.F1.value:
        try:
            P1 = -((f.model.F0.value) ** (-2)) * f.model.F1.value
            round_numb = int(-np.log10(P1)) + 3
        except ValueError as e:
            P1 = 0
            round_numb = 1
    else:
        P1 = 0
        round_numb = 1
    ax.set_title(
        f"{m.PSR.value} Final Post-Fit Timing Residuals (reduced chisq={round(chi2_reduced, 4)}, (P1 = {round(P1, round_numb)}))"
    )
    ax.set_xlabel("MJD")
    ax.set_ylabel("Residual (us)")
    twinx.set_ylabel("Residual (phase)", labelpad=15)
    span = (0.5 / float(f.model.F0.value)) * (10**6)
    plt.grid()

    time_end_main = time.monotonic()
    print(
        f"Final Runtime (not including plots): {time_end_main - for_loop_start} seconds, or {(time_end_main - for_loop_start) / 60.0} minutes"
    )
    if args.plot_final:
        plt.show()

    fig.savefig(
        alg_saves_mask_solutions_Path / Path(f"{pulsar_name}_i{iteration}.png"),
        bbox_inches="tight",
    )
    plt.close()

    # if success, stop trying and end program
    if chi2_reduced < float(args.chisq_cutoff):
        print(
            "SUCCESS! A solution was found with reduced chi2 of",
            pint.residuals.Residuals(t, f.model).chi2_reduced,
            "after",
            iteration,
            "iterations",
        )
        if args.parfile_compare:
            while True:
                try:
                    identical_solution = solution_compare(
                        args.parfile_compare,
                        fin_Path,
                        timfile,
                    )
                    # if succesful, break the loop
                    break

                # if an error occurs, attempt again with the correct solution path
                except FileNotFoundError as e:
                    args.parfile_compare = input(
                        "Solution file not found. Input the full path here or enter 'q' to quit: "
                    )
                    if args.parfile_compare == "q":
                        identical_solution = "Unknown"
                        break
                    # else, try the loop again

            if identical_solution != "Unknown":
                print(
                    f"\n\nThe .fin solution and comparison solution ARE{[' NOT', ''][identical_solution]} identical.\n\n"
                )
            else:
                print(
                    f"\nSolution compare failed because the solution file could not be found."
                )

        print(f"The input parameters for this fit were:\n {args}")
        print(
            f"\nThe final fit parameters are: {[key for key in f.get_fitparams().keys()]}"
        )
        starting_TOAs = t[mask]
        print(
            f"starting points (clusters):\n {starting_TOAs.get_clusters(gap_limit = args.cluster_gap_limit * u.h)}"
        )
        print(f"starting points (MJDs): {starting_TOAs.get_mjds()}")
        print(f"TOAs Removed (MJD): {bad_mjds}")
        return "success"
