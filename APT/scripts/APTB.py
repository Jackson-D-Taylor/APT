import sys
import os
from copy import deepcopy
import argparse
from pathlib import Path
import numpy as np
from loguru import logger as log
import time

from astropy.coordinates import Angle
from astropy import units as u

import pint.toa
import pint.models.model_builder as mb
import pint.logging

from APT import APTB
from APT import APTB_extension

logformat = "{time} {level} {message}"


def APTB_argument_parse(parser, argv):
    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("timfile", help="tim file to read toas from")
    parser.add_argument(
        "--binary_model",
        help="which binary pulsar model to use.",
        choices=["ELL1", "ell1", "BT", "bt", "DD", "dd"],
        default=None,
    )
    parser.add_argument(
        "-sp",
        "--starting_points",
        help="mask array to apply to chose the starting points, clusters or mjds",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--maskfile",
        help="csv file of bool array for fit points",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n_pred",
        help="Number of predictive models that should be calculated",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--ledge_multiplier",
        help="scale factor for how far to plot predictive models to the left of fit points",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--redge_multiplier",
        help="scale factor for how far to plot predictive models to the right of fit points",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--F0_lim",
        help="minimum time span (days) before Spindown (F0) can be fit for (default = fit for F0 immediately)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--RAJ_lim",
        help="[days] minimum time span before Right Ascension (RAJ) can be fit for",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--DECJ_lim",
        help="[days] minimum time span before Declination (DECJ) can be fit for",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--RAJ_prior",
        help="Bounds on lower and upper RAJ value. Input in the form [lower bound (optional)],[upper bound (optional)] with no space after the comma."
        + "\nTo not include a lower or upper bound, still include the comma in the appropriate spot."
        + "\nThe bound should be entered in a form readable to the astropy.Angle class. ",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--DECJ_prior",
        help="Bounds on lower and upper DECJ value. Input in the form [lower bound (optional)],[upper bound (optional)] with no space after the comma."
        + "\nTo not include a lower or upper bound, still include the comma in the appropriate spot."
        + "\nThe bound should be entered in a form readable to the astropy.Angle class",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--F1_lim",
        help="[days] minimum time span before Spindown (F1) can be fit for (default = time for F1 to change residuals by 0.35phase)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--F2_lim",
        help="[days] minimum time span before Spindown (F1) can be fit for (default = infinity)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--F1_sign_always",
        help="require that F1 be either positive or negative (+/-/None) for the entire run time.",
        choices=["+", "-", "None"],
        type=str,
        default=None,
    )
    parser.add_argument(
        "--F1_sign_solution",
        help="require that F1 be either positive or negative (+/-/None) at the end of a solution.",
        choices=["+", "-", "None"],
        type=str,
        default="-",
    )
    parser.add_argument(
        "--EPS_lim",
        help="[days] minimum time span before EPS1 and EPS2 can be fit for (default = PB*5)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ECC_lim",
        help="[days] minimum time span before E can be fit for (default = PB*3)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--OM_lim",
        help="minimum time span before OM can be fit for (default = 0)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--OMDOT_lim",
        help="minimum time span before OMDOT can be fit for (default = 0)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--Ftest_lim",
        help="Upper limit for successful Ftest values",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--check_bad_points",
        help="whether the algorithm should attempt to identify and ignore bad data",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--plot_bad_points",
        help="Whether to actively plot the polynomial fit on a bad point. This will interrupt the program and require manual closing",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--check_bp_min_diff",
        help="minimum residual difference to count as a questionable point to check",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--check_bp_max_resid",
        help="maximum polynomial fit residual to exclude a bad data point",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--check_bp_n_clusters",
        help="how many clusters ahead of the questionable group to fit to confirm a bad data point",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--try_poly_extrap",
        help="whether to try to speed up the process by fitting ahead where polyfit confirms a clear trend",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--plot_poly_extrap",
        help="Whether to plot the polynomial fits during the extrapolation attempts. This will interrupt the program and require manual closing",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--pe_min_span",
        help="minimum span (days) before allowing polynomial extrapolation attempts",
        type=float,
        default=30,
    )
    parser.add_argument(
        "--pe_max_resid",
        help="maximum acceptable goodness of fit for polyfit to allow the polynomial extrapolation to succeed",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--span1_c",
        help="coefficient for first polynomial extrapolation span (i.e. try polyfit on current span * span1_c)",
        type=float,
        default=1.3,
    )
    parser.add_argument(
        "--span2_c",
        help="coefficient for second polynomial extrapolation span (i.e. try polyfit on current span * span2_c)",
        type=float,
        default=1.8,
    )
    parser.add_argument(
        "--span3_c",
        help="coefficient for third polynomial extrapolation span (i.e. try polyfit on current span * span3_c)",
        type=float,
        default=2.4,
    )
    parser.add_argument(
        "--vertex_wrap",
        help="how many phase wraps from the vertex derived in quad_phase_wrap_checker in each direction to try",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--plot_final",
        help="whether to plot the final residuals at the end of each attempt",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--data_path",
        help="where to store data",
        type=str,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--parfile_compare",
        help="par file to compare solution to. Default (False) will not activate this feature.",
        type=str,
        default=False,
    )
    parser.add_argument(
        "--chisq_cutoff",
        help="The minimum reduced chisq to be admitted as a potential solution.",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--max_starts",
        help="maximum number of initial JUMP configurations",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--multiprocessing",
        help="whether to include multiprocessing or not.",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--check_phase_wraps",
        help="whether to check for phase wraps or not",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--maxiter_while",
        help="sets the maxiter argument for f.fit_toas for fittings done within the while loop",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--score_power",
        help="The power that the score is raised to",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--try_all_masks",
        help="Whether to loop through different starting masks after having found at least one solution.\n"
        + "Distinct from args.find_all_solutions",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--save_state",
        help="Whether to take the time to save state. Setting to False could lower runtime, however making diagnosing issues very diffuclt.",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--pre_save_state",
        help="Whether to save pre-fit and wrap_checker states. False by default because the ordinary states are usually sufficient.",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--branches",
        help="Whether to try to solve phase wraps that yield a reduced chisq less than args.prune_condition",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--debug_mode",
        help="Whether to enter debug mode. (default = True is recommended for the time being)",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--depth_pursue",
        help="Past this tree depth, APTB will pursue the solution (no pruning) regardless of the chisq.\n"
        + "This is helpful for solutions with a higher chisq than normal, and with only one path that went particularly far.",
        type=int,
        default=np.inf,
    )
    parser.add_argument(
        "-pc",
        "--prune_condition",
        help="The reduced chisq above which to prune a branch.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--find_all_solutions",
        help="Whether to continue searching for more solutions after finding the first one",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--start_warning",
        help="Whether to turn on the start_warning if the reduced chisq is > 3 or not.",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--final_fit_everything",
        help="Whether to fit for the main parameters after phase connecting every cluster.",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--JUMPs_in_fit_params_list",
        help="Whether to fit for the main parameters after phase connecting every cluster.",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--cluster_gap_limit",
        help="Maximum time span, in hours, between separate clusters. (Default 2 hours)",
        type=float,
        default=2,
    )
    parser.add_argument(
        "-n",
        "--pulsar_name",
        help="The name of pulsar. This will decide the folder name where the iterations are saved. Defaults to name in par file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-sd",
        "--serial_depth",
        help="The depth that APTB will no longer connect clusters serially.",
        type=int,
        default=np.inf,
    )

    parser.add_argument(
        "-pwss",
        "--pwss_estimate",
        help="The iteration at which APTB will estimate the size of the solution tree.",
        type=int,
        default=35,
    )

    parser.add_argument(
        "-i",
        "--iteration_limit",
        help="The iteration at which APTB will stop, whether it found a solution or not.",
        type=int,
        default=10000,  # TODO change this to 2000 (maybe 1000?) for any public version. Ter5aq takes 90 minutes for i = ~500 (10 seconds/iteration)
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        default=0,
        action="count",
        help="Increase output verbosity",
        dest="verbosity",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        default=0,
        action="count",
        help="Decrease output verbosity",
        dest="quiet",
    )

    args = parser.parse_args(argv)
    verbosity = args.verbosity - args.quiet
    log.remove()
    if verbosity <= -1:
        log.add(sys.stderr, format=logformat, filter="APTB", level="ERROR")
        pint.logging.setup("ERROR")
    elif verbosity == 0:
        log.add(sys.stderr, format=logformat, filter="APTB", level="WARNING")
        pint.logging.setup("WARNING")
    elif args.verbosity == 1:
        log.remove()
        log.add(sys.stderr, format=logformat, filter="APTB", level="INFO")
        pint.logging.setup("INFO")
    elif args.verbosity >= 2:
        log.remove()
        log.add(sys.stderr, format=logformat, filter="APTB", level="DEBUG")
        pint.logging.setup("DEBUG")

    if args.branches and not args.check_phase_wraps:
        args.branches = False
        # raise argparse.ArgumentTypeError(
        #     "Branches only works if phase wraps are being checked."
        # )
    # interpret strings as booleans
    if args.depth_pursue != np.inf:
        raise NotImplementedError("depth_puruse")
    if args.F1_sign_always == "None":
        args.F1_sign_always = None
    if args.F1_sign_solution == "None":
        args.F1_sign_solution = None

    if args.RAJ_lim == "inf":
        args.RAJ_lim = np.inf
    if args.DECJ_lim == "inf":
        args.DECJ_lim = np.inf

    if args.RAJ_prior:
        RAJ_prior = args.RAJ_prior.split(",")
        args.RAJ_prior = [None, None]
        for i in (0, 1):
            if RAJ_prior[i]:
                args.RAJ_prior[i] = Angle(RAJ_prior[i]).hour

    if args.DECJ_prior:
        DECJ_prior = args.DECJ_prior.split(",")
        args.RAJ_prior = [None, None]
        for i in [0, 1]:
            if DECJ_prior[i]:
                args.DECJ_prior[i] = Angle(DECJ_prior[i]).deg

    return args, parser


def main():
    start_time = time.monotonic()

    parser = argparse.ArgumentParser(
        description="PINT tool for agorithmically timing binary pulsars."
    )
    args, parser = APTB_argument_parse(parser, argv=None)

    # save the inputted arguments to a file so can checked later
    # should save this string now before the args object is changed
    args_log = ""
    for arg in vars(args):
        args_log += f"{arg}={getattr(args, arg)}\n"

    # if given starting points from command line, check if ints (group numbers) or floats (mjd values)
    start_type = None
    start = None
    if args.starting_points != None:
        start = args.starting_points.split(",")
        try:
            start = [int(i) for i in start]
            start_type = "clusters"
        except:
            start = [float(i) for i in start]
            start_type = "mjds"

    """start main program"""
    # construct the filenames
    # datadir = os.path.dirname(os.path.abspath(str(__file__))) # replacing these lines with the following lines
    # allows APT to be run in the directory that the command was ran on
    # parfile = os.path.join(datadir, args.parfile)
    # timfile = os.path.join(datadir, args.timfile)
    data_path = Path(args.data_path)
    parfile = data_path / Path(args.parfile)
    timfile = data_path / Path(args.timfile)

    toas = pint.toa.get_TOAs(timfile)
    toas.table["clusters"] = toas.get_clusters(gap_limit=args.cluster_gap_limit * u.h)
    mjds_total = toas.get_mjds().value

    # every TOA, should never be edited
    all_toas_beggining = deepcopy(toas)
    if args.pulsar_name is None:
        pulsar_name = str(mb.get_model(parfile).PSR.value)
        args.pulsar_name = pulsar_name
    else:
        pulsar_name = args.pulsar_name
    alg_saves_Path = data_path / Path(f"alg_saves/{pulsar_name}")
    if not alg_saves_Path.exists():
        alg_saves_Path.mkdir(parents=True)

    with open(alg_saves_Path / Path("arguments.txt"), "w") as file:
        file.write(args_log)

    APTB.set_RAJ_lim(args, parfile)
    APTB.set_DECJ_lim(args)
    APTB.set_F1_lim(args, parfile)
    APTB.set_F2_lim(args, parfile)

    # this sets the maxiter argument for f.fit_toas for fittings done within the while loop
    # (default to 1)
    maxiter_while = args.maxiter_while

    mask_list, starting_cluster_list = APTB.starting_points(
        toas,
        args,
        score_function="original_different_power",
        start_type=start_type,
        start=start,
    )

    if args.multiprocessing:
        from multiprocessing import Pool

        log.error(
            "DeprecationWarning: This program's use of multiprocessing will likely be revisited in the future\n."
            + "Please be careful in your use of it as it could use more CPUs than you expect..."
        )
        log.warning(
            "Multiprocessing in use. Note: using multiprocessing can make\n"
            + "it difficult to diagnose errors for a particular starting cluster."
        )
        p = Pool(len(mask_list))
        results = []
        solution_trees = []
        for mask_number, mask in enumerate(mask_list):
            print(f"mask_number = {mask_number}")
            starting_cluster = starting_cluster_list[mask_number]
            solution_trees.append(APTB.CustomTree(node_class=APTB.CustomNode))
            solution_trees[-1].create_node("Root", "Root", data=APTB.NodeData())
            results.append(
                p.apply_async(
                    APTB.main_for_loop,
                    (
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
                        time.monotonic(),
                        solution_trees[-1],
                        start_time,
                    ),
                )
            )
        # these two lines make the program wait until each process has concluded
        p.close()
        p.join()
        print("Processes joined, looping through results:")
        for i, r in enumerate(results):
            print(f"\ni = {i}")
            print(r)
            try:
                print(r.get())
            except Exception as e:
                print(e)
        print()
        for solution_tree in solution_trees:
            print(solution_tree.blueprint)
            skeleton_tree = APTB_extension.skeleton_tree_creator(
                solution_tree.blueprint
            )
            skeleton_tree.show()
            skeleton_tree.save2file(
                solution_tree.save_location / Path("solution_tree.tree")
            )
            print(f"tree depth = {skeleton_tree.depth()}")

    else:
        for mask_number, mask in enumerate(mask_list):
            starting_cluster = starting_cluster_list[mask_number]
            solution_tree = APTB.CustomTree(node_class=APTB.CustomNode)
            solution_tree.create_node("Root", "Root", data=APTB.NodeData())
            mask_start_time = time.monotonic()
            try:
                result = APTB.main_for_loop(
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
                    mask_start_time,
                    solution_tree,
                    start_time,
                )
                if result == "success" and not args.try_all_masks:
                    break
            # usually handling ANY exception is frowned upon but if a mask
            # fails, it could be for a number of reasons which do not
            # preclude the success of other masks
            except Exception as e:
                if args.debug_mode:
                    raise e
                print(f"\n{e}\n")
                print(
                    f"mask_number {mask_number} (cluster {starting_cluster}) failed,\n"
                    + "moving onto the next cluster."
                )
            finally:
                print(solution_tree.blueprint, "\n")
                # print(solution_tree.explored_blueprint)
                skeleton_tree, normal_bp_string = APTB_extension.skeleton_tree_creator(
                    solution_tree.blueprint, blueprint_string=True
                )
                (
                    explored_tree,
                    explored_bp_string,
                ) = APTB_extension.skeleton_tree_creator(
                    solution_tree.blueprint,
                    solution_tree.node_iteration_dict,
                    blueprint_string=True,
                )
                tree_graph_fig = APTB_extension.solution_tree_grapher(
                    skeleton_tree, solution_tree.blueprint
                )
                tree_graph_fig.savefig(
                    solution_tree.save_location / Path("solution_tree_schematic.jpg"),
                    bbox_inches="tight",
                )

                depth = skeleton_tree.depth()

                skeleton_tree.show()
                tree_file_name = solution_tree.save_location / Path(
                    "solution_tree.tree"
                )
                explored_tree_file_name = solution_tree.save_location / Path(
                    "explored_solution_tree.tree"
                )
                skeleton_tree.save2file(tree_file_name)
                explored_tree.save2file(explored_tree_file_name)

                with open(tree_file_name, "a") as file:
                    file.write("\n")
                    # file.write(str(solution_tree.blueprint))
                    file.write(
                        "The following allows for the reconstruction of the tree:\n"
                    )
                    file.write(normal_bp_string)
                with open(explored_tree_file_name, "a") as file:
                    file.write("\n")
                    # file.write(str(solution_tree.node_iteration_dict))
                    file.write(
                        "The following allows for the reconstruction of the tree:\n"
                    )
                    file.write(f"g = {solution_tree.g}\n")
                    file.write(f"G = {solution_tree.G}\n")
                    file.write(explored_bp_string)

                print(f"tree depth = {depth}")
                mask_end_time = time.monotonic()
                print(
                    f"Mask time: {round(mask_end_time - mask_start_time, 1)} seconds, or {round((mask_end_time - mask_start_time) / 60.0, 2)} minutes"
                )

            # a successful run will prevent others from running if args.try_all_masks is the default
    return "Completed"
