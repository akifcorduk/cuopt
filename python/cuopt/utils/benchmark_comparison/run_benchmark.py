# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from multiprocessing import Process

import bks_vehicle_counts
import pandas as pd
import xlsxwriter


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def initial_population_path(args, file):
    return os.path.join(
        args.initial_population_dir, os.path.splitext(file)[0].lower()
    )


def run_single(file, row, gpu, worksheet, args):
    import cudf
    from scipy.spatial.distance import cdist

    import cuopt
    import cuopt.routing.utils as utils

    if args.initial_population_dir:
        os.environ["CUOPT_SOL_INPUT_DIR"] = initial_population_path(args, file)
    else:
        os.environ["CUOPT_SOL_INPUT_DIR"] = os.path.join(
            "/acoerduek-lustre/solution_dir", file
        )
    os.environ["CUOPT_BKS_INPUT_DIR"] = os.path.join(
        "/acoerduek-lustre/bks_dir/", file
    )

    _, extension = os.path.splitext(file)
    if extension == ".pdptw":
        is_pdp = True
    else:
        is_pdp = False
    abs_path = os.path.join(args.dataset_path, file)

    # FIXME use order loc is broken
    use_order_loc = False
    print(f"running: {file} on gpu:{gpu}")
    if extension == ".tsp":
        fixed_route = False
        df = utils.create_from_file_tsp(abs_path)
    else:
        fixed_route = not args.route_minimization_only
        df, vehicle_capacity, fleet_size = utils.create_from_file(
            abs_path, is_pdp
        )
    nodes = df["vertex"].shape[0]
    n_orders = nodes

    if fixed_route:
        bks_vehicle_count = bks_vehicle_counts.bks_counts[file]
        fleet_size = bks_vehicle_count
    elif extension == ".tsp":
        fleet_size = 1

    if use_order_loc:
        n_orders -= 1

    d = cuopt.routing.DataModel(nodes, fleet_size, n_orders)
    coordinates = df[["xcord", "ycord"]].to_pandas().to_numpy()
    matrix = cdist(coordinates, coordinates)

    d.add_cost_matrix(matrix.astype("float32"))
    if args.travel_speed != 1.0:
        d.add_transit_time_matrix(
            (matrix / args.travel_speed).astype("float32")
        )
    del matrix
    if extension != ".tsp":
        utils.fill_demand(df, d, vehicle_capacity, fleet_size, use_order_loc)
        utils.fill_tw(d, df, use_order_loc)
    if is_pdp:
        utils.fill_pdp_index(d, df, use_order_loc)

    if use_order_loc:
        loc_list = list(range(1, n_orders + 1))
        d.set_order_locations(cudf.Series(loc_list))

    s = cuopt.routing.SolverSettings()

    if fixed_route:
        # this is used as target vehicle count
        d.set_min_vehicles(bks_vehicle_count)
    elif extension == ".tsp":
        d.set_min_vehicles(1)

    s.set_time_limit(args.run_time)
    if args.route_minimization_only:
        s.set_verbose_mode(True)
    if args.intermediate_output:
        file_w_gpu = file
        if args.wr_improve:
            file_w_gpu = str(gpu) + file
        intermediate_result_file = os.path.join(
            args.intermediate_dir, file_w_gpu
        )
        s.dump_best_results(intermediate_result_file, 60)

    start = time.perf_counter()
    routing_solution = cuopt.routing.Solve(d, s)
    elapsed = time.perf_counter() - start
    final_cost = routing_solution.get_total_objective()
    vehicle_count = routing_solution.get_vehicle_count()
    status = int(routing_solution.get_status())
    unserviced = len(routing_solution.get_infeasible_orders())
    routing_solution.get_route().to_pandas().to_csv(
        os.path.join(args.out_dir, file + ".routes.csv"), index=False
    )
    result = {
        "instance": file,
        "objective": float(final_cost),
        "vehicles": int(vehicle_count),
        "time_limit": args.run_time,
        "solve_seconds": elapsed,
        "status": status,
        "message": routing_solution.get_message(),
        "unserviced_orders": unserviced,
        "route_minimization_only": args.route_minimization_only,
        "initial_population_dir": os.environ["CUOPT_SOL_INPUT_DIR"]
        if args.initial_population_dir
        else None,
        "travel_speed": args.travel_speed,
    }
    with open(os.path.join(args.out_dir, file + ".json"), "w") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result), flush=True)
    col = 0
    worksheet.write(row, col, file)
    col += 1
    worksheet.write(row, col, float(final_cost))
    col += 1
    worksheet.write(row, col, int(vehicle_count))
    col += 1
    worksheet.write(row, col, args.run_time)
    worksheet.write(row, col + 1, status)
    worksheet.write(row, col + 2, elapsed)
    worksheet.write(row, col + 3, unserviced)
    if status != 0 or unserviced:
        raise RuntimeError(f"{file}: incomplete routing solution: {result}")


def run_chunk(files, proc_id, args):
    import cupy

    cupy.cuda.Device(proc_id).use()
    print(f"run chunk proc id {proc_id} files {files}")
    temp_path = os.path.join(
        args.out_dir, f"benchmark_homberger_{proc_id}.xlsx"
    )
    workbook = xlsxwriter.Workbook(temp_path)
    worksheet = workbook.add_worksheet()
    try:
        for row, file in enumerate(files):
            run_single(file, row, proc_id, worksheet, args)
    finally:
        workbook.close()


def split(a, n):
    k, m = divmod(len(a), n)
    return list(
        a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    )


def main():
    multiprocessing.set_start_method("spawn", force=True)
    cwd = os.getcwd()
    out_dir = os.path.join(cwd, "benchmark_output")

    parser = argparse.ArgumentParser(
        description="A benchmark script for cuOpt"
    )
    parser.add_argument(
        "-g", "--gpu-count", help="Number of GPUs", required=True, type=int
    )
    parser.add_argument(
        "-p",
        "--problem-type",
        help="Problem types. One of cvrptw, pdptw",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--run-time",
        help="Run time per instance",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-bk",
        "--bks-search",
        help="Search with BKS enabled",
        required=False,
        default=False,
        type=str2bool,
    )
    parser.add_argument(
        "-wr",
        "--wr-improve",
        help="Search with BKS enabled",
        required=False,
        default=False,
        type=str2bool,
    )
    parser.add_argument(
        "-l",
        "--only-large",
        help="Run only 800 and 1000 nodes",
        required=False,
        default=False,
        type=str2bool,
    )
    parser.add_argument(
        "-s",
        "--only-small",
        help="Run only 200,400,600 nodes",
        required=False,
        default=False,
        type=str2bool,
    )
    parser.add_argument(
        "-sl",
        "--only-selected",
        help="Run only the selected files",
        required=False,
        default=False,
        type=str2bool,
    )
    parser.add_argument(
        "-i",
        "--intermediate-output",
        help="Dump intermediate outputs every 60 seconds",
        required=False,
        default=False,
        type=str2bool,
    )
    parser.add_argument(
        "-f",
        "--file-count-as-gpu",
        help="Only run GPU number of files",
        required=False,
        default=False,
        type=str2bool,
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        help="Output dir",
        required=False,
        default=out_dir,
        type=str,
    )
    parser.add_argument(
        "-b",
        "--batch-num",
        help="ID of a the Batch of datasets",
        required=False,
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--dataset-path",
        help="Instance directory (overrides the dataset root)",
    )
    parser.add_argument("--files", nargs="+", help="Instance filenames to run")
    parser.add_argument(
        "--travel-speed",
        type=float,
        default=1.0,
        help="Coordinate distance units per time unit (750 for Shanghai)",
    )
    parser.add_argument(
        "--route-minimization-only",
        action="store_true",
        help="Use the full search budget for vehicle minimization; skip cost improvement",
    )
    parser.add_argument(
        "--initial-population-dir",
        help="Start the search from the solutions in <dir>/<lowercase instance stem>; "
        "skips construction and route minimization",
    )

    args = parser.parse_args()
    if args.gpu_count < 1 or args.run_time < 1:
        parser.error("GPU count and run time must be positive")
    if not math.isfinite(args.travel_speed) or args.travel_speed <= 0:
        parser.error("Travel speed must be finite and positive")
    if args.route_minimization_only and args.problem_type == "tsp":
        parser.error("Route minimization requires cvrptw or pdptw instances")
    if args.route_minimization_only:
        os.environ["CUOPT_ROUTE_MINIMIZATION_ONLY"] = "1"
    if args.initial_population_dir:
        if args.route_minimization_only:
            parser.error(
                "Initial population and route minimization only are exclusive"
            )
        os.environ["CUOPT_SOL_FROM_DIR"] = "1"
        os.environ["CUOPT_LOAD_BKS"] = "0"
    n_gpus = args.gpu_count
    bks_search = args.bks_search
    only_big = args.only_large
    only_small = args.only_small
    only_selected = args.only_selected
    file_count_as_gpu = args.file_count_as_gpu
    out_dir = args.out_dir
    batch_num = args.batch_num
    problem_type = args.problem_type

    if (
        problem_type == "cvrptw"
        or problem_type == "pdptw"
        or problem_type == "tsp"
    ):
        if args.dataset_path is None:
            args.dataset_path = os.path.join(
                os.getenv("RAPIDS_DATASET_ROOT_DIR", "../../datasets/"),
                problem_type,
            )
    else:
        sys.exit("Problem type must be one of cvrptw or pdptw!")

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    if args.intermediate_output:
        args.intermediate_dir = os.path.join(
            args.out_dir, "intermediate_results"
        )
        if not os.path.isdir(args.intermediate_dir):
            os.mkdir(args.intermediate_dir)
    files = sorted(os.listdir(args.dataset_path))

    if not problem_type == "tsp":
        if bks_search:
            files = [i for i in files if "_10_" in i]

        if only_big:
            files = [
                i for i in files if "_10_" in i or "_6_" in i or "_8_" in i
            ]

        if only_small:
            files = [i for i in files if "_2_" in i or "_4_" in i]

        if only_selected:
            files = [
                "R1_10_1.TXT",
                "R1_10_2.TXT",
                "R1_10_3.TXT",
                "R1_10_4.TXT",
                "R1_10_5.TXT",
                "R1_10_6.TXT",
                "R1_10_7.TXT",
                "R1_10_8.TXT",
                "R1_10_9.TXT",
                "R1_10_10.TXT",
                "RC1_10_1.TXT",
                "RC1_10_2.TXT",
                "RC1_10_3.TXT",
                "RC1_10_4.TXT",
                "RC1_10_5.TXT",
                "RC1_10_6.TXT",
                "RC1_10_7.TXT",
                "RC1_10_8.TXT",
                "RC1_10_9.TXT",
                "RC1_10_10.TXT",
                "R2_10_1.TXT",
                "R2_10_2.TXT",
                "R2_10_3.TXT",
                "R2_10_4.TXT",
                "R2_10_5.TXT",
                "R2_10_6.TXT",
                "R2_10_7.TXT",
                "R2_10_8.TXT",
                "R2_10_9.TXT",
                "R2_10_10.TXT",
                "RC2_10_1.TXT",
                "RC2_10_2.TXT",
                "RC2_10_3.TXT",
                "RC2_10_4.TXT",
                "RC2_10_5.TXT",
                "RC2_10_6.TXT",
                "RC2_10_7.TXT",
                "RC2_10_8.TXT",
                "RC2_10_9.TXT",
                "RC2_10_10.TXT",
                "C1_10_2.TXT",
                "C1_10_3.TXT",
                "C1_10_6.TXT",
                "C1_10_7.TXT",
                "C1_10_8.TXT",
                "C1_10_9.TXT",
                "C1_10_10.TXT",
                "C2_10_9.TXT",
                "RC1_8_1.TXT",
                "RC1_8_2.TXT",
                "RC1_8_3.TXT",
                "RC1_8_4.TXT",
                "RC1_8_5.TXT",
                "RC1_8_6.TXT",
                "RC1_8_7.TXT",
                "RC1_8_8.TXT",
            ]

        if file_count_as_gpu:
            files = [
                "C1_6_4.TXT",
                "C2_2_10.TXT",
                "R1_2_3.TXT",
                "R2_2_1.TXT",
                "RC1_2_8.TXT",
                "RC2_2_9.TXT",
                "R2_4_8.TXT",
                "R1_4_4.TXT",
            ]
    else:
        files = [
            "a280.tsp",
            "ali535.tsp",
            "ch150.tsp",
            "eil101.tsp",
            "lin318.tsp",
            "dsj1000.tsp",
            "gil262.tsp",
            "kroA200.tsp",
            "rat783.tsp",
            "gr666.tsp",
            "rat575.tsp",
            "tsp225.tsp",
        ]

    if file_count_as_gpu:
        files = [
            "C1_6_4.TXT",
            "C2_2_10.TXT",
            "R1_2_3.TXT",
            "R2_2_1.TXT",
            "RC1_2_8.TXT",
            "RC2_2_9.TXT",
            "R2_4_8.TXT",
            "R1_4_4.TXT",
        ]

    if args.files:
        files = args.files
    if not files or len(set(files)) != len(files):
        parser.error("Select at least one instance, with no duplicates")
    for file in files:
        if os.path.basename(file) != file or not os.path.isfile(
            os.path.join(args.dataset_path, file)
        ):
            parser.error(f"Instance is missing or not a filename: {file}")
        if args.initial_population_dir and not os.path.isdir(
            initial_population_path(args, file)
        ):
            parser.error(
                f"Initial population missing: {initial_population_path(args, file)}"
            )

    num_files = len(files)
    n_gpus = min(n_gpus, num_files)
    if batch_num != -1:
        files_per_process = list(
            split(
                range(batch_num * n_gpus, ((batch_num + 1) * n_gpus)), n_gpus
            )
        )
        last_batch = (num_files // n_gpus) == batch_num
        if last_batch:
            n_files_to_run = num_files % n_gpus
            del files_per_process[n_files_to_run:]
            n_gpus = n_files_to_run
    else:
        files_per_process = list(split(range(num_files), n_gpus))

    # run_chunk(files, 0, args)
    print(files_per_process)
    processes = [
        Process(
            target=run_chunk,
            args=(
                files[
                    files_per_process[x][0] : (files_per_process[x][-1] + 1)
                ],
                x,
                args,
            ),
        )
        for x in range(n_gpus)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    failed = [p.pid for p in processes if p.exitcode != 0]
    if failed:
        sys.exit(
            f"Benchmark workers failed: {failed}; see per-instance results"
        )

    all_data = []
    for file_id in range(n_gpus):
        temp_path = os.path.join(
            args.out_dir, f"benchmark_homberger_{file_id}.xlsx"
        )
        if os.path.exists(temp_path):
            all_data.append(pd.read_excel(temp_path, header=None))

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_excel(os.path.join(args.out_dir, "benchmark_homberger.xlsx"))
    final_df.columns = [
        "instance",
        "objective",
        "vehicles",
        "time_limit",
        "status",
        "solve_seconds",
        "unserviced_orders",
    ]
    final_df.to_csv(
        os.path.join(args.out_dir, "benchmark_results.csv"), index=False
    )
    for file_id in range(n_gpus):
        abs_path = os.path.join(
            args.out_dir, f"benchmark_homberger_{file_id}.xlsx"
        )
        if os.path.exists(abs_path):
            os.remove(abs_path)


if __name__ == "__main__":
    main()
