import sys
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict, Tuple, List


class SetupFitConfig(TypedDict, total=False):
    """Typed view of the setup_fit configuration file."""
    sim: str
    narrow: bool
    halos: bool
    output_folder: str
    zbin: str
    z_eff_Vol_js: str
    poly_order: int
    scalerange: float
    LP_guess: float
    delta_rN: float
    b10Fid: float
    N: int
    n_realizations: int
    cholesky: bool


def _default_output_folder() -> Path:
    """Resolve default output folder path: <repo_root>/outputs/setup_fit."""
    return Path(__file__).resolve().parents[3] / "outputs" / "setup_fit"


def _resolve_output_folder(conf: SetupFitConfig) -> Path:
    out = conf.get("output_folder")
    return Path(out) if out else _default_output_folder()


def _extract_zbin_info(conf: SetupFitConfig) -> Tuple[float, float]:
    """Return (zRed, z_bin_vol) given config's zbin and z_eff_Vol_js.

    Exits with code 1 on invalid config structure.
    """
    from lp_utils.utils import read_json

    zbin = conf["zbin"]
    z_and_V = read_json(conf["z_eff_Vol_js"])
    try:
        z_red = z_and_V[zbin][0]
        z_bin_vol = z_and_V[zbin][1]
    except (KeyError, IndexError, TypeError) as e:
        logging.error("Could not access zbin '%s' or its elements in z_and_V. Details: %s", zbin, e)
        sys.exit(1)
    return float(z_red), float(z_bin_vol)


def _build_shell_centers(lp_guess: float, scalerange: float, delta_rN: float):
    import numpy as np

    sshift = scalerange / 2.0
    r_start = lp_guess - sshift
    r_end = lp_guess + sshift
    ri_arr = np.arange(r_start + delta_rN / 2, r_end + 1e-8, delta_rN)
    return ri_arr, sshift


def _compute_covariance(cosmo, ri_arr, delta_rN: float, z_red: float, ng: float, b10: float):
    import numpy as np
    from lp_utils.lp_analysis import cov_int_2d_vec, cov_noPS_vec

    RI, RJ = np.meshgrid(ri_arr, ri_arr, indexing="ij")
    cov_int = cov_int_2d_vec(cosmo, RI, RJ, z_red, delta_rN, ng, b10)
    cov_nops = cov_noPS_vec(RI, RJ, delta_rN)
    return cov_int + cov_nops / ng**2


def _compute_lp_true(cosmo, z_red: float) -> Tuple[float, List[float]]:
    import numpy as np
    from scipy.interpolate import UnivariateSpline

    s_array = np.logspace(np.log10(25), np.log10(150), 600)
    xiNL = cosmo.Pk2xiNL(s_array, z=z_red)
    dxiNL_ds = np.gradient(xiNL, s_array, edge_order=2)

    spline = UnivariateSpline(s_array, dxiNL_ds, s=0)
    s_zeros = spline.roots()
    if len(s_zeros) < 2:
        logging.warning("Expected at least two zero crossings, found %d. Using fallback LP.", len(s_zeros))
        if len(s_zeros) == 1:
            lp_true = float(s_zeros[0])
        else:
            lp_true = float(0.5 * (s_array[0] + s_array[-1]))
        return lp_true, list(map(float, s_zeros))

    dipNL = s_zeros[0]
    peakNL = s_zeros[1]
    lp_true = 0.5 * (dipNL + peakNL)
    return float(lp_true), list(map(float, s_zeros))


def _as_list(x):
    """Normalize a scalar or list-like config value to a Python list."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _run_scalerange_job(
    cosmo,
    *,
    scalerange: float,
    poly_orders: List[int],
    sim: str,
    narrow: bool,
    halos: bool,
    out_folder: str,
    zbin: str,
    zRed: float,
    z_bin_vol: float,
    LP_guess: float,
    delta_rN: float,
    ng: float,
    b10: float,
    n_realizations: int,
    cholesky: bool,
    LP_true: float,
) -> List[str]:
    """Worker: run all poly_orders for a given scalerange and write outputs.

    Returns list of written file paths.
    """
    from lp_utils.cosmology import Cosmology
    from lp_utils.lp_analysis import (
        create_random_xi_data,
        process_realization,
        aggregate_lp_statistics,
    )
    from lp_utils.utils import write_lp_setup_results_json

    # Build shells and covariance for this scalerange
    riN_arr, sshift = _build_shell_centers(LP_guess, scalerange, delta_rN)
    CovTableVol = _compute_covariance(cosmo, riN_arr, delta_rN, zRed, ng, b10) / z_bin_vol

    # Sample realizations once per scalerange
    xiFid = cosmo.binned_xiNL(s_arr=riN_arr, z=zRed, delta_r=delta_rN, b10=b10)
    xi_distr = create_random_xi_data(
        xi=xiFid, covariance=CovTableVol, n_samples=n_realizations, seed=42
    )

    # Fit windows
    margin_low_dip = sshift
    margin_high_dip = sshift
    margin_low_peak = sshift
    margin_high_peak = sshift
    dip_window = (LP_guess - margin_low_dip, LP_guess + margin_high_dip)
    peak_window = (LP_guess - margin_low_peak, LP_guess + margin_high_peak)

    narrow_str = "narrow" if narrow else "fullsky"
    halos_str = "_halos_" if halos else ""

    written_files: List[str] = []
    for poly_order in poly_orders:
        results = []
        for i in range(n_realizations):
            y_obs = xi_distr.samples[i]
            res = process_realization(
                s=riN_arr,
                y=y_obs,
                covariance=CovTableVol,
                order=poly_order,
                dip_window=dip_window,
                peak_window=peak_window,
                cholesky=cholesky,
            )
            results.append(res)

        agg = aggregate_lp_statistics(results, fiducial_lp=LP_true)

        out_filename = (
            f"{sim}_{narrow_str}{halos_str}zbin_{zbin}_"
            f"binsize{delta_rN}_scalerange{int(scalerange)}_polyorder{poly_order}.json"
        )
        output_path = str(Path(out_folder) / out_filename)
        write_lp_setup_results_json(output_path, agg, results, LP_true=LP_true)
        written_files.append(output_path)

    return written_files


def main():
    """Run LP setup fit pipeline."""
    from lp_utils.utils import read_json
    import argparse

    from lp_utils.cosmology import Cosmology

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # CLI to allow passing input JSON files
    parser = argparse.ArgumentParser(description="Run LP setup fit pipeline")
    parser.add_argument(
        "--conf",
        dest="conf_js",
        default="test.json",
        help="Filename of setup_fit config JSON inside folder config/setup_fit",
    )

    args, _ = parser.parse_known_args()

    conf_js = args.conf_js

    print(f"Using config file: {conf_js}")
    conf = read_json(conf_js, type="setup_fit")

    sim = conf.get("sim", "Raygal")
    cosmo = Cosmology(preset=sim.lower())

    narrow = conf.get("narrow", True)
    halos = conf.get("halos", False)

    out_folder = _resolve_output_folder(conf)

    zRed, z_bin_vol = _extract_zbin_info(conf)

    poly_order_cfg = conf["poly_order"]
    scalerange_cfg = conf["scalerange"]
    LP_guess = conf["LP_guess"]
    delta_rN = conf["delta_rN"]
    b10 = conf.get("b10Fid", 1.0)
    N = int(conf["N"])  # number of particles in the z-bin

    ng = N / z_bin_vol
    n_realizations = int(conf.get("n_realizations", 100))

    # Compute LP truth once (independent of scalerange and poly order)
    LP_true, s_zeros = _compute_lp_true(cosmo, zRed)
    logging.info("--------------------------------")
    logging.info("Zero crossings of dxi_ds at s = %s", s_zeros)
    logging.info("LP true: %.3f", LP_true)

    # Normalize to lists
    scaleranges = [float(x) for x in _as_list(scalerange_cfg)]
    poly_orders = [int(x) for x in _as_list(poly_order_cfg)]

    # Ensure output dir exists
    out_folder.mkdir(parents=True, exist_ok=True)

    # Run scalerange jobs in parallel; each job handles all poly_orders for that scalerange
    max_workers = min(len(scaleranges), (os.cpu_count() or 1)) if len(scaleranges) > 0 else 1
    logging.info("Launching %d parallel scalerange job(s)", max_workers)

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for scalerange in scaleranges:
            job_kwargs = dict(
                cosmo=cosmo,
                scalerange=scalerange,
                poly_orders=poly_orders,
                sim=sim,
                narrow=narrow,
                halos=halos,
                out_folder=str(out_folder),
                zbin=conf["zbin"],
                zRed=zRed,
                z_bin_vol=z_bin_vol,
                LP_guess=LP_guess,
                delta_rN=delta_rN,
                ng=ng,
                b10=b10,
                n_realizations=n_realizations,
                cholesky=conf.get("cholesky", False),
                LP_true=LP_true,
            )
            futures.append(pool.submit(_run_scalerange_job, **job_kwargs))

        for fut in as_completed(futures):
            try:
                written = fut.result()
                for w in written:
                    logging.info("Wrote %s", w)
            except Exception as e:
                logging.error("A scalerange job failed: %s", e)


if __name__ == "__main__":
    main()
