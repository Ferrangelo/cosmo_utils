import sys
import logging
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


def main():
    """Run LP setup fit pipeline."""
    from lp_utils.utils import read_json, write_lp_setup_results_json
    import numpy as np
    import argparse
    from lp_utils.lp_analysis import (
        create_random_xi_data,
        process_realization,
        aggregate_lp_statistics,
        RealizationResult,
    )

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

    narrow_str = "narrow" if narrow else "fullsky"
    halos_str = "_halos_" if halos else ""
    out_folder.mkdir(parents=True, exist_ok=True)

    # Loop over scaleranges (reuse samples per scalerange), then poly orders
    for scalerange in scaleranges:
        riN_arr, sshift = _build_shell_centers(LP_guess, scalerange, delta_rN)
        CovTableVol = _compute_covariance(cosmo, riN_arr, delta_rN, zRed, ng, b10) / z_bin_vol

        xiFid = cosmo.binned_xiNL(s_arr=riN_arr, z=zRed, delta_r=delta_rN, b10=b10)
        xi_distr = create_random_xi_data(
            xi=xiFid, covariance=CovTableVol, n_samples=n_realizations, seed=42
        )

        margin_low_dip = sshift
        margin_high_dip = sshift
        margin_low_peak = sshift
        margin_high_peak = sshift
        dip_window = (LP_guess - margin_low_dip, LP_guess + margin_high_dip)
        peak_window = (LP_guess - margin_low_peak, LP_guess + margin_high_peak)

        for poly_order in poly_orders:
            logging.info(
                "Running scalerange=%.3f, poly_order=%d (n_realizations=%d)",
                scalerange,
                poly_order,
                n_realizations,
            )
            chol = conf.get("cholesky", False)
            results: list[RealizationResult] = []
            for i in range(n_realizations):
                y_obs = xi_distr.samples[i]
                res = process_realization(
                    s=riN_arr,
                    y=y_obs,
                    covariance=CovTableVol,
                    order=poly_order,
                    dip_window=dip_window,
                    peak_window=peak_window,
                    cholesky=chol,
                )
                results.append(res)

            n_fail = sum(1 for r in results if not r.success)
            logging.info("Processed %d realizations. Failures: %d", n_realizations, n_fail)

            # Aggregate statistics over successful realizations
            agg = aggregate_lp_statistics(results, fiducial_lp=LP_true)
            logging.info("--------------------------------")
            logging.info(
                "Total: %d  Success: %d  Fail rate: %.3f",
                agg.n_total,
                agg.n_success,
                agg.fail_rate,
            )
            logging.info(
                "Mean LP: %.4f  Std LP: %.4f  Mean sigma_LP: %.4f",
                agg.mean_lp,
                agg.std_lp,
                agg.mean_sigma_lp,
            )
            logging.info("Bias / sample std (if fiducial provided): %s", str(agg.bias_over_sigma))
            logging.info("Reduced chi^2 (mean over fits): %.3f", agg.chi2_reduced_mean)
            logging.info("KS normality p-value (LP standardized): %s", str(agg.ks_lp_pvalue))
            logging.info("KS chi2 p-value (LP standardized): %s", str(agg.ks_chi2_pvalue))

            if not np.isnan(agg.std_lp) and not np.isnan(agg.mean_sigma_lp):
                ratio = agg.std_lp / agg.mean_sigma_lp
                logging.info("Std of LP distr / mean of analytic sigmas: %.3f", ratio)

            out_filename = (
                f"{sim}_{narrow_str}{halos_str}zbin_{conf['zbin']}_"
                f"binsize{delta_rN}_scalerange{int(scalerange)}_polyorder{poly_order}.json"
            )
            output_path = out_folder / out_filename
            written = write_lp_setup_results_json(str(output_path), agg, results, LP_true=LP_true)
            logging.info("--------------------------------")
            logging.info("Results written to %s", written)


if __name__ == "__main__":
    main()
