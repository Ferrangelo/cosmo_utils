import numpy as np
from cosmo_utils.cosmology import (
    Cosmology,
    growth_factor_flat_lcdm,
    d_dz_growth_factor_flat_lcdm,
    growth_rate_flat_lcdm,
    Pk2xi as Pk2xi_global,
    Pk2d_xi_ds as Pk2d_xi_ds_global,
    dxi_ds as dxi_ds_global,
    Asq as Asq_global,
    sigma0sq as sigma0sq_global,
    coefficient_form as coefficient_form_global,
    compute_sigma_v,
    compute_sigma8,
)

def rel_diff(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-15, np.abs(b)))

def main():
    cosmo = Cosmology(preset="raygal")  # Must exist in JSON presets
    Om = cosmo.Omega_m

    k = cosmo.k
    P = cosmo.P

    # Test grids
    z_vals = np.linspace(0.0, 2.0, 7)
    s_vals = np.linspace(1.0, 150.0, 40)

    # 1. Growth factor
    D_class = cosmo.growth_factor(z_vals)
    D_global = growth_factor_flat_lcdm(z_vals, Om)
    print("growth_factor rel diff:", rel_diff(D_class, D_global))

    # 2. dD/dz
    dD_class = cosmo.d_dz_growth_factor(z_vals)
    dD_global = d_dz_growth_factor_flat_lcdm(z_vals, Om)
    print("d_dz_growth_factor rel diff:", rel_diff(dD_class, dD_global))

    # 3. Growth rate
    f_class = cosmo.growth_rate(z_vals)
    f_global = growth_rate_flat_lcdm(z_vals, Om)
    print("growth_rate rel diff:", rel_diff(f_class, f_global))

    # 4. sigma8 consistency (class vs direct computation)
    sig8_class = cosmo.sigma8
    sig8_direct = compute_sigma8(k, P)
    print("sigma8 rel diff:", rel_diff(np.array(sig8_class), np.array(sig8_direct)))

    # 5. sigma_v
    sv_class = cosmo.sigma_v()
    sv_global = compute_sigma_v(k, P)
    print("sigma_v rel diff:", rel_diff(np.array(sv_class), np.array(sv_global)))

    # 6. xi(s)
    xi_class = cosmo.Pk2xi(s_vals, z=0.5)
    xi_global = Pk2xi_global(k, P, s_vals, z=0.5, Om=Om, cosmo_obj=cosmo)
    print("Pk2xi rel diff:", rel_diff(xi_class, xi_global))

    # 7. d xi / ds (analytic integral derivative)
    dxi_class = cosmo.Pk2d_xi_ds(s_vals, z=0.5)
    dxi_global = Pk2d_xi_ds_global(k, P, s_vals, z=0.5, Om=Om, cosmo_obj=cosmo)
    print("Pk2d_xi_ds rel diff:", rel_diff(dxi_class, dxi_global))

    # 8. numerical gradient of xi
    grad_class = cosmo.dxi_ds(s_vals, z=0.5)
    grad_global = dxi_ds_global(k, P, s_vals, z=0.5, Om=Om, cosmo_obj=cosmo)
    print("dxi_ds (gradient) rel diff:", rel_diff(grad_class, grad_global))

    # 9. Asq
    Asq_class = cosmo.Asq(z=0.5, rsd=True, b10=1.7)
    Asq_glob = Asq_global(0.5, Om=Om, b10=1.7, cosmo_obj=cosmo)
    print("Asq rel diff:", rel_diff(np.array(Asq_class), np.array(Asq_glob)))

    # 10. sigma0sq
    sig0_class = cosmo.sigma0sq(z=0.5, rsd=True, b10=1.7, b01=0.2, sigma_p=3.0)
    sig0_glob = sigma0sq_global(
        0.5, k, P, Om=Om, b10=1.7, b01=0.2, sigma_p=3.0, cosmo_obj=cosmo
    )
    print("sigma0sq rel diff:", rel_diff(np.array(sig0_class), np.array(sig0_glob)))

    # 11. coefficient_form
    coeff_class = cosmo.coefficient_form(0.5, rsd=True, b10=1.7, b01=0.2, sigma_p=3.0)
    # Reconstruct global coefficient_form(k, Asq, sigma0sq)
    coeff_glob = coefficient_form_global(k, Asq_glob, sig0_glob)
    print("coefficient_form rel diff:", rel_diff(coeff_class, coeff_glob))

    # Assertions (choose tolerances)
    tol = 1e-10  # very strict; relax if needed
    checks = {
        "growth_factor": rel_diff(D_class, D_global),
        "d_dz_growth_factor": rel_diff(dD_class, dD_global),
        "growth_rate": rel_diff(f_class, f_global),
        "sigma8": rel_diff(np.array(sig8_class), np.array(sig8_direct)),
        "sigma_v": rel_diff(np.array(sv_class), np.array(sv_global)),
        "Pk2xi": rel_diff(xi_class, xi_global),
        "Pk2d_xi_ds": rel_diff(dxi_class, dxi_global),
        "dxi_ds": rel_diff(grad_class, grad_global),
        "Asq": rel_diff(np.array(Asq_class), np.array(Asq_glob)),
        "sigma0sq": rel_diff(np.array(sig0_class), np.array(sig0_glob)),
        "coefficient_form": rel_diff(coeff_class, coeff_glob),
    }

    failed = [name for name, val in checks.items() if val > tol]
    if failed:
        print("\nFAILED (tolerance {:.1e}):".format(tol), ", ".join(failed))
    else:
        print("\nAll comparisons passed within tolerance {:.1e}".format(tol))

if __name__ == "__main__":
    main()