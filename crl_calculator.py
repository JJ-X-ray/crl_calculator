import streamlit as st
import pandas as pd
import numpy as np


# =====================
# DATA LOADING
# =====================
@st.cache_data
def load_lenses():
    # Adjust path to your CSV
    df = pd.read_csv("lenses.csv", delimiter=';')
    # Columns: Lens, Radius (µm), Aperture (µm)
    df["R0"] = df["Aperture"] / 2  # physical radius in µm
    df["d_neck"] = 30  # µm, constant for all
    return df


@st.cache_data
def load_optical_constants():
    # Columns: Energy(eV), Delta, Beta
    return pd.read_csv("diamond_optical.csv", delimiter=';')


# =====================
# INTERPOLATION
# =====================
def get_delta_beta(energy_eV, table):
    delta = np.interp(energy_eV, table["Energy(eV)"], table["Delta"])
    beta = np.interp(energy_eV, table["Energy(eV)"], table["Beta"])
    return delta, beta


# =====================
# CRL CALCULATIONS
# =====================
def wavelength_angstrom(energy_eV):
    return 12398.4 / energy_eV


def absorption_coeff(beta, wavelength_A):
    """Linear absorption coefficient µ in 1/µm"""
    wavelength_um = wavelength_A * 1e-4
    return 4 * np.pi * beta / wavelength_um


def focal_length(R_um, N, delta):
    """Focal length in µm"""
    return R_um / (2 * N * delta)


def lens_parameter_a(mu, N, R_um, delta, wavelength_A, sigma_um=0.1):
    """Lens parameter a (accounts for absorption + scattering)"""
    scatter_term = 2 * N * (2 * np.pi * delta / wavelength_A) ** 2 * sigma_um ** 2
    return mu * N * R_um + scatter_term


def aperture_parameter(a, R0_um, R_um):
    """Standard aperture parameter ap"""
    return a * R0_um ** 2 / (2 * R_um ** 2)


def effective_aperture(R0_um, ap):
    """Effective aperture diameter Deff in µm"""
    if ap < 1e-10:
        return 2 * R0_um
    return 2 * R0_um * np.sqrt((1 - np.exp(-ap)) / ap)


def peak_transmission(N, mu, d_neck_um, ap):
    """Peak transmission Tp"""
    if ap < 1e-10:
        return np.exp(-N * mu * d_neck_um)
    return np.exp(-N * mu * d_neck_um) * (1 / (2 * ap)) * (1 - np.exp(-2 * ap))


# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="CRL Calculator", layout="centered")
st.title("Diamond CRL Calculator")

# Load data
lenses = load_lenses()
optical = load_optical_constants()

# Sidebar inputs
st.sidebar.header("Parameters")
energy_keV = st.sidebar.slider("Energy (keV)", 5.0, 30.0, 12.0, 0.1)
energy_eV = energy_keV * 1000

# Get optical constants
delta, beta = get_delta_beta(energy_eV, optical)
wavelength_A = wavelength_angstrom(energy_eV)
mu = absorption_coeff(beta, wavelength_A)

st.sidebar.markdown(f"**δ** = {delta:.3e}  \n**β** = {beta:.3e}  \n**µ** = {mu:.4f} /µm")

# Lens selection
st.header("Select Lenses")
selected = {}
cols = st.columns(2)
for i, (_, lens) in enumerate(lenses.iterrows()):
    with cols[i % 2]:
        n = st.number_input(
            f"{lens['Lens']} (R={lens['Radius']}µm)",
            min_value=0, max_value=50, value=0, key=lens['Lens']
        )
        if n > 0:
            selected[lens['Lens']] = {"n": n, **lens}

# Calculate
if selected:
    st.header("Results")

    # Combined calculation for all selected lenses (assuming same R for simplicity)
    # For mixed lenses, you'd sum 1/f contributions

    total_N = sum(s["n"] for s in selected.values())

    # Weighted effective R (harmonic mean for focal length)
    inv_f_total = 0
    for s in selected.values():
        f_single = focal_length(s["Radius"], s["n"], delta)
        inv_f_total += 1 / f_single if f_single > 0 else 0

    f_total_um = 1 / inv_f_total if inv_f_total > 0 else float('inf')
    f_total_m = f_total_um * 1e-6

    # For transmission, use the smallest aperture (limiting)
    min_R0 = min(s["R0"] for s in selected.values())
    min_R = min(s["Radius"] for s in selected.values())

    a = lens_parameter_a(mu, total_N, min_R, delta, wavelength_A)
    ap = aperture_parameter(a, min_R0, min_R)
    Tp = peak_transmission(total_N, mu, 30, ap)
    Deff = effective_aperture(min_R0, ap)

    col1, col2 = st.columns(2)
    col1.metric("Focal length", f"{f_total_m:.3f} m")
    col1.metric("Total lenses N", total_N)
    col2.metric("Peak transmission", f"{Tp * 100:.1f}%")
    col2.metric("Effective aperture", f"{Deff:.0f} µm")
else:
    st.info("Select at least one lens above.")