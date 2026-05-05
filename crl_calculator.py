import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
from scipy.special import erf


# =========================================================================================================
# DATA LOADING
# =========================================================================================================
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


# =========================================================================================================
# INTERPOLATION
# =========================================================================================================
def get_delta_beta(energy_eV, table):
    delta = np.interp(energy_eV, table["Energy(eV)"], table["Delta"])
    beta = np.interp(energy_eV, table["Energy(eV)"], table["Beta"])
    return delta, beta


# =========================================================================================================
# CRL CALCULATIONS
# =========================================================================================================
def wavelength_angstrom(energy_eV):
    return 12398.4 / energy_eV


def absorption_coeff(beta, wavelength_A):
    """Linear absorption coefficient µ in 1/µm"""
    wavelength_um = wavelength_A * 1e-4  # 1 Å = 1e-4 µm
    return 4 * np.pi * beta / wavelength_um


def focal_length(R_um, N, delta):
    """Focal length in µm"""
    return R_um / (2 * N * delta)


def aperture_parameter(a, R0_um, R_um):
    """Standard aperture parameter ap"""
    return a * R0_um ** 2 / (2 * R_um ** 2)


def effective_aperture(R0_um, ap):
    """Effective aperture diameter Deff in µm"""
    if ap < 1e-10:
        return 2 * R0_um
    return 2 * R0_um * np.sqrt((1 - np.exp(-ap)) / ap)


def aperture_averaged_transmission(N, mu, d_neck_um, ap):
    """Peak transmission Tp"""
    if ap < 1e-10:
        return np.exp(-N * mu * d_neck_um)
    return np.exp(-N * mu * d_neck_um) * (1 / (2 * ap)) * (1 - np.exp(-2 * ap))


def gaussian_beam_transmission(N, mu, d_neck_um, a, R_um, sigma_h_um, sigma_v_um):
    """Beam-weighted transmission for an elliptical Gaussian beam.
    sigma_h, sigma_v are the rms (1-sigma) beam sizes in um.
    Exact when the beam is narrow compared to the geometric aperture R0."""
    T_neck = np.exp(-N * mu * d_neck_um)
    factor_h = 1.0 + 2.0 * a * sigma_h_um ** 2 / R_um ** 2
    factor_v = 1.0 + 2.0 * a * sigma_v_um ** 2 / R_um ** 2
    return T_neck / np.sqrt(factor_h * factor_v)


def tophat_beam_transmission(N, mu, d_neck_um, a, R_um, H_um, V_um, R0_um):
    """Beam-weighted transmission for a uniform rectangular (slit-cut) beam.
    H_um, V_um are full widths in um. Clamped to the geometric aperture 2*R0."""
    T_neck = np.exp(-N * mu * d_neck_um)
    H_eff = min(H_um, 2.0 * R0_um)
    V_eff = min(V_um, 2.0 * R0_um)
    u_h = H_eff * np.sqrt(a) / (2.0 * R_um)
    u_v = V_eff * np.sqrt(a) / (2.0 * R_um)

    def _F(u):
        if u < 1e-6:
            return 1.0 - u ** 2 / 3.0  # Taylor expansion for small u
        return (np.sqrt(np.pi) / (2.0 * u)) * erf(u)

    return T_neck * _F(u_h) * _F(u_v)


def calc_2R0_optical(R_um, W_um=1000, d_neck_um=30):
    """Optical aperture 2×R₀ from geometry"""
    return 2 * np.sqrt(R_um * (W_um - d_neck_um))


def image_distance(f_um, L1_um):
    """Image distance from thin lens equation: 1/L2 = 1/f - 1/L1"""
    inv_L2 = 1 / f_um - 1 / L1_um
    if inv_L2 <= 0:
        return float('inf')  # Virtual image or at infinity
    return 1 / inv_L2


def calc_gain(Tp, aperture_2R0_um, Bh_um, Bv_um):
    """Gain = Tp × (2R₀)² / (Bh × Bv)"""
    if Bh_um <= 0 or Bv_um <= 0:
        return 0
    return Tp * (aperture_2R0_um ** 2) / (Bh_um * Bv_um)


def calc_N_from_focal(R_um, f_um, delta):
    """Inverse: calculate N from desired focal length"""
    return R_um / (2 * f_um * delta)


def lens_parameter_a(mu, N, R_um, delta, wavelength_A, sigma_um=0.1):
    wavelength_um = wavelength_A * 1e-4
    scatter_term = 2 * N * (2 * np.pi * delta / wavelength_um) ** 2 * sigma_um ** 2
    return mu * N * R_um + scatter_term


def to_sigma(size_um, convention):
    """Convert a user-supplied 'beam size' to the Gaussian rms sigma."""
    if convention == "FWHM":
        return size_um / 2.3548
    if convention == "1-sigma (rms)":
        return size_um
    if convention == "2-sigma":
        return size_um / 2.0
    if convention == "1/e² full width":
        return size_um / 4.0
    raise ValueError(f"Not a Gaussian convention: {convention}")


def effective_transmission(convention, N, mu, d_neck_um, a, R_um, R0_um,
                           beam_h_um, beam_v_um):
    if convention == "Slit full width (top-hat)":
        return tophat_beam_transmission(
            N, mu, d_neck_um, a, R_um, beam_h_um, beam_v_um, R0_um
        )
    elif convention in ("FWHM", "1-sigma (rms)", "2-sigma", "1/e² full width"):
        sigma_h = to_sigma(beam_h_um, convention)
        sigma_v = to_sigma(beam_v_um, convention)
        return gaussian_beam_transmission(
            N, mu, d_neck_um, a, R_um, sigma_h, sigma_v
        )
    else:  # "Overfilled" fallback
        ap_geom = a * R0_um ** 2 / (2.0 * R_um ** 2)
        return aperture_averaged_transmission(N, mu, d_neck_um, ap_geom)


def beam_overfills_aperture(beam_h_um, beam_v_um, convention, R0_um):
    """Return True when the beam footprint exceeds the geometric aperture 2*R0 in both axes."""
    aperture = 2.0 * R0_um
    if convention == "Slit full width (top-hat)":
        return beam_h_um >= aperture and beam_v_um >= aperture
    # Gaussian conventions: convert to FWHM for comparison
    sigma_h = to_sigma(beam_h_um, convention)
    sigma_v = to_sigma(beam_v_um, convention)
    fwhm_h = 2.3548 * sigma_h
    fwhm_v = 2.3548 * sigma_v
    return fwhm_h >= aperture and fwhm_v >= aperture


def sidebar_divider(thickness=1):
    st.sidebar.markdown(
        f'<hr style="border:none; border-top:{thickness}px solid rgba(7,78,110,1); margin:0.75rem 0;">',
        unsafe_allow_html=True
    )


# =========================================================================================================
# STREAMLIT UI
# =========================================================================================================
st.set_page_config(
    page_title="CRL Calculator",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;600;700&family=Titillium+Web:wght@400;600&display=swap');    

    /* Apply fonts globally */
    [data-testid="stMain"],
    [data-testid="stSidebar"] {
        font-family: 'Titillium Web', sans-serif !important;
    }

    /* Headings and titles */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMetricLabel"],
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        font-family: 'IBM Plex Serif', serif !important;
    }

    /* Main title color */
    h1 {
        color: #7a0041 !important;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
    }


    /* Keep taupe for Results section cards */
    [data-testid="stMetric"] {
        background-color: rgba(174, 163, 162, 0.15);
        padding: 1rem;
        border-radius: 8px;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #9B0052;
        border-color: #9B0052;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #9B0052;
        border-color: #9B0052;
    }


    /* Slider tooltip (floating value above thumb) */
    [data-testid="stThumbValue"],
    .stSlider div[data-testid="stThumbValue"],
    [data-baseweb="slider"] [data-testid="stThumbValue"],
    .stSlider span,
    [data-baseweb="tooltip"] {
        color: #FFFFFF !important;
        background-color: transparent !important;
    }

    /* Even more aggressive - target all text inside slider */
    .stSlider * {
        color: #074E6E !important;
    }

    /* Number inputs in main area - white bg, black text */
    [data-testid="stMain"] .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #6F6764 !important;
    }

    /* Sidebar inputs - keep dark theme */
    [data-testid="stSidebar"] .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #6F6764 !important;
    }

    /* Quotation form text inputs - white bg, black text */
    [data-testid="stSidebar"] .stTextInput input {
        background-color: #FFFFFF !important;
        color: #6F6764 !important;
    }

    /* Info box styling */
    .stAlert [data-testid="stAlertContentInfo"] {
        color: #9B0052 !important;
    }
    [data-testid="stAlert"] {
        background-color: rgba(155, 0, 82, 0.1) !important;
        border-color: #9B0052 !important;
    }
    .stAlert svg {
        fill: #9B0052 !important;
    }

    /* Main area - dark text for light background */

    [data-testid="stMain"] .stMarkdown,
    [data-testid="stMain"] .stText,
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stMetricValue"],
    [data-testid="stMain"] [data-testid="stMetricLabel"] {
        color: #6F6764 !important;
    }

    /* Keep main title magenta */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] label{
        color: #074E6E !important;
    }

    /* Sidebar - white text */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {
        color: #074E6E !important;
    }

    /* Tab text - navy blue */
    [data-testid="stTabs"] button {
        color: #074E6E !important;
    }

    /* Dropdown container and menu */
    [data-baseweb="popover"] {
        background-color: #FFFFFF !important;
    }

    [data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }

    /* Navy blue buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #074E6E !important;
        border-color: #074E6E !important;
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #3d4570 !important;
        border-color: #3d4570 !important;
        color: #FFFFFF !important;
    }

    /* Keep primary button magenta */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: #9B0052 !important;
        border-color: #9B0052 !important;
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: #9B0052 !important;
        border-color: #9B0052 !important;
        color: #FFFFFF !important;
    }

    /* Sidebar selectbox selected value */
    [data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #6F6764 !important;
    }

    /* Force selectbox text color - targets all possible inner elements */
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #6F6764 !important;
    }

    /* Keep the dropdown arrow white on dark sidebar */
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #6F6764 !important;
    }

    /* Force white text on ALL elements inside sidebar primary button */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] *,
    [data-testid="stSidebar"] .stButton > button[kind="primary"] p,
    [data-testid="stSidebar"] .stButton > button[kind="primary"] span,
    [data-testid="stSidebar"] .stButton > button[kind="primary"] div {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)
st.title("Diamond CRL Calculator")
# =========================================================================================================
# Load data
# =========================================================================================================
lenses = load_lenses()
optical = load_optical_constants()

# =========================================================================================================
# SESSION STATE INIT
# =========================================================================================================
if "energy" not in st.session_state:
    st.session_state.energy = 12.0

# =========================================================================================================
# SIDEBAR: BEAM PARAMETERS
# =========================================================================================================

st.sidebar.header("Beam Parameters")
st.sidebar.write("**Energy (keV)**")
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    new_val = st.slider("Energy", 5.0, 30.0, st.session_state.energy, 0.1)
with col2:
    new_val2 = st.number_input("keV", 5.0, 30.0, st.session_state.energy, 0.1)

# Whichever changed, use it
if new_val != st.session_state.energy:
    st.session_state.energy = new_val
    st.rerun()
if new_val2 != st.session_state.energy:
    st.session_state.energy = new_val2
    st.rerun()

energy_keV = st.session_state.energy
energy_eV = energy_keV * 1000

sidebar_divider(2)

# Source size
st.sidebar.write("**Source size (σ, µm)**")
col1, col2 = st.sidebar.columns(2)
with col1:
    beam_h = st.number_input("Horizontal", 1.0, 5000.0, 150.0, 10.0, key="beam_h")
with col2:
    beam_v = st.number_input("Vertical", 1.0, 5000.0, 150.0, 10.0, key="beam_v")

size_convention = st.sidebar.selectbox(
    "Size definition",
    ["FWHM", "Slit full width (top-hat)"],
    index=0,
    help=(
        "How the beam size is defined. FWHM is the most common for beamline diagnostics. "
        "1-sigma is standard in accelerator physics. Slit full width means hard-edged "
        "rectangular beam clipped by upstream slits."
    ),
)

# Distance
L1 = st.sidebar.number_input("Distance source → lens (m)", 1.0, 200.0, 40.0, 1.0, key="L1")

sidebar_divider(2)

st.sidebar.write("**Source divergence (σ', µrad) — optional**")
col1, col2 = st.sidebar.columns(2)
with col1:
    divh = st.number_input("Horizontal", 0.0, 500.0, 0.0, 1.0, key="divh")
with col2:
    divv = st.number_input("Vertical", 0.0, 500.0, 0.0, 1.0, key="divv")

# When user hasn't entered a beam size, auto-estimate from source + divergence
sigma_h_from_source = np.sqrt(beam_h ** 2 + (divh * 1e-6 * L1 * 1e6) ** 2)  # µm
sigma_v_from_source = np.sqrt(beam_v ** 2 + (divv * 1e-6 * L1 * 1e6) ** 2)

# Get optical constants
delta, beta = get_delta_beta(energy_eV, optical)
wavelength_A = wavelength_angstrom(energy_eV)
mu = absorption_coeff(beta, wavelength_A)

st.sidebar.markdown(f"**δ** = {delta:.2e}  \n**β** = {beta:.2e}  \n**µ** = {mu:.2e} /µm")

# =========================================================================================================
# TABS
# =========================================================================================================

tab1, tab2 = st.tabs(["Calculate Lenses", "Select Lenses"])

# =========================================================================================================
# TAB 1: SELECT LENSES (FORWARD CALCULATOR)
# =========================================================================================================

with tab2:
    st.header("Select Lenses")
    # Custom lens input (internal use)
    with st.expander("➕ Add Custom Lens", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            custom_R = st.number_input("Radius R (µm)", 10, 2000, 100, 10, key="custom_R")
        with col2:
            custom_aperture = st.number_input("Aperture (µm)", 50, 3000, 500, 50, key="custom_aperture")
        with col3:
            custom_n = st.number_input("Quantity", 0, 200, 0, key="custom_n")

    selected = {}
    # Add custom lens to selection if quantity > 0
    if custom_n > 0:
        selected["Custom"] = {
            "n": custom_n,
            "Lens": "Custom",
            "Radius": custom_R,
            "Aperture": custom_aperture,
            "R0": custom_aperture / 2,
            "d_neck": 30
        }

    cols = st.columns(2)
    for i, (_,
            lens) in enumerate(lenses.iterrows()):
        with cols[i % 2]:
            st.markdown(
                f"<span style='font-size:1.1em; font-weight:bold; color:#6F6764;'>{lens['Lens']}</span> "
                f"<span style='color:#6F6764;'>(R={lens['Radius']}µm, Aperture={int(lens['Aperture'])}µm)</span>",
                unsafe_allow_html=True
            )
            n = st.number_input(
                f"{lens['Lens']}",
                min_value=0, max_value=100, value=0,
                key=lens['Lens'],
                label_visibility="collapsed"
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

        # Image distance (L2)
        L1_um = L1 * 1e6  # convert m to µm
        L2_um = image_distance(f_total_um, L1_um)
        L2_m = L2_um * 1e-6

        # Image size
        Bh = beam_h * L2_um / L1_um  # µm
        Bv = beam_v * L2_um / L1_um  # µm

        # Optical aperture (use smallest R in stack)
        # TODO: add surface roughness for (un-)polished: 0.25 um / 0.05 um; add check mark
        a = lens_parameter_a(mu, total_N, min_R, delta, wavelength_A)
        ap = aperture_parameter(a, min_R0, min_R)
        Deff = effective_aperture(min_R0, ap)
        Tp_Deff = aperture_averaged_transmission(total_N, mu, 30, ap)
        Tp_neck = aperture_averaged_transmission(total_N, mu, 30, 0)

        # Decide which scenario transmission to show
        overfilled = beam_overfills_aperture(beam_h, beam_v, size_convention, min_R0)
        if overfilled:
            Tp_scenario = Tp_Deff
            scenario_label = "Integrated Transmission (aperture limited)"
        else:
            Tp_scenario = effective_transmission(
                size_convention, total_N, mu, 30, a, min_R, min_R0,
                beam_h, beam_v
            )
            scenario_label = "Integrated Transmission (beam limited)"

        G = calc_gain(Tp_scenario, min_R0, Bh, Bv)

        # Display

        col1, col2 = st.columns(2)
        col1.metric("Focal length", f"{f_total_m:.3f} m")
        col1.metric("Total lenses N", total_N)
        col1.metric("Transmission at Lens Center", f"{Tp_neck * 100:.1f}%")
        col2.metric("Effective aperture", f"{Deff:.0f} µm")
        col2.metric(scenario_label, f"{Tp_scenario * 100:.1f}%")
        # col1.metric("Gain", f"{G:.1f}")
    else:
        # Subtle inline note for validation
        st.markdown(
            '<p style="color: #6F6764; font-style: italic; margin: 2rem 0;">'
            '↑ Select at least one lens above to see results.</p>',
            unsafe_allow_html=True
        )
        # More prominent CTA box for contact
        st.markdown(
            '<div style="background: rgba(155,0,82,0.08); border-left: 3px solid #9B0052; '
            'padding: 1rem; margin-top: 1rem;">'
            'Need a custom configuration? <a href="mailto:info@jjxray.dk?subject=Custom%20Diamond%20CRL%20Inquiry" '
            'style="color: #9B0052; font-weight: 600;">Contact us</a> directly.'
            '</div>',
            unsafe_allow_html=True
        )

    import urllib.parse

# =========================================================================================================
# TAB 2: CALCULATE LENSES (INVERSE CALCULATOR)
# =========================================================================================================

with tab1:
    st.header("Calculate Number of Lenses")
    st.markdown("*Enter your desired focal length and select a lens type to calculate how many lenses you need.*")

    # Toggle for custom lens
    use_custom = st.checkbox("Use custom lens parameters", key="tab2_use_custom")

    if use_custom:
        col1, col2 = st.columns(2)
        with col1:
            custom_R_tab2 = st.number_input("Radius R (µm)", 10, 2000, 100, 10, key="custom_R_tab2")
        with col2:
            custom_aperture_tab2 = st.number_input("Aperture (µm)", 50, 3000, 500, 50, key="custom_aperture_tab2")

        selected_lens = {
            'Lens': 'Custom',
            'Radius': custom_R_tab2,
            'Aperture': custom_aperture_tab2,
            'R0': custom_aperture_tab2 / 2,
            'd_neck': 30
        }
    else:
        # Lens type selection
        lens_options = {row['Lens']: row for _, row in lenses.iterrows()}
        selected_lens_name = st.selectbox(
            "Lens type",
            options=list(lens_options.keys()),
            format_func=lambda
                x: f"{x} (R={lens_options[x]['Radius']}µm, Aperture={int(lens_options[x]['Aperture'])}µm)"
        )
        selected_lens = lens_options[selected_lens_name]

    # Target focal length input
    target_f_m = st.number_input("Desired focal length (m)", 0.01, 100.0, 1.0, 0.01, key="target_f")
    target_f_um = target_f_m * 1e6

    # Calculate N
    N_exact = calc_N_from_focal(selected_lens['Radius'], target_f_um, delta)
    N_rounded = max(1, round(N_exact))

    # Recalculate actual focal length with rounded N
    f_actual_um = focal_length(selected_lens['Radius'], N_rounded, delta)
    f_actual_m = f_actual_um * 1e-6

    # Calculate transmission and effective aperture for the result
    R_um = selected_lens['Radius']
    R0_um = selected_lens['R0']

    a = lens_parameter_a(mu, N_rounded, R_um, delta, wavelength_A)
    ap = aperture_parameter(a, R0_um, R_um)
    Tp_Deff = aperture_averaged_transmission(N_rounded, mu, 30, ap)
    Tp_neck = aperture_averaged_transmission(N_rounded, mu, 30, 0)
    Deff = effective_aperture(R0_um, ap)

    # Decide which scenario transmission to show
    overfilled = beam_overfills_aperture(beam_h, beam_v, size_convention, R0_um)
    if overfilled:
        Tp_scenario = Tp_Deff
        scenario_label = "Integrated Transmission (aperture limited)"
    else:
        Tp_scenario = effective_transmission(
            size_convention, N_rounded, mu, 30, a, R_um, R0_um,
            beam_h, beam_v
        )
        scenario_label = "Integrated Transmission (beam limited)"

    # Image distance and gain
    L1_um = L1 * 1e6
    L2_um = image_distance(f_actual_um, L1_um)
    Bh = beam_h * L2_um / L1_um
    Bv = beam_v * L2_um / L1_um
    G = calc_gain(Tp_scenario, R0_um, Bh, Bv)

    st.header("Results")

    col1, col2 = st.columns(2)
    col1.metric("Number of lenses (N)", N_rounded)
    col1.metric("Actual focal length", f"{f_actual_m:.3f} m")
    col1.metric("Transmission at Lens Center", f"{Tp_neck * 100:.1f}%")
    col2.metric("Effective aperture", f"{Deff:.0f} µm")
    col2.metric(scenario_label, f"{Tp_scenario * 100:.1f}%")
    # col1.metric("Gain", f"{G:.1f}")

    # Show deviation from target
    deviation_pct = abs(f_actual_m - target_f_m) / target_f_m * 100
    if deviation_pct > 1:
        st.info(f"ℹ️ Actual focal length deviates {deviation_pct:.1f}% from target due to rounding N to integer.")

# =========================================================================================================
# QUOTATION REQUEST
# =========================================================================================================
sidebar_divider(2)

st.sidebar.header("Request Quotation")

quote_name = st.sidebar.text_input("Your name", key="quote_name")
quote_institution = st.sidebar.text_input("Institution", key="quote_institution")
quote_email = st.sidebar.text_input("Your email", key="quote_email")

# Check if lenses are selected (assumes 'selected' dict exists from your lens selection code)
has_lenses = 'selected' in dir() and len(selected) > 0

if st.sidebar.button("Prepare Quotation Email", type="primary"):
    # Validation
    if not quote_name:
        st.sidebar.error("Please enter your name.")
    elif not quote_email:
        st.sidebar.error("Please enter your email.")
    elif not has_lenses:
        st.sidebar.error("Please select at least one lens.")
    else:
        # Build lens summary
        lens_lines = []
        for s in selected.values():
            lens_lines.append(f"  - {s['Lens']} (R={s['Radius']}µm): {s['n']} pcs")
        lens_list = "\n".join(lens_lines)

        # Build results summary (use variables from your calculation section)
        subject = f"Diamond CRL Quotation Request - {quote_name}"
        body = f"""Dear JJ X-Ray team,

I would like to request a quotation for the following diamond CRL lenses:

{lens_list}

Application parameters:
  - Energy: {energy_keV} keV
  - Source size: {beam_h} × {beam_v} µm (H × V)
  - Distance source to lens: {L1} m

Calculated results:
  - Total lenses: {total_N}
  - Focal length: {f_total_m:.3f} m
  - Peak transmission: {Tp_Deff * 100:.1f}%
  - Effective aperture: {Deff:.0f} µm

Contact information:
  - Name: {quote_name}
  - Institution: {quote_institution}
  - Email: {quote_email}

Best regards,
{quote_name}
"""

        # Create mailto link
        mailto_link = (
            f"mailto:info@jjxray.dk"
            f"?cc=jm@jjxray.dk"
            f"&subject={urllib.parse.quote(subject)}"
            f"&body={urllib.parse.quote(body)}"
        )

        st.sidebar.success("Email prepared!")
        st.sidebar.markdown(f"👉 **[Click here to open in your email client]({mailto_link})**")

st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image("logo.png", width=300)
with col3:
    st.image("palm_logo.png", width=300)