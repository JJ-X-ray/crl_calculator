# Diamond CRL Calculator

An interactive **Compound Refractive Lens (CRL)** calculator for diamond X-ray lenses, built with [Streamlit](https://streamlit.io/) and Python. Developed by [JJ X-Ray](https://www.jjxray.dk/).

The app lets users calculate optical performance of diamond CRL stacks across a range of X-ray energies, lens types, and beam conditions.

---

## Features

- **Forward calculator ("Select Lenses")**: Choose lenses from the JJ X-Ray catalog, set beam parameters, and calculate focal length, transmission, and effective aperture.
- **Inverse calculator ("Calculate Lenses")**: Enter a desired focal length and lens type to find how many lenses are needed.
- **Mixed lens stacks**: Combine different lens types; focal lengths are combined via harmonic sum.
- **Beam-weighted transmission**: Supports overfilled, Gaussian (elliptical), and flat-top (slit-cut) beam profiles.
- **Multiple beam size conventions**: FWHM, slit full width (top-hat).
- **Quotation request form**: Prepare and send a pre-filled email to JJ X-Ray with your configuration.

---

## Getting Started

### Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

### Installation

```

pip install -r requirements.txt

```

### Running Locally

```

streamlit run crl_[calculator.py](http://calculator.py)

```

### Project Structure

```

crl_calculator/

├── crl_[calculator.py](http://calculator.py)      # Main Streamlit app
├── lenses.csv             # Lens catalog (semicolon-delimited)
├── diamond_optical.csv    # δ, β vs energy (semicolon-delimited)
├── logo.png               # JJ X-Ray logo
├── palm_logo.png          # Partner logo
├── requirements.txt       # Python dependencies
└── .streamlit/
└── config.toml        # Streamlit theme configuration

```

### Data Files

| File                  | Description                                                         | Format              |
|-----------------------|---------------------------------------------------------------------|---------------------|
| `lenses.csv`          | Lens catalog: Lens name, Radius (µm), Aperture (µm)                 | Semicolon-delimited |
| `diamond_optical.csv` | Diamond optical constants: Energy (eV), Delta, Beta (henke.lbl.gov) | Semicolon-delimited |

---

## Lens Catalog

| Lens         | R (µm) | Aperture (µm) | R₀ (µm) | d_neck (µm) |
|--------------|---------|----------------|----------|--------------|
| R30A180      | 30      | 180            | 90       | 30           |
| R50A300      | 50      | 300            | 150      | 30           |
| R100A448     | 100     | 448            | 224      | 30           |
| R100A626     | 100     | 626            | 313      | 30           |
| R200A885     | 200     | 885            | 442.5    | 30           |
| R300A1084    | 300     | 1084           | 542      | 30           |
| R400A1200    | 400     | 1200           | 600      | 30           |
| R500A1400    | 500     | 1400           | 700      | 30           |
| R1000A1979   | 1000    | 1979           | 989.5    | 30           |
| R1500A2800   | 1500    | 2800           | 1400     | 30           |

Where **R** is the parabolic radius of curvature, **R₀** = Aperture / 2 is the physical half-aperture, and **d_neck** is the minimum material thickness at the center of the lens.

---

## Physics and Formulas

### Optical Constants

The refractive index of diamond at X-ray energies is written as:

$$n = 1 - \delta + i\beta$$

where **δ** (refractive decrement) and **β** (absorption index) are interpolated from the lookup table `diamond_optical.csv` at the user-selected energy.

The **linear absorption coefficient** is:

$$\mu = \frac{4\pi\beta}{\lambda}$$

where λ is the X-ray wavelength in the same units as µ (µm in this code). The wavelength in Ångströms is:

$$\lambda\,[\text{Å}] = \frac{12398.4}{E\,[\text{eV}]}$$

### Focal Length

For a stack of **N** identical bi-concave parabolic lenses with radius of curvature **R**:

$$f = \frac{R}{2N\delta}$$

For **mixed stacks** (different lens types), the combined focal length is the harmonic sum:

$$\frac{1}{f_{\text{total}}} = \sum_i \frac{1}{f_i} = \sum_i \frac{2 N_i \delta}{R_i}$$

### Inverse Calculator

Given a desired focal length *f* and a single lens type with radius *R*, the required number of lenses is:

$$N = \frac{R}{2 f \delta}$$

This is rounded to the nearest integer (minimum 1), and the actual focal length is recalculated.

### Lens Parameter

The dimensionless lens parameter **a** encodes both photoabsorption and surface roughness:

$$a = \mu N R + 2N\left(\frac{2\pi\delta}{\lambda}\right)^2 \sigma_{\text{surf}}^2$$

- The first term (**µNR**) is the photoabsorption/Compton contribution. A ray at height *s* through a parabolic lens traverses material thickness *s²/R* per lens, so at the geometric edge (*s* = *R₀*) the total absorption exponent over *N* lenses is *µNR₀²/R*.
- The second term accounts for **surface roughness** scattering (σ_surf defaults to 0.1 µm). For pressed diamond lenses this term is typically negligible.

### Aperture Parameter

$$a_p = \frac{a \cdot R_0^2}{2R^2}$$

This is the optical depth at the geometric edge of the aperture. Both the effective aperture and the peak transmission are functions of $a_p$.

### Effective Aperture

$$D_{\text{eff}} = 2R_0 \sqrt{\frac{1 - e^{-a_p}}{a_p}}$$

The lens is geometrically open out to *R₀*, but rays far from the axis pass through more material and are exponentially absorbed. $D_{eff}$ is the diameter of a hypothetical perfect top-hat aperture that would transmit the same total flux under uniform illumination.

**Limits:**
- No absorption $(a_p → 0): D_eff → 2R_0$ (full geometric aperture).
- Strong absorption $(a_p ≫ 1): D_eff → 2R₀ / \sqrt{a_p}$ (rim goes dark, useful aperture shrinks).

### Peak Transmission (Overfilled Beam)

$$T_p = e^{-N\mu\,d_{\text{neck}}} \cdot \frac{1 - e^{-2a_p}}{2a_p}$$

Two factors:

1. **Neck loss** $e^{-Nµd_{neck}}$: Every ray passes through the flat neck region regardless of radius. This is a constant, unavoidable cost.
2. **Aperture-averaged parabolic loss** $(1 - e^{-2a_p}) / (2a_p)$: Averaging the Gaussian absorption profile $e^{-2as²/R²}$ over the disk of radius $R_0$. The factor of 2 (vs. 1 in $D_{eff}$) comes from transmission being an intensity, not an amplitude.

**Limits:**
- $a_p → 0: T_p → e^{-Nµd_{neck}}$ (neck loss only).
- $a_p ≫ 1: T_p ≈ e^{-Nµd_{neck}} / (2a_p)$

### Beam-Weighted Transmission

When the beam does **not** overfill the lens, actual transmission depends on the beam profile. The general expression is:

$$T = e^{-N\mu\,d_{\text{neck}}} \cdot \frac{\iint I_{\text{beam}}(x, y)\;e^{-a(x^2+y^2)/R^2}\,dx\,dy}{\iint I_{\text{beam}}(x, y)\,dx\,dy}$$

The lens factor separates as $e^{-ax²/R²} · e^{-ay²/R²}$ due to the rotational symmetry of the parabolic profile, so the 2D integral becomes a product of two 1D integrals for separable beam profiles.

#### Elliptical Gaussian Beam

For a Gaussian intensity profile $I(x,y) = I_0 \cdot exp(-x²/(2σ_h²)) \cdot exp(-y²/(2σ_v²))$:

$$T_{\text{gauss}} = \frac{e^{-N\mu\,d_{\text{neck}}}}{\sqrt{\left(1 + \frac{2a\sigma_h^2}{R^2}\right)\left(1 + \frac{2a\sigma_v^2}{R^2}\right)}}$$

**Limits:**
- Pencil beam $(σ → 0): T → *e^{-Nµd_{neck}}$ (neck loss only).
- Very wide beam (σ ≫ *R₀*): converges toward $T_p$.

#### Flat-Top Rectangular (Slit-Cut) Beam

For a uniform beam of full width *H* × *V*, clamped to the geometric aperture 2*R₀*:

$$T_{\text{tophat}} = e^{-N\mu\,d_{\text{neck}}} \cdot F(u_h) \cdot F(u_v)$$

where

$$F(u) = \frac{\sqrt{\pi}}{2u}\,\text{erf}(u), \quad u_h = \frac{H\sqrt{a}}{2R}, \quad u_v = \frac{V\sqrt{a}}{2R}$$

**Limits:**
- Narrow beam $(u → 0): F(u) ≈ 1 - u²/3$. Approaches neck-only loss.
- Wide beam $(u ≫ 1): F(u) → \sqrt{π} / (2u)$. Most flux is absorbed at the edges.

### Beam Size Conventions

| Convention            | Relation to σ         | Common context                          |
|-----------------------|-----------------------|-----------------------------------------|
| FWHM                  | FWHM ≈ 2.355 σ       | Beamline diagnostics, knife-edge scans  |
| Slit full width       | Not Gaussian; use H,V | Beam defined by upstream slits          |

### Image Distance

Thin-lens equation:

$$\frac{1}{L_2} = \frac{1}{f} - \frac{1}{L_1}$$

where *L₁* is the source-to-lens distance and *L₂* is the lens-to-image distance.

### Summary of Formulas

| Quantity                      | Formula                                                             |
|-------------------------------|---------------------------------------------------------------------|
| Effective aperture $D_{eff}$  | $$2R₀ \sqrt{(1 - e^{-a_p}) / a_p}$$                                 |
| Overfilled transmission $T_p$ | $$e^{-Nµd_neck} · (1 - e^{-2a_p}) / 2a_p$$                          |
| Gaussian beam $T_{gauss}$     | $$e^{-Nµd_neck} / \sqrt{(1 + 2aσ_h²/R²)(1 + 2aσ_v²/R²)}$$           |
| Flat-top beam $T_{tophat}$    | $$e^{-Nµd_neck} · F(u_h) · F(u_v),  F(u) = (\sqrt{π} / 2u) erf(u)$$ |
| Lens center upper bound       | $$e^{-Nµd_neck}$$                                                   |

All formulas share the lens parameter *a* and differ only in how they average the lens absorption profile over the beam footprint.

---

## Configuration

The Streamlit theme is defined in `.streamlit/config.toml` with JJ X-Ray brand colors:

```

[theme]

primaryColor = "#9B0052"

backgroundColor = "#6F6764"

secondaryBackgroundColor = "#2E3458"

textColor = "#FFFFFF"

```

---

## License

Proprietary. Copyright JJ X-Ray A/S. All rights reserved.
```