# Three-Phase Separator — Dynamic Simulation Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Publication](https://img.shields.io/badge/Published-%20JBTH-green)](https://doi.org/10.34178/jbth.v9i3.596)

A dynamic simulation model for three-phase oil-water-gas separation, built from first principles using mass balance ODEs and solved with SciPy's RK45 integrator. Developed as part of a research program funded by **ANP/PRH-27.1** and associated with a peer-reviewed publication in the **Journal of Bioengineering, Technologies and Health (2026)**.

---

## 📌 Motivation

Three-phase separators are the central unit operation in primary petroleum treatment. In mature onshore fields — such as those in Brazil's Recôncavo Basin — liquid rates decline over time while water cuts (BS&W) increase, often exceeding 90%. Understanding the **dynamic behavior** of these vessels under varying crude API gravity is critical for:

- Designing control strategies for BS&W reduction
- Sizing separation chambers and weir geometry
- Predicting residence times for regulatory compliance
- Selecting appropriate technologies in multicriteria frameworks

This model provides a **transient simulation** of separator performance across six crude oil types, from asphaltic (12°API) to extra-light (42.5°API).

---

## ⚙️ Physical Model

### System Description

The separator is modeled as a **horizontal three-phase vessel** with two internal chambers:

- **Separation chamber** (length `C_csy = 18.0 m`): receives the inlet stream; oil, water, and gas disengage
- **Oil chamber** (length `C_cly = 3.903 m`): receives liquid overflow from the weir; oil exits via outlet valve

A **weir** at height `h_vert = 2.8 m` controls the interface between chambers.

### State Variables

The ODE system tracks **7 state variables** over time:

| Variable | Description | Unit |
|---|---|---|
| `h_tst` | Total liquid level in separation chamber | m |
| `h_wst` | Water phase level in separation chamber | m |
| `h_lst` | Liquid level in oil chamber | m |
| `P_st` | Separator operating pressure | kgf/cm² |
| `Xlfwcs` | Oil-in-water volumetric fraction (separation chamber) | — |
| `xwflcs` | Water-in-oil volumetric fraction (separation chamber) | — |
| `xwlcl` | Water-in-oil volumetric fraction (oil chamber) — output BS&W | — |

### Governing Equations

**Level dynamics** — derived from volumetric mass balances on circular cross-sections:

For `h_tst < h_vert` (weir not overflowing):

```
dh_tst/dt = (W_e + L_e - Lvy - W_sst) / [2 · C_csy · √(h_tst · (D - h_tst))]
dh_lst/dt = (Lvy - L_sst)              / [2 · C_cly · √(h_lst · (D - h_lst))]
```

For `h_tst ≥ h_vert` (weir overflowing, chambers coupled):

```
dh_tst/dt = dh_lst/dt = (W_e + L_e - L_sst - W_sst) / [2·(C_csy + C_cly)·√(h_tst·(D-h_tst))]
```

**Weir overflow** (Francis-type correlation):

```
Lvy = 110.2046/60 · (L_vert - 0.2·max(h_tst - h_vert, 0)) · max(h_tst - h_vert, 0)^1.5
```

**Outlet valve flow** (orifice-based):

```
L_sst = [CV_max_l · S_l / (0.0693 · 60 · ρ_fl)] · √[(P_st - P_jus)·D_l + γ_l · h_lst · 1e-4]
W_sst = [CV_max_w · S_w / (0.0693 · 60 · ρ_fw)] · √[(P_st - P_jus)·D_w + (γ_w·h_wst + γ_l·(h_tst-h_wst))·1e-4]
G_sst = [CV_max_g · S_g · R · T / (2.832 · 60 · MW_g · P_st)] · √[(P_st + P_comp)·(P_st - P_comp)·D_g]
```

**Pressure dynamics** — from overall volumetric balance on the gas cap:

```
dP_st/dt = [(W_e + L_e + G_e - W_sst - L_sst - G_sst) · P_st] / (V_total - V_sep_chamber - V_oil_chamber)
```

**Separation efficiency** — polynomial correlations for cross-contamination:

```
η_wl = f(h_wst, L_e)    [oil removal efficiency from water phase]
η_lw = g(h_wst, W_e)    [water removal efficiency from oil phase]
```
Coefficients sourced from Model Predictive Control literature (2016).

**Cross-contamination dynamics:**

```
dXlfwcs/dt = [L_e · BSW_in · (1 - η_wl) - Lvy · xwflcs] / (V_sep - V_water)
dxwflcs/dt = [W_e · TOG_in · (1 - η_lw) - W_sst · Xlfwcs] / V_water
dxwlcl/dt  = [Lvy · xwflcs - L_sst · xwlcl] / V_oil_chamber
```

The output **BS&W** is `xwlcl`, representing water fraction in the oil outlet stream.

### Oil Density Model

Oil density is computed from API gravity using the standard correlation:

```
D_l = 141.5 / (API + 131.5)    [relative density, dimensionless]
ρ_l = D_l × 1000               [kg/m³]
```

---

## 🛢️ Simulated Oil Types

| Oil Type | API Gravity | ρ_l (kg/m³) |
|---|---|---|
| Asphaltic | 12 | 986 |
| Extra Heavy | 17 | 953 |
| Heavy | 23 | 916 |
| Medium | 30 | 876 |
| Light | 36.5 | 844 |
| Extra Light | 42.5 | 815 |

---

## 📊 Outputs

The simulation generates, for each oil type:

- Transient profiles of all 7 state variables
- Oil, water, and gas outlet flow rates (m³/h)
- Final BS&W at the oil outlet (%)
- Phase residence times (min) — oil and water chambers
- Pressure–BS&W cross-plot
- Summary table via `pandas` DataFrame

### Example Result (Medium Crude, 30°API)

| Metric | Value |
|---|---|
| Final BS&W | ~0.3% |
| Final Total Level | ~2.5 m |
| Final Pressure | ~15 kgf/cm² |
| Oil Residence Time | ~10 min |
| Water Residence Time | ~8 min |

---

## 🚀 How to Run

### Requirements

```
Python >= 3.8
numpy
scipy
matplotlib
pandas
```

Install dependencies:

```bash
pip install numpy scipy matplotlib pandas
```

### Run procedural version

```bash
python three_phase_separator_simulation.py
```

### Run OOP version (interactive input)

```bash
python three_phase_version_POO.py
```
The OOP version prompts for temperature (K) and simulation time (s) in the terminal.

### Key input parameters

All inputs are defined at the top of the script. The most relevant ones to modify:

| Parameter | Description | Default |
|---|---|---|
| `W_e` | Inlet water flow rate (m³/s) | 0.184 |
| `L_e` | Inlet oil flow rate (m³/s) | 6.006e-3 |
| `G_e` | Inlet gas flow rate (m³/s) | 7.182e-2 |
| `BSW_eflw` | Inlet water-in-oil fraction | 0.02 |
| `TOG_eflw` | Inlet oil-in-water fraction | 3.013e-3 |
| `T_st` | Operating temperature (K) | 304 |
| `P_st0` | Initial pressure (kgf/cm²) | 15 |
| `D` | Vessel diameter (m) | 3.048 |
| `L` | Vessel length (m) | 23.503 |

---

## 🔬 Numerical Method

The ODE system is integrated using **`scipy.integrate.solve_ivp`** with the **Runge-Kutta 45** method:

- Relative tolerance: `rtol = 1e-6`
- Absolute tolerance: `atol = 1e-6`
- Time span: 0 to 2000 s (adjustable)
- Evaluation points: 200,000 (uniform)

Conditional logic handles the weir transition and prevents negative arguments in square root expressions (`np.maximum(..., 0)`).

---

## 📄 Related Publication

If you use this code in your work, please cite:

> NATIVIDADE, M. C.; MIRRE, R. C.; CAMPOS, I. O. F. **A Dynamic Simulation Framework for Performance Analysis of a Three-Phase Separator.** *Journal of Bioengineering, Technologies and Health*, v. 9, p. 237–246, 2026.

```bibtex
@article{natividade2026,
  author  = {Natividade, Michel Cardoso and Mirre, Reinaldo Coelho and Campos, Igor Oliveira de Freitas},
  title   = {A Dynamic Simulation Framework for Performance Analysis of a Three-Phase Separator},
  journal = {Journal of Bioengineering, Technologies and Health},
  volume  = {9},
  pages   = {237--246},
  year    = {2026}
}
```

---

## 🗺️ Roadmap

- [ ] Jupyter Notebook with interactive example and inline plots
- [ ] Parametric sensitivity analysis module (API, inlet BSW, valve opening)
- [ ] `requirements.txt` for easy environment setup
- [ ] Validation against Aspen HYSYS steady-state results
- [ ] Streamlit dashboard for real-time parameter exploration

---

## 👤 Author

**Michel Cardoso Natividade**
Chemical Engineering — SENAI CIMATEC, Salvador, Bahia, Brazil
ANP/PRH-27.1 Research Fellow | Process Modeling & Simulation | Oil & Gas Primary Treatment

[![LinkedIn](https://img.shields.io/badge/LinkedIn-michelnatividade-blue)](https://www.linkedin.com/in/michelnatividade/)
[![Lattes](https://img.shields.io/badge/Lattes-CNPq-orange)](http://lattes.cnpq.br/6493246851690351)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Profile-teal)](https://www.researchgate.net/profile/Michel-Natividade/research)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
