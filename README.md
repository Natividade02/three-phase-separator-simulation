# Three-Phase Separator Simulation

This repository contains a dynamic simulation model for a three-phase separator (oil-water-gas).

## 📄 Related publication

This work is associated with a publication in the XI SIINTEC.

## ⚙️ Model Features

* Dynamic mass balance
* Oil-water-gas separation modeling
* Separator efficiency correlations
* Transient simulation using ODE solver (SciPy)

## 📊 Outputs

The model generates:

* Oil, water, and gas flow rates
* Pressure behavior
* Phase levels
* BS&W (Basic Sediment and Water)
* Residence time calculations
* Graphical results (matplotlib)
* Tabulated results (pandas)

## 🚀 How to run

python three_phase_separator_simulation.py or three_phase_version_POO.py
To version POO, you need to input in terminal the temperature and time of simulation

## 📌 Notes

* Input parameters are defined directly in the code
* Multiple oil types (API) are simulated
* Future versions will include object-oriented implementation and new graphs
