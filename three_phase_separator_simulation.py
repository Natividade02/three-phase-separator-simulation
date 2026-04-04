import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Ellipse, Polygon
import pandas as pd

oils = {
    "asphaltic": 12,
    "Extra Heavy": 17,
    "Heavy": 23,
    "Medium": 30,
    "Light": 36.5,
    "Extra Light": 42.5
}
results = {}

# ======================
# INPUT PARAMETERS
# ======================

    # Opening valve Factor
S_w, S_l, S_g = 0.79, 0.305, 0.5

    # Dimensions valves

CV_max_w, CV_max_l, CV_max_g = 245.11, 19.98, 25

    # Densities

D_w, D_g = 0.965, 0.565 # (filgueiras, 2005)
MW_g = 16.48 # kg/kmol
R = 0.0821 # Gas constant (L·atm/(K·kmol))
rho_w = 965 # kg/m³
gama_w = rho_w #kgf/m³
    #Pressure conditions

P_comp = 8  # Compression Pressure (kgf/cm²)
P_jus = 0.2  # Downstream Pressure (kgf/cm²)

for oil_type, api in oils.items():
    D_l = 141.5 / (api + 131.5)  # Density of the oil (kg/m³)
    rho_l = D_l * 1000  # Convert to kg/m³
    gama_l = rho_l  # kgf/m³

    # 🔹 Initial conditions of separator
        
        # Phases's initial levels

    h_tst0, h_wst0, h_lst0 = 2.5171, 1.152, 2.328  # initial separator levels (m)
    P_st0 = 15  # Initial pressure (kgf/cm²)
    Vwflcs, Vlwcs = 0.95, 9.586e-2      # Volume of water separator chamber [m³]
    Vwcl = 0.32      # Oil Critical Volume [m³]



        # Separator dimensions

    D = 3.048  # diameter of the separator (m)
    L = 23.503  # Lenght of the separator (m)
    V_total = np.pi * (D / 2) ** 2 * L  # Total Volume of the separator (m³)
    L_vert = 1.5  # Weir Lenght (m)
    h_vert = 2.8  # Weir level (m)

    # Chamber dimensions

    C_csy = 18.0  # Comprimento da Câmara de Separação (m)
    C_cly = 3.903  # Comprimento da Câmara de Óleo (m)


    thetaty_0 = np.arccos(1 - 2 * (h_tst0 / D))
    Vcs_inicial = (C_csy * D**2 / 4) * (thetaty_0 - np.sin(thetaty_0) * np.cos(thetaty_0))
    Vcs_0 = Vcs_inicial

    thetaly_0 = np.arccos(1 - 2 * (h_lst0 / D))
    Vcl_inicial = (C_cly * D**2 / 4) * (thetaly_0 - np.sin(thetaly_0) * np.cos(thetaly_0))
    Vcl_0 = Vcl_inicial

    thetawy_0 = np.arccos(1 - 2 * (h_wst0 / D))
    Vwcs_inicial = (C_csy * D**2 / 4) * (thetawy_0 - np.sin(thetawy_0) * np.cos(thetawy_0))
    Vwcs_0 = Vwcs_inicial 

    # Input Conditions of the Separator

        # Initial flow rates

    W_e = 0.184  # Water
    L_e = 6.006e-3  # Oil
    G_e = 7.182e-2  # Gas

    # Compositions components

    BSW_eflw = 0.02  # Water in Oil Content
    TOG_eflw = 3.013e-3  # Oil in Water Content
    T_st = 304  # Temperature (K)

    # Calculating the fractional compositions in the separation chamber

    Xlfwcs0 = Vwflcs/(Vcs_0-Vwcs_0) # Water in Oil chamber separator composition
    xwflcs0 = Vlwcs/Vwcs_0  # Oil in water chamber separator composition
    xwlcl0 = Vwcl/Vcl_0  # Water in oil chamber separator composition

    def separador_trifasico(t, y):
        h_tst, h_wst, h_lst, P_st, Xlfwcs, xwflcs, xwlcl = y
        hw=h_wst
        We = W_e

        # Efficiency of the separator 
        C_wl = [0.5486, -0.2675, -0.3626, 0.1356, -0.0747, 0.0890] # (Coefficients used from Model Predictive Control - 2016)
        efwlst = C_wl[0] + C_wl[1] * hw + C_wl[2] * L_e + C_wl[3] * hw * L_e + C_wl[4] * hw**2 + C_wl[5] * L_e**2

        C_lw = [0.5322, 2.1956, -0.4602, 0.1294, -2.8631, 0.1213, 1.2763, -0.0185] # (Coefficients used from Model Predictive Control - 2016)
        eflwst = C_lw[0] + C_lw[1] * hw + C_lw[2] * We + C_lw[3] * hw * We + C_lw[4] * hw**2 + C_lw[5] * We**2 + C_lw[6] * hw**3 + C_lw[7] * We**3
        
        # Geometry of circule segments
        thetaty = np.arccos(1 - 2*(h_tst / D))
        thetawy = np.arccos(1 - 2*(h_wst / D))
        thetaly = np.arccos(1 - 2*(h_lst / D))

        # Vt = (L * np.pi * D**2)/4 * (thetaty - np.sin(thetaty)*np.cos(thetaty))  # Total Volume of the separator (m³)
        Vwcs = (C_csy * D**2)/4 * (thetawy - np.sin(thetawy)*np.cos(thetawy))
        Vclst = (C_cly * D**2)/4 * (thetaly - np.sin(thetaly)*np.cos(thetaly))
        Vcsst = (C_csy * D**2)/4 * (thetaty - np.sin(thetaty)*np.cos(thetaty))

        # Averages Densities
        rho_fw = rho_w * (1 - Xlfwcs) + rho_l * Xlfwcs
        rho_fl = rho_l * (1 - xwflcs) + rho_w * xwflcs

        L_sst = ((CV_max_l * S_l) / (0.0693 * 60 * rho_fl)) * np.sqrt(max((P_st - P_jus) * D_l + gama_l * h_lst * 1e-4, 0))
        W_sst = ((CV_max_w * S_w) / (0.0693 * 60 * rho_fw)) * np.sqrt(max((P_st - P_jus) * D_w + (gama_w * h_wst + gama_l * (h_tst - h_wst)) * 1e-4, 0))
        G_sst = ((CV_max_g * S_g * R * T_st) / (2.832 * 60 * MW_g * P_st)) * np.sqrt(max((P_st + P_comp) * (P_st - P_comp) * D_g, 0))
            
        Lvy = 110.2046/60 * (L_vert - 0.2*max((h_tst - h_vert), 0)) * max((h_tst - h_vert), 0)**1.5
        
        if h_lst < h_vert:
            dht_dt = (1/(2*C_csy * np.sqrt(h_tst * (D - h_tst)))) * (W_e + L_e - Lvy - W_sst)
            dhl_dt = (Lvy - L_sst) / (2 * C_cly * np.sqrt(h_lst * (D - h_lst)))
        else:
            dht_dt = (W_e + L_e - L_sst - W_sst) / (2 * (C_csy + C_cly) * np.sqrt(h_tst * (D - h_tst)))
            dhl_dt = dht_dt

        dVwflcs_dt = L_e * BSW_eflw * (1 - efwlst) - Lvy * xwflcs   
        dXlfwcs_dt = dVwflcs_dt / (Vcsst - Vwcs)

        dVlwcs_dt = W_e * TOG_eflw * (1 - eflwst) - W_sst * Xlfwcs
        dxwflcs_dt = dVlwcs_dt / Vwcs

        dVwlcl_dt = (Lvy*xwflcs - L_sst*xwlcl) # For out BSW calculation
        dxwlcl_dt = dVwlcl_dt / Vclst

        dhw_dt = (W_e * (1 - TOG_eflw * eflwst) + L_e * BSW_eflw * efwlst - W_sst) / (2 * C_csy * np.sqrt(h_wst * (D - h_wst)))

        dP_dt = ((W_e + L_e + G_e - W_sst - L_sst - G_sst) * P_st) / (V_total - Vcsst - Vclst)
        
        return [
            dht_dt, dhw_dt, dhl_dt, dP_dt,
            dXlfwcs_dt, dxwflcs_dt, dxwlcl_dt
            ]

    y0 = [h_tst0, h_wst0, h_lst0, P_st0, Xlfwcs0, xwflcs0, xwlcl0]

    t_span = (0, 2000)
    t_eval = np.linspace(t_span[0], t_span[-1], 200000) 

    sol = solve_ivp(separador_trifasico, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)
    results[oil_type] = sol

    t = sol.t
    h_tst = sol.y[0]
    h_wst = sol.y[1]
    h_lst = sol.y[2]
    P_st = sol.y[3]
    Xlfwcs = sol.y[4]
    xwflcs = sol.y[5]
    xwlcl = sol.y[6]

    rho_fw_final = rho_w * (1 - xwflcs[-1]) + rho_l * xwflcs[-1]
    rho_fl_final = rho_l * (1 - Xlfwcs[-1]) + rho_w * Xlfwcs[-1]

    Lv_plot = []
    for h in h_tst:
        if h <= h_vert:
            lvy_inst = 0
        else:
            lvy_inst = 110.2046/60 * (L_vert - 0.2*max((h - h_vert), 0)) * max((h - h_vert), 0)**1.5
        Lv_plot.append(lvy_inst)


    Lv_plot = np.array(Lv_plot)

    L_sst_final = ((CV_max_l * S_l) / (0.0693 * 60 * rho_fl_final)) * np.sqrt(((P_st[-1] - P_jus) * D_l + gama_l * h_lst[-1] * 1e-4))
    W_sst_final = ((CV_max_w * S_w) / (0.0693 * 60 * rho_fw_final)) * np.sqrt(((P_st[-1] - P_jus) * D_w + (gama_w * h_wst[-1] + gama_l * (h_tst[-1] - h_wst[-1])) * 1e-4))
    G_sst_final = (CV_max_g * S_g * (R * T_st) / (2.832 * 60 * P_st[-1] * MW_g)) * np.sqrt(((P_st[-1] + P_comp) * (P_st[-1] - P_comp) * D_g))

    L_sst = ((CV_max_l * S_l) / (0.0693 * 60 * rho_fl_final)) * np.sqrt(np.maximum((P_st - P_jus) * D_l + gama_l * h_lst * 1e-4, 0)) * 3600

    print(f"\nTriphasic Separator - Grau API: {oil_type}")
    print(f"➡️ Output Oil flow: {L_sst_final * 3600:.6f} m³/h")
    print(f"➡️ Output water flow: {W_sst_final * 3600:.6f} m³/h")
    print(f"➡️ Output Gas flow: {G_sst_final * 3600:.6f} m³/h")
    print(f"Final Out BSW: {xwlcl[-1]*100:.6f}%")
    print(f"Total height: {h_tst[-1]:.6f} m")

# 1.1. Pressure separator by BS&W (P_st)
plt.figure(figsize=(8, 6))
for oil_type, sol in results.items():
    plt.plot(sol.y[3],sol.y[6]*100, label=f'{oil_type}')
plt.xlabel('Pressure (kgf/cm²)')
plt.ylabel('Final Out BSW (%)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# 1. Output Oil flow (L_sst)
for oil_type, sol in results.items():
    api_val = oils[oil_type]
    D_l_val = 141.5 / (api_val + 131.5)
    rho_l_val = D_l_val * 1000
    rho_fl = rho_l_val * (1 - sol.y[5]) + rho_w * sol.y[5]
    L_sst = ((CV_max_l * S_l) / (0.0693 * 60 * rho_fl)) * np.sqrt(np.maximum((sol.y[3] - P_jus) * D_l_val + rho_l_val * sol.y[2] * 1e-4, 0)) * 3600
    plt.plot(sol.t, L_sst, label=f'{oil_type}')
plt.xlabel('time (s)')
plt.ylabel('flow (m³/h)')
plt.title('Output Oil flow (L_sst)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# 2. Output water flow (W_sst)
plt.figure(figsize=(5, 3))
for oil_type, sol in results.items():
    api_val = oils[oil_type]
    rho_l_val = (141.5 / (api_val + 131.5)) * 1000
    rho_fw = rho_w * (1 - sol.y[4]) + rho_l_val * sol.y[4]
    W_sst = ((CV_max_w * S_w) / (0.0693 * 60 * rho_fw)) * np.sqrt(np.maximum((sol.y[3] - P_jus) * D_w + (gama_w * sol.y[1] + rho_l_val * (sol.y[0] - sol.y[1])) * 1e-4, 0)) * 3600
    plt.plot(sol.t, W_sst, label=f'{oil_type}')
plt.xlabel('time (s)')
plt.ylabel('flow (m³/h)')
plt.title('output water flow (W_sst)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# 3. Pressure separator (P_st)
plt.figure(figsize=(5, 3))
for oil_type, sol in results.items():
    plt.plot(sol.t, sol.y[3], label=f'{oil_type}')
plt.xlabel('time (s)')
plt.ylabel('Pressure (kgf/cm²)')
plt.title('Sparator pressure (P_st)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# 4. Total Level (h_tst)
plt.figure(figsize=(5, 3))
for oil_type, sol in results.items():
    plt.plot(sol.t, sol.y[0], label=f'{oil_type}')
plt.axhline(y=h_vert, color='r', linestyle=':', label=f'Weir Level({h_vert}m)')
plt.xlabel('time (s)')
plt.ylabel('Level (m)')
plt.title('Total level separator (h_tst)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# 5. Oil chamber level (h_lst)
plt.figure(figsize=(5, 3))
for oil_type, sol in results.items():
    plt.plot(sol.t, sol.y[2], label=f'{oil_type}')
plt.xlabel('time (s)')
plt.ylabel('level (m)')
plt.title('Oil chamber level (h_lst)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# 6. separation chamber level (h_wst)
plt.figure(figsize=(5, 3))
for oil_type, sol in results.items():
    plt.plot(sol.t, sol.y[1], label=f'{oil_type}')
plt.xlabel('time (s)')
plt.ylabel('level (m)')
plt.title('Water level chamber level (h_wst)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# 7. Levels of BSW OUT
plt.figure(figsize=(7,7))
hatches = ['//','\\','+','xx','..','--']
for i, (oil_type, sol) in enumerate(results.items()):
    final_bsw = sol.y[6][-1] * 100
    hatches_select = hatches[i % len(hatches)]
    bar_container = plt.bar(oil_type, final_bsw, edgecolor='black', hatch=hatches_select)
    plt.bar_label(bar_container, fmt='%.3f%%', padding=3)
plt.ylabel("Final Out BSW (%)")
plt.grid(False)
plt.show()

# Prepare a list to hold the data for each oil type
table_data = []

def calculate_segment_area(D, h):
    if h <= 0:
        return 0.0
    h = min(h, D)
    r = D / 2
    theta = 2 * np.arccos((r - h) / r)
    area = r**2 * (theta - np.sin(theta)) / 2
    return area

# Iterate through the stored results to build the table
for oil_type, sol in results.items():
    # Get oil properties for this run
    api_val = oils[oil_type]
    D_l_val = 141.5 / (api_val + 131.5)
    rho_l_val = D_l_val * 1000

    # Recalculate final oil flow for the table (in m³/h)
    rho_fl_final = rho_l_val * (1 - sol.y[5][-1]) + rho_w * sol.y[5][-1]
    L_sst_final_m3h = ((CV_max_l * S_l) / (0.0693 * 60 * rho_fl_final)) * np.sqrt(
        np.maximum((sol.y[3][-1] - P_jus) * D_l_val + rho_l_val * sol.y[2][-1] * 1e-4, 0)
    ) * 3600

    # --- Início do cálculo de tempo de residência ---
    # 1. Obter os níveis finais da simulação
    h_wst_final = sol.y[1][-1]
    h_tst_final = sol.y[0][-1]
    h_lst_final = sol.y[2][-1]

    # 2. Calcule final volumes
    A_w_final = calculate_segment_area(D, h_wst_final)
    V_w_final = A_w_final * C_csy

    A_total_liquido_sep = calculate_segment_area(D, h_tst_final)
    V_oleo_sep = (A_total_liquido_sep - A_w_final) * C_csy
    

    A_oleo_oleo = calculate_segment_area(D, h_lst_final)
    V_oleo_oleo = A_oleo_oleo * C_cly

    # Total Oil Volume
    V_oleo_total = V_oleo_sep + V_oleo_oleo

    # 3. Calcule residence time in minutes
    tr_water_min = (V_w_final / W_e) / 60 if W_e > 0 else 0
    tr_oil_min = (V_oleo_total / L_e) / 60 if L_e > 0 else 0

    # Append a dictionary with all final results
    table_data.append({
        "Oil Type": oil_type,
        "Final BSW %": sol.y[6][-1] * 100,
        "Final Oil Level (m)": sol.y[2][-1],
        "Final Water Level (m)": sol.y[1][-1],
        "Final Total Level (m)": sol.y[0][-1],
        "Final Pressure (kgf/cm²)": sol.y[3][-1],
        "Out Flow Oil (m³/h)": L_sst_final_m3h,
        "Tr Óleo (min)": tr_oil_min,
        "Tr Água (min)": tr_water_min
    })

# Create and display the pandas DataFrame
table_results = pd.DataFrame(table_data)
print("\n--- Simulation Summary ---")
print(table_results.to_string())
