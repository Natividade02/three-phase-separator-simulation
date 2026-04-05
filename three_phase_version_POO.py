import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Oil:
    def __init__ (self, name, api, R, rho_w, mw_g, gamma_w):
        self.name = name
        self.api = api
        self.d_l = self.rho_l(api) / 1000
        self.d_w = 0.965 
        self.d_g = 0.565 
        self.R = R
        self.rho_w = rho_w
        self.mw_g = mw_g
        self.gamma_l = self.rho_l(api)
        self.gamma_w = gamma_w

    def rho_l(self, api):
        return 141.5 / (api + 131.5) * 1000
    
    def get_mixture_densities(self, api, xlfwcs, xwflcs):

        # Fase Óleo (contaminada com água)
        rho_fl = self.rho_l(api) * (1 - xlfwcs) + self.rho_w * xlfwcs
        
        # Fase Água (contaminada com óleo)
        rho_fw = self.rho_w * (1 - xwflcs) + self.rho_l(api) * xwflcs
        
        return rho_fw, rho_fl

class Separator:
    def __init__(self, d, l, l_vert, h_vert, c_csy, c_cly, temp, cv_max_l, cv_max_w, cv_max_g, s_l, s_w, s_g):
        self.d = d
        self.l = l
        self.l_vert = l_vert
        self.h_vert = h_vert        
        self.c_csy = c_csy
        self.c_cly = c_cly
        self.T_st = temp

        # Valve Capacity
        self.cv_max_l = cv_max_l
        self.cv_max_w = cv_max_w
        self.cv_max_g = cv_max_g
        self.s_w = s_w
        self.s_l = s_l
        self.s_g = s_g

        #effs. constants
        self.c_wl = [0.5486, -0.2675, -0.3626, 0.1356, -0.0747, 0.0890]
        self.c_lw = [0.5322, 2.1956, -0.4602, 0.1294, -2.8631, 0.1213, 1.2763, -0.0185]

    def get_area(self, h):
        # PROTEÇÃO: Garante que h/d esteja entre -1 e 1 para o arccos
        # e que h esteja entre 0 e d
        h_safe = np.clip(h, 0, self.d)
        val = 1 - 2 * (h_safe / self.d)
        val = np.clip(val, -1.0, 1.0) 
        
        theta = np.arccos(val)
        return ((self.d**2 /4) * (theta - np.sin(theta) * np.cos(theta)))
    
    def get_v_oil(self, h):
        return self.c_cly * self.get_area(h)
    
    def get_v_w(self, h):
        return self.c_csy * self.get_area(h)
    
    def get_v_total(self, h):
        return self.c_csy * self.get_area(h)
    
    def get_efficience(self, l_e, h_w, w_e):
        eflw = (self.c_wl[0] + self.c_wl[1] * h_w + self.c_wl[2] * l_e +
                self.c_wl[3] * h_w * l_e + self.c_wl[4] * h_w**2 + 
                self.c_wl[5] * l_e**2) 
        
        efwl = (self.c_lw[0] + self.c_lw[1] * h_w + self.c_lw[2] * w_e + 
                self.c_lw[3] * h_w * w_e + self.c_lw[4] * h_w**2 + self.c_lw[5] * w_e**2 + 
                self.c_lw[7] * h_w**3 + self.c_lw[7] * w_e**3)
        
        # Eficiência deve estar entre 0 e 1
        return np.clip(eflw, 0, 1), np.clip(efwl, 0, 1)
    
    def out_flows(self, pressure_vessel, oleo, params, h_l, h_w, h_t, rho_fw, rho_fl):
        pres_l = (pressure_vessel - params['p_jus']) * oleo.d_l \
                + oleo.gamma_l * h_l * 1e-4

        pres_w = (pressure_vessel - params['p_jus']) * oleo.d_w + \
                (oleo.gamma_w * h_w + oleo.gamma_l * (h_t - h_w)) * 1e-4

        press_g = ((pressure_vessel + params['p_comp']) *
                (pressure_vessel - params['p_comp']) *
                oleo.d_g)

        l_s = ((self.cv_max_l * self.s_l) /
            (0.0693 * 60 * rho_fl)) * np.sqrt(max(pres_l, 0))

        w_s = ((self.cv_max_w * self.s_w) /
            (0.0693 * 60 * rho_fw)) * np.sqrt(max(pres_w, 0))

        p_vessel_safe = max(pressure_vessel, 1e-5)
        g_s = ((self.cv_max_g * self.s_g * oleo.R * self.T_st) /
            (2.832 * 60 * oleo.mw_g * p_vessel_safe)) * np.sqrt(max(press_g, 0))

        h_over_weir = max(h_t - self.h_vert, 0)
        lv = (110.2046 / 60) * \
            (self.l_vert - 0.2 * h_over_weir) * \
            (h_over_weir ** 1.5)

        return l_s, w_s, g_s, lv
 
    def simulation(self, t, y, oleo, params):
        h_t, h_w, h_l, pressure_vessel, xlfwcs, xwflcs, xwlcl = y
        
        h_t = np.clip(h_t, 1e-6, self.d - 1e-6)
        h_w = np.clip(h_w, 1e-6, h_t - 1e-6)
        h_l = np.clip(h_l, 1e-6, self.d - 1e-6)
        pressure_vessel = max(pressure_vessel, 1e-3)

        rho_fw_mix, rho_fl_mix = oleo.get_mixture_densities(oleo.api, xlfwcs, xwflcs)

        l_s, w_s, g_s, lv = self.out_flows(
            pressure_vessel, oleo, params, h_l, h_w, h_t, rho_fw_mix, rho_fl_mix
        )
        
        def get_surface_width(h):
            term = h * (self.d - h)
            return 2 * np.sqrt(max(term, 1e-8))
        
        width_t = get_surface_width(h_t)
        width_l = get_surface_width(h_l)
        width_w = get_surface_width(h_w)

        if h_l < self.h_vert:
            dht_dt = (params['w_e'] + params['l_e'] - lv - w_s) / (self.c_csy * width_t)
            dhl_dt = (lv - l_s) / (self.c_cly * width_l)
        else:
            dht_dt = (params['w_e'] + params['l_e'] - l_s - w_s) / ((self.c_csy + self.c_cly) * width_t)
            dhl_dt = dht_dt

        eflw, efwl = self.get_efficience(params['l_e'], h_w, params['w_e'])
        dhw_dt = (params['w_e'] * (1 - params['TOG_eflw'] * eflw) + params['l_e'] * params['BSW_eflw'] * efwl - w_s) / (self.c_csy * width_w)

        vol_total_atual = self.get_v_total(h_t) 
        vol_agua_atual = self.get_v_w(h_w)
        vol_oil_chamber = self.get_v_oil(h_l)
        
        vol_oil_sep = max(vol_total_atual - vol_agua_atual, 1e-6)
        vol_agua_atual = max(vol_agua_atual, 1e-6)
        vol_oil_chamber = max(vol_oil_chamber, 1e-6)
    
        dVwflcs_dt = params['l_e'] * params['BSW_eflw'] * (1 - efwl) - lv * xlfwcs 
        dXlfwcs_dt = dVwflcs_dt / vol_oil_sep

        dVlwcs_dt = params['w_e'] * params['TOG_eflw'] * (1 - eflw) - w_s * xwflcs
        dxwflcs_dt = dVlwcs_dt / vol_agua_atual

        dVwlcl_dt = (lv * xlfwcs - l_s * xwlcl)
        dxwlcl_dt = dVwlcl_dt / vol_oil_chamber

        vol_balance = max(vol_total_atual - vol_agua_atual - vol_oil_chamber, 1e-3)
        flow_in = params['w_e'] + params['l_e'] + params['g_e']
        flow_out = w_s + l_s + g_s
        
        dP_dt = ((flow_in - flow_out) * pressure_vessel) / vol_balance

        derivations = [
            dht_dt, dhw_dt, dhl_dt, dP_dt,
            dXlfwcs_dt, dxwflcs_dt, dxwlcl_dt
        ]

        return derivations

if __name__ == "__main__":

    y0 = [2.5171, 1.152, 2.328, 15, 0.0135, 0.00211, 0.0137]
    
    temp = float(input("Digite a temperatura (k):"))

    vaso1 = Separator(d=3.048, l=23.503, l_vert=1.5, h_vert=2.5, c_csy=18.0,
                      c_cly=3.903, cv_max_l=19.98, cv_max_w=245.11, cv_max_g=25,
                      s_l=0.305, s_w=0.79, s_g=0.5, temp=temp
                      )
    
    a_api = Oil(name="Asphaltic", api=12, R=0.0821,
                    rho_w=965, mw_g=16.48, gamma_w=965)
    eh_api = Oil(name="Extra Heavy", api=17, R=0.0821,
                       rho_w=965, mw_g=16.48, gamma_w=965)
    h_api = Oil(name="Heavy", api=23, R=0.0821,
                       rho_w=965, mw_g=16.48, gamma_w=965)
    m_api = Oil(name="Medium", api=34.2, R=0.0821,
                       rho_w=965, mw_g=16.48, gamma_w=965)
    l_api = Oil(name="Light", api=36.5, R=0.0821,
                       rho_w=965, mw_g=16.48, gamma_w=965)
    el_api = Oil(name="Extra Light", api=42.5, R=0.0821,
                       rho_w=965, mw_g=16.48, gamma_w=965)
    lista_oleos = [a_api, eh_api, h_api, m_api, l_api, el_api]

    layout = [vaso1]

    params = {
        'w_e': 0.184,
        'l_e': 6.006e-3,
        'g_e': 7.182e-2,
        'BSW_eflw': 0.02,
        'TOG_eflw': 3.013e-3,
        'p_jus': 0.2,
        'p_comp': 8
            }

    time = float(input("Tempo de simulação (s): "))
    t_span = (0, time)
    t_eval = np.linspace(t_span[0], t_span[-1])

    print("Solving...", end='', flush=True)

    results = {}
    for o in lista_oleos:
        print(f"Simulando óleo: {o.name}...")
        sol = solve_ivp(vaso1.simulation, t_span, y0, t_eval=t_eval, 
                        args=(o, params), 
                        method='RK45', rtol=1e-6, atol=1e-6)
        results[o.name] = sol
        print("\nDone!")

    fig, ax = plt.subplots(figsize=(8, 6))
    for nome_oleo, sol in results.items():
        ax.plot(sol.t, sol.y[3], label=f'{nome_oleo}')
    ax.set_title("Pressure in Vessel")
    ax.set_xlabel('Pressure (kgf/cm²)')
    ax.set_ylabel('Time(s)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()