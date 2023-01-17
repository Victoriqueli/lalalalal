"""
Created on Fri Dec  2 00:51:41 2022

@author: yutongli

"""
"""
Immune System is a complicated system, and it plays a really important role when we take various medicine. 
So, basically, we decide to stimulate how the immune system works after taking atezolizumab and nab-paclitaxel. 
By comparing the tumor size, amount of T cells and antigen releasing rate, we could know which medicine works better. 
Therefore, four module is established to mimic the immune system, which are the cancer module, 
the T cell module, and the APC module (including the antigen), and the checkpoints module. Each module are connected and 
placed in one function. Different parts are connected by Hill Equations. By adding atezolizumab, checkpoint affinity will get influenced
and by adding nab-paclitaxel, a chemotherapy medicine, is working on changing the cytokines producing rate and antigen releasing rate. 
Two modules will be done by each person and connection between various elements in the immune system is listed below. 
(Written by Matlab)
All the data is from the Github link listed below and the paper we mimiced is [1].
'''https://github.com/popellab/QSPIO-TNBC/tree/main/parameters'''
Reference:
    [1]Wang H, Ma H, Sové RJ, et al
Quantitative systems pharmacology model predictions for efficacy of atezolizumab and nab-paclitaxel in triple-negative breast cancer
Journal for ImmunoTherapy of Cancer 2021;9:e002100. doi: 10.1136/jitc-2020-002100


"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from math import log
from math import pi
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from IPython.display import HTML

"""
Create a dictionary to store the parameters' values needed in the following calculation.
"""
RC = {
    #Those parameters are for the APC module.
    "APC0_T": 4.0e5,
    "k_APC_mat": 1.5,  # (1/day)
    "c50": 1.0e-9,  # molarity
    "k_APC_death": 0.01,  # (1/day)
    "APC0_LN": 1.2e6,  # (cell/milliliter)
    "nLNs": 17,  # dimensionless
    "D_LN": 5,  # millimeter
    "k_APC_mig": 4,  # (1/day)
    "k_mAPC_death": 0.02,  # (1/day)
    "k_c": 2.0,  # (1/day)
    "c0": 1.0e-9,  # (molarity)
    "DAMPs": 1.34e-14,  # (mole/cell)

    #Those part are for the antigens.
    "k_dep": 0.0034,  # 1/day
    "k_xP_deg": 2.0,  # Rate of Extracellular Antigen Degradation(1/day)
    "k_up": 14.4,  # Rate of Antigen Uptake (1/day/cell)
    "cell": 1,  # Define Cell Dimension (cell)
    "k_P_deg": 17.28,  # Rate of Endosomal Antigen Degradation (1/day)
    "k_p_deg": 144.0,  # Rate of Endosomal Epitope Degradation (1/day)
    "k_on": 1.44e5,  # Rate of Antigen Binding (1/day/molarity)
    "A_s": 900.0,  # Endosomal Surface Area(micrometer**2)
    "kout": 28.8,  # Rate of MHC Externalization (1/day)
    "N_MHC": 1,  # Number of MHC Molecule Types (dimensionless)
    "n_MHC_T": 2e6,  # Total Amount of MHC (molecule)
    "kin": 14.4,  # Rate of MHC Internalization (1/day)
    "N_endo": 10,  # Number of Endosomal Vesicles per Cell (dimensionless)
    "V_endo": 4.0e-17,  # Endosomal Volume (liter)
    "A_endo": 1.5,  # Endosomal Surface Area (micrometer**2)
    "A_APC": 900.0,  # Surface Area of APC Cells (micrometer**2)
    # Number of Epitope Molecules for Half-Maximal T Cell Activation (molecule)
    "N_p_50": 1e-3,

    "A_syn": 37.8,  # Surface area of the synapse (micrometer**2)
    # Number of TCR molecules on naive T cells (molecule)
    "TCR_tot_abs": 15708,
    "D_cell": 17,  # Cancer Cell Diameter (micrometer)
    "D_Tcell": 6.94,  # T Cell Diameter (micrometer)
    # Number of TCR molecules on naive T cells (molecule)
    "TCR_tot_abs": 15708,
    "k_TCR_p": 1,  # Rate of modification of TCRs (1/second)
    "k_TCR_off": 1,  # Unbinding rate of ag/MHC to TCR (1/second)
    # binding rate of ag/MHC to TCR (1/(second*molecule/micrometer**2))
    "k_TCR_on": 1e-0,
    # Rate of modification of TCR that leads to non-signaling (1/second)
    "phi_TCR": 0.09,
    "N_TCR": 10,  # Number of intermediate steps (dimensionless)

    #Those part are for the Teff cells.
    "Q_nCD8_thym": 3.5e7,  # cell/day
    "nCD8_div": 1.11e6,  # dimensionless
    "k_nCD8_pro": 3.2e8,  # cell/day
    "K_nT_pro": 1e9,  # cell
    "k_nT_death": 0.002,  # 1/day
    "q_nCD8_P_out": 5.1,  # 1/day
    "q_nCD8_LN_in": 0.076,  # 1/day
    "q_nCD8_LN_out": 1.8,  # 1/day

    "k_nT_mig": 4.2e-13,  # Naive T Rate of Transmigration (1/minute/cell)
    "rho_adh": 5e8,  # T cell Adhesion Density (cell/centimeter**3)
    "gamma_P": 0.068,  # Peripheral Vascular Volume Fractions (dimensionless)
    "V_P": 60,  # Peripheral Compartment Volume (liter)
    "k_nCD8_act": 23,  # Maximum Rate of CD8+ T Cell Activation by mAPCs(1/day)
    # Tumor-specific T cell clone number (TMB) (Dimensionless)
    "n_clones_tum": 63,

    "kc_growth": 0.0072,  # (1/day)
    "kc_death": 0.0001,  # (1/day)
    "C_max": 2.7e4,  # (cell)


    "k_CD8_pro": 1.0,  # Rate of CD8+ T Cell Proliferation (1/day)
    "k_CD8_death": 0.01,  # Rate of CD8 T Cell Decay (1/day)
    "k_cell_clear": 0.1,  # Dead Cell Clearance Rate (1/day)
    # Half-Maximal cancer cell number for T cell recruitment (cell**2)
    "Kc_rec": 2.02e7,
    "k_Treg": 0.1,  # Rate of T Cell Death by Tregs (1/day)
    "k_Tcell": 0.1,  # Rate of T Cell Exhaustion by Cancer Cells (1/day)
    # Activated CD8 Rate of Transmigration(1/minute/cell)
    "k_CD8_mig": 5.8e-12,


    "q_CD8_P_out": 24,  # Activated CD8+ T Cell Transport P->C (1/day)
    "gamma_T": 0.522,  # Tumour Vascular Volume Fractions (dimensionless)
    "q_CD8_LN_out": 24.0,  # Activated CD8+ T Cell Transport LN->C (1/day)
    "k_IL2_deg": 0.2,  # Degradation Rate (1/minute)
    # Maximum Consumption Rate by T Cells (nanomole/cell/hour)
    "k_IL2_cons":  6.0e-6,
    # IL2 Concentration for Half-Maximal T Cell Proliferation (nanomolarity)
    "IL2_50": 0.32,
    "N0": 2,  # Baseline Number of Activated T Cell Generations (dimensionless)
    # Baseline Number of Activated T Cell Generations for co-stimulation (dimensionless)
    "N_costim": 3,
    # Additional Number of Activated CD8+ T Cell Generations Due to IL2 (dimensionless)
    "N_IL2_CD8": 11,
    "Ve_T": 0.37,  # Tumor Cell Volume Fraction(dimensionless)
    "PD1": 3.1e3*20*.45,  # PD1 Expression on T Cells (molecule)
    # PD1/PDL1 Concentration for Half-Maximal T Cell Killing (molecule/micrometer**2)
    "PD1_50": 6,
    "n_PD1": 2,  # Hill Coefficient for PD1/PDL1 (dimensionless)
    "kon_PD1_PDL1": 0.18,  # PD1-PDL1 kon(1/(micromolarity*second))
    "kd_PD1_PDL1": 8.2,  # PD1-PDL1 kd (micromolarity)
    "k_out_PDL1": 5e4,  # Expression rate of PDL1 on tumor cells (molecule/day)
    # Half-Maximal IFNg level for PD-L1 induction (picomolarity)
    "IFNg_50_ind": 2.96,
    "k_in_PDL1": 1,  # Degradation rate of PDL1 on tumor cells(1/day)
    "PDL1_base": 120000,  # molecule
    "PDL1_total": 83700,  # molecule
    "k_C1_death": 0.0001,  # 1/day
    "k_C_nabp":  0.06,  # 1/hour
    "IC50_nabp": 92,  # nanomolarity

    "n_sites_APC": 10,  # dimensionless
    "cell": 1,
    # number of neoantigen clones for naive CD4 T cell
    "n_T0_clones": log(63),
    # number of neoantigen clones for naive CD8 T cell
    "n_T1_clones": log(63),
    "Kc_nabp": 8e+07,  # cell
    "K_T_C": 1.2,  # dimensionless
    "nT0_LN": 155.6897,  # cell
    "nT1_LN": 113.7387,  # cell
    "p0_50": 2.6455e-5,  # molecule/micromter2
    "k_M1p0_TCR_off": 1,  # 1/second
    "phi_M1p0_TCR": 0.09,  # 1/second
    "k_M1p0_TCR_p":  1,  # 1/second
    "N_M1p0_TCR": 10,  # dimensionless
    "TCR_p0_tot": 103.813,  # molecule/micrometer**2
    "k_M1p0_TCR_on": 1,  # 1/(second*molecule/micrometer**2)

    # Thymic output of naive CD4+ T Cells into the blood (cell/day)
    "Q_nCD4_thym": 7e7,
    "nCD4_div": 1.16e6,  # Naive CD4+ T Cell Diversity (dimensionless)
    "k_nCD4_pro": 3.2e8,  # Rate of naive CD4+ T Cell proliferation (cell/day)
    # Naive T cell density for half-maximal peripheral proliferation (cell)
    "K_nT_pro": 1e9,
    "k_nT_death": 0.002,  # Rate of naive T cell death (1/day)

    "q_nCD4_P_out": 5.1,  # Naive CD4+ T Cell Transport P->C (1/day)
    "q_nCD4_LN_in": 0.1,  # Naive CD4+ T Cell Lymph Node Entry Rate (1/day)
    "q_nCD4_LN_out": 2.88,  # Naive CD4+ T Cell Lymph Node Exit Rate (1/day)

    "k_nCD4_act": 5,  # Maximum Rate of CD4+ Treg Activation by APCs (1/day)
    # % IL2 Concentration for Half-Maximal Treg Proliferation (nanomolarity)
    "IL2_50_Treg": 0.32,
    # Additional Number of Activated CD4+ T Cell Generations Due to IL2 (dimensionless)
    "N_IL2_CD4": 8.5,
    "H_CD28_APC": 0.1,  # dimensionless
    "k_M1p1_TCR_off": 1,  # 1/second
    "phi_M1p1_TCR": 0.09,  # 1/second
    "k_M1p1_TCR_p": 1,  # 1/second
    "N_M1p1_TCR": 10,  # dimensionless
    "TCR_p1_tot": 103.8131,  # molecule/micromiter**2
    "k_M1p1_TCR_on": 1,  # 1/(second * molecule/micrometer**2)

    # Those are the parameters needed.
    "k_Th_act": 10,  # T helper cell activation rate (1/day)
    "k_T0_pro": 1,  # 1/day
    "k_Th_Treg": 0.022,  # Th differentiation rate to Treg (1/day)
    "k_T0_death": 0.01,  # 1/day
    # [T cell clearance upon Ag clearance] Dead Cell Clearance Rate (1/day)
    "k_cell_clear": 0.1,
    "q_T0_P_in": 2.436,  # 1/minute
    "q_T0_P_out": 24,  # 1/day
    "q_T0_T_in":  5.8e-05,  # 1/(centimeter**3*minute)
    "q_T0_LN_out": 24,  # 1/day
    "TGFb_50": 0.07,  # nanomolarity
    "k_TGFb_deg": 14.3,  # 1/day
    "TGFbase": 0.016,  # nanomolarity
    "k_TGFb_Tsec": 1.2e-10,  # nanomole/cell/day
    "k_C_Tcell": 0.9,  # Death Rate Due to T Cells(1/day)
    # Dependence of Teff killing rate on Teff/Treg ratio (dimensionless)
    "K_T_Treg": 11,
    # Half-Maximal TGFb level for CD8+ T cell inhibition (nanomolarity)
    "TGFb_50_Teff": 0.14,
    "C1_PDL1_base": 1.2e+5,  # molecule
    "r_PDL1_IFNg": 6,  # dimensionless
    "r_PDL2C1": 0.07,  # dimensionless
    "k_C_growth": 0.0072,  # Growth Rate (1/day)
    "k_C_T1": 0.9,  # 1/day
    "K_T_C": 1.2,  # dimensionless
    "r_nabp": 1.5,  # dimensionless
    "V_1_NabP": 100,  # mg/ml
    "MW_nabp": 853.9,  # gram/mole

    # Checkpoint Module Parameters
    # Expression rate of PDL1 on tumor cells (molecule/day)
    "k_out_PDL1": 5e4,
    # Half-Maximal IFNg level for PD-L1 induction (picomolarity)
    "IFNg_50_ind": 2.96,
    # Number of folds increase of PDL1 expression by IFNg (dimensionless)
    "r_PDL1_IFNg": 6,
    "kon_PD1_PDL1_3D": 0.18,  # PD1-PDL1 kon (1/(micromolarity*second))
    "kd_PD1_PDL1": 8.2,  # PD1-PDL1 kd (micromolarity)
    # PD1-aPD1 Chi (antibody cross-arm binding strength)
    "Chi_PD1_aPD1_3D": 100,
    "N_avg": 6.0221409e23,  # Avogadro's Number (molecule/mole)
    "kon_PDL1_aPDL1": 4.3e5,  # PDL1-aPDL1 kon (1/(molarity*second))
    "kd_PDL1_aPDL1": 0.4,  # PDL1-aPDL1 kd (nanomolarity)
    # PDL1-aPDL1 Chi (antibody cross-arm binding strength)
    "Chi_PDL1_aPDL1_3D": 100,
    "r_PDL2": 0.07,  # PDL2/PDL1 ratio in tumor (dimensionless)
    "d_syn": 3.0,  # The synapse gap distance (nanometer)
    "kon_PD1_PDL2_3D": 0.24,  # PD1-PDL2 kon (1/(micromolarity*second))
    "kd_PD1_PDL2": 2.3,  # PD1-PDL2 kd (micromolarity)
    "kon_PD1_aPD1": 6.7e5,  # PD1-aPD1 kon (1/(molarity*second))
    "kd_PD1_aPD1": 4,  # PD1-aPD1 kd (nanomolarity)
    "Chi_PD1_aPD1": 33.3333,  # 1/nanometer
    "aPD1": 840,  # mg
    "k_C_death": 0.0001,  # 1/day
    "p1_50": 2.6455e-05,  # molecule/micrometer^2
    "k_CD4_pro": 1,  # Rate of CD4+ T Cell Proliferation(1/day)
    "k_CD4_death": 0.01,  # 1/day
    "k_CD4_mig": 5.8e-12,  # 1/minute/cell
    "q_CD4_P_out": 24,  # 1/day
    "q_CD4_LN_out": 24,  # 1/day
    "k_IL2_sec": 3.0e-5,  # nanomole/cell/hour
    "k_IFNg_sec": 5e-13,  # nanomole/cell/day
    "k_IFNg_deg": 7.68,
    "k_GI": 0.0138,
    "k_a1": 1.9,
    "k_a2": 2.5,
    "MW_ENT": 3.764085e5,
    "q_P": 8.7e-6,
    "gamma_C": 0.61,
    "gamma_P": 0.068,
    "q_T": 8.52e-5,
    "q_LN": 3.25e-6,
    "gamma_LN": 0.2,
    "q_LD": 0.0015,
    "k_cl":0.324,
    "Vmt": 325000,
    "Kt": 4260,
    "Vmcl": 8070, 
    "Kcl": 40.2,
    "k_vas_Csec": 1.1e-4,
    "k_vas_deg": 16.6,
    "c_vas_50": 1.07e3,
    "k_K_d":3.4e-3,
    "k_vas_nabp": 2.8e-4,
    "IC50_nabp_vas": 5.2,
    "k_K_nabp": 17.8,
    "gamma_aPD1": 0.522,
    "gamma_aPDL1": 0.522,
    "k_cl_aPD1": 0.315,
    "k_cl_aPDL1": 0.324,
    "k_K_g": 4.12,
    
    
}

RC['V_LN'] = RC["nLNs"] * 4/3*pi*(RC["D_LN"]/2)**3  # milliliter
RC["V_e"] = RC["V_endo"] * RC["N_endo"]  # Endosomal Volume
# Total Amount of MHC per Area
RC["MHC_T"] = RC["n_MHC_T"] / (RC["A_endo"] * RC["N_endo"] + RC["A_s"])
# Endosomal Surface Area (micrometer**2)
RC["A_e"] = RC['A_endo'] * RC['N_endo']
# TCR-pMHC Concentration for Half-Maximal T Cell Activation (molecule/micrometer**2)]
RC["p_50"] = RC["N_p_50"] / RC["A_syn"]
# Surface Area of Cancer Cells (micrometer**2)
RC["A_cell"] = 4*pi*(RC["D_cell"]/2)**2
# Surface Area of T Cells (micrometer**2)
RC["A_Tcell"] = 4*pi*(RC["D_Tcell"]/2)**2
# TCR molecules density on naive T cells (molecule/micrometer**2)
RC["TCR_tot"] = RC["TCR_tot_abs"]/RC["A_Tcell"]
RC["q_nCD8_P_in"] = RC["k_nT_mig"] * RC["rho_adh"] * RC["gamma_P"] * \
    RC["V_P"]  # Naive CD8+ T Cell Transport C->P (1/minute)
RC["q_CD8_P_in"] = RC["k_CD8_mig"] * RC["rho_adh"] * RC["gamma_P"] * RC["V_P"]
RC["q_CD8_T_in"] = RC["k_CD8_mig"] * RC["rho_adh"] * RC["gamma_T"]
# Volume of a cancer cell calculated based on cancer cell diameter
RC["vol_cell"] = 4/3*pi*(RC["D_cell"]/2)**3/RC["cell"]
# Volume of a T cell calculated based on the average T cell diameter
RC["vol_Tcell"] = 4/3*pi*(RC["D_Tcell"]/2)**3/RC["cell"]
RC["koff_PD1_PDL1"] = RC["kon_PD1_PDL1"] * RC["kd_PD1_PDL1"]
RC["q_nCD4_P_in"] = RC["k_nT_mig"] * RC["rho_adh"] * RC["gamma_P"] * RC["V_P"]
RC["kon_PD1_PDL2"] = RC["kon_PD1_PDL2_3D"] / RC["d_syn"]
RC["koff_PD1_PDL2"] = RC["kon_PD1_PDL2_3D"] * RC["kd_PD1_PDL2"]
RC["koff_PD1_aPD1"] = RC["kon_PD1_aPD1"] * RC["kd_PD1_aPD1"]
RC["kon_PD1_PDL1"] = RC["kon_PD1_PDL1_3D"] / RC["d_syn"]
RC["koff_PD1_PDL1"] = RC["kon_PD1_PDL1_3D"] * RC["kd_PD1_PDL1"]
RC["Chi_PD1_aPD1"] = RC["Chi_PD1_aPD1_3D"] / RC["d_syn"] * RC["N_avg"]
RC["koff_PDL1_aPDL1"] = RC["kon_PDL1_aPDL1"] * RC["kd_PDL1_aPDL1"]
RC["Chi_PDL1_aPDL1"] = RC["Chi_PDL1_aPDL1_3D"] / RC["d_syn"] * RC["N_avg"]
RC["V_T_NabP"] = RC["r_nabp"]*RC["V_1_NabP"]/RC["MW_nabp"]
RC["PDL2"] = RC["C1_PDL1_base"]*RC["r_PDL2C1"] / RC["A_cell"]
RC["q_CD4_P_in"] = RC["k_CD4_mig"]*RC["rho_adh"]*RC["gamma_P"]*RC["V_P"]
RC["q_CD4_T_in"] = RC["k_CD4_mig"]*RC["rho_adh"]*RC["gamma_T"]
RC["k_cln"] = 46/RC["MW_ENT"]
RC["Kc"] = 3.53/RC["MW_ENT"]

"""This is a class to describe the immune systems working pricniple. The equations 
of how the four modules connected to each other is written in the immmune function and 
the ODEs are solved in the simulate function.
"""
class Immune_systems():
    def __init__(self, initials, nabp, atezo):
        """
        
        
        Parameters
        ----------
        initials : list
            This are the initial conditions for  the parameters needed in the immune system. 
        nabp : integear
            0 or 1 in this case, where 0 means no nab-paclitaxel is added and 1 means nab-paclitaxel
            is added. 
        atezo : integear
            0 or 1 in this case, where 0 means no atezolizumab is added and 1 means atezolizumab
            is added. 

        Returns
        -------
        None.

        """
        self.initials = initials
        self.nabp = nabp
        self.atezo = atezo
        self.VT_list = []
        self.antigen_releasing_rate_List = []
        self.t = 0

    def immune(self, t, Immune):
        """
        

        Parameters
        ----------
        t : list
            The reaction time between 0 to 50 days and the time step is one dat]y.
        Immune :list
            The systems is a list to describe ODE function of various component.

        Returns
        -------
        list
            This is a list to show the various rates for each component with time. And this can provide to 
            the following function to solve the ODE later. 

        """
        if t >= 1 + self.t:
            print(t)
            self.t = t
        
        
            
        #All this parameters are we needed for the calculation of cancer size.
        Cx = Immune[0]
        T1_exh = Immune[1]
        Th_exh = Immune[2]

        #All the parameters that we needed are put inside the APC matrix.
        APC_tumor = Immune[3]
        APC_LN = Immune[4]
        mAPC_tumor = Immune[5]
        mAPC_LN = Immune[6]
        c = Immune[7]

        #All the calculated parameters are inside the list.
        V_e_P = Immune[8]  # concentration of antigen in endosome
        V_e_p = Immune[9]  # concentration of peptide in endosome
        A_e_Mp = Immune[10]  # concentration of pMHC complexes in endosome
        A_s_Mp = Immune[11]  # concentration of pMHC complexes on surface
        A_e_M = Immune[12]  # concentration of MHC in endosome
        A_s_M = Immune[13]  # concentration of MHC on surface
        V_T_P = Immune[14]  # concentration of antigen in tumor

        #All the parameters needed for the naive CD4 cells.
        V_P_nTreg = Immune[15]
        V_LN_nTreg = Immune[16]
        V_C_nTreg = Immune[17]

        #Those are the parameters needed for the activated Treg cells and their transportation.
        V_LN_aTreg = Immune[18]
        V_LN_Treg = Immune[19]
        V_C_Treg = Immune[20]
        V_P_Treg = Immune[21]
        V_T_Treg = Immune[22]
        V_LN_IL2_reg = Immune[23]

        #Those are parameters for the native CD8 cells. .
        V_C_nT = Immune[24]
        V_P_nT = Immune[25]
        V_LN_nT = Immune[26]

        #Those are the parameters for the activated Teffective cells.
        V_LN_aT = Immune[27]
        V_LN_T = Immune[28]
        V_C_T = Immune[29]
        V_P_T = Immune[30]
        V_T_T = Immune[31]
        V_LN_IL2 = Immune[32]

        #Those are the parameters for the activated Thelper cells.
        V_LN_aTh = Immune[33]
        V_T_Th = Immune[34]
        V_C_Th = Immune[35]
        V_P_Th = Immune[36]
        V_LN_Th = Immune[37]
        V_T_TGFb = Immune[38]

        # Those are the parameters to show the number of checkpoinits.
        PD1_PDL1 = Immune[39]
        PD1_PDL2 = Immune[40]
        PDL1_aPDL1 = Immune[41]
        PDL1_aPDL1_PDL1 = Immune[42]
        PDL1 = Immune[43]
        V_T_IFNg = Immune[44]
        V_T_C = Immune[45]
        

        V_1_NabP = Immune[46]
        V_2_NabP = Immune[47]
        V_3_NabP = Immune[48]
        V_T_c_vas = Immune[49]
        V_T_K = Immune[50]
        
        aPD1 = Immune[51]
        aPDL1 = Immune[52]
    

        #Those are the equaiton to calculate the size of the tumor.
        if t == 15:
            V_1_NabP = V_1_NabP + 100
            aPD1 = aPD1 + 840
        T_total = V_T_Treg + V_T_T + V_T_Th
        V_T = ((Cx + V_T_C)*RC["vol_cell"]+(T1_exh +
               Th_exh + T_total)*RC["vol_Tcell"])/RC["Ve_T"]
        V_T_NabP = RC["r_nabp"]*V_1_NabP/RC["MW_nabp"]
        PDL1_total = PD1_PDL1 + PDL1_aPDL1  + PDL1_aPDL1_PDL1 + PDL1
        

        #This the parameters needed to calculate the activated T effective and T regulatory cells.
        N_aT = RC["N0"] + RC["N_costim"] * RC["H_CD28_APC"] + \
            RC["N_IL2_CD8"]*V_LN_IL2/(RC["IL2_50"]+V_LN_IL2)
        N_aT0 = RC["N0"] + RC["N_costim"] * RC["H_CD28_APC"] + \
            RC["N_IL2_CD4"]*V_LN_IL2/(RC["IL2_50"]+V_LN_IL2)
            
        if self.nabp != 0:
            k_C1_therapy = RC["k_C_nabp"]*(V_T_NabP/(
                V_T_NabP+RC["IC50_nabp"]))*min(V_T_C, RC["Kc_nabp"])/V_T_C
            #Angiogenic factor release in response to nab-paclitaxel
            r_ang_rel_nabp = RC["k_vas_nabp"]*V_T_C*V_T_NabP/(V_T_NabP+RC["IC50_nabp_vas"])
            # Inhibition of tumor vasculature due to endothelial cell death by nab-paclitaxel
            r_nabp_inhi = RC["k_K_nabp"]*V_T_K*V_T_NabP
            #Intercompartmental distribution of nab-paclitaxel between V1 and V2 compartment
            r_V1_V2 = RC["Vmt"]/(V_1_NabP+RC["Kt"])*V_1_NabP - RC["Vmt"]/(V_2_NabP+RC["Kt"])*V_2_NabP
            #Intercompartmental clearance of nab-paclitaxel between V1 and V3 compartment
            r_V1_V3 = RC["Q2"]*V_1_NabP - RC["Q2"]*V_3_NabP
            #Clearance of nab-paclitaxel from V1 compartment
            r_V1 = RC["Vmcl"]/(V_1_NabP+RC["Kcl"])*V_1_NabP
            
            # Cancer cell death by nab-paclitaxel
            r_cancer_death_nabp = RC["k_C_nabp"]*V_T_C*(V_T_NabP/(V_T_NabP+RC["IC50_nabp"]))*min(V_T_C,RC["Kc_nabp"])/V_T_C
            
        else:
            k_C1_therapy = 0
            r_ang_rel_nabp = 0 
            r_nabp_inhi = 0
            r_V1_V2 = 0
            r_V1_V3 = 0
            r_V1 = 0
            r_cancer_death_nabp = 0
            

        ##Those are Hill equations for bindng affinity in later calculation.
        #Hill equation between APC and T cell
        H_APC = RC["n_sites_APC"]*mAPC_LN / \
            (RC["n_sites_APC"]*mAPC_LN + RC["nT0_LN"]
             * RC["n_T0_clones"]+RC["cell"])
        #Hill equation between mAPC and T cell
        H_mAPC = RC["n_sites_APC"]*mAPC_LN / \
            (RC["n_sites_APC"]*mAPC_LN+RC["nT1_LN"]*RC["n_T1_clones"]+RC["cell"])
        #Hill equation between APC and helper T cell
        H_APCh = RC["n_sites_APC"]*mAPC_LN / \
            (RC["n_sites_APC"]*mAPC_LN+RC["nT0_LN"]*RC["n_T1_clones"]+RC["cell"])
        #Those are Hill equation between antigen and TCR on the T cell.
        pTCR_p0_MHC_tot = RC["k_M1p0_TCR_off"]/(RC["k_M1p0_TCR_off"] + RC["phi_M1p0_TCR"]) * (RC["k_M1p0_TCR_p"]/(RC["k_M1p0_TCR_off"]+RC["k_M1p0_TCR_p"]))**RC["N_M1p0_TCR"]*(
            0.5*(A_s_Mp / RC["n_T0_clones"] + RC["TCR_p0_tot"] + RC["k_M1p0_TCR_off"]/RC["k_M1p0_TCR_on"])/RC["TCR_p0_tot"])**2-4 * A_s_Mp/RC["n_T0_clones"]/RC["TCR_p0_tot"]
        H_Ag = pTCR_p0_MHC_tot/(pTCR_p0_MHC_tot+RC["p0_50"])
        #Those are Hill equation between antigen and TCR on the Th cell.
        pTCR_p1_MHC_tot = RC["k_M1p1_TCR_off"]/(RC["k_M1p1_TCR_off"] + RC["phi_M1p1_TCR"]) * (RC["k_M1p1_TCR_p"]/(RC["k_M1p1_TCR_off"]+RC["k_M1p1_TCR_p"]))**RC["N_M1p1_TCR"]*(
            0.5*(A_s_Mp/RC["n_T1_clones"] + RC["TCR_p1_tot"] + RC["k_M1p1_TCR_off"]/RC["k_M1p1_TCR_on"])/RC["TCR_p1_tot"])**2-4 * A_s_Mp/RC["n_T1_clones"]/RC["TCR_p1_tot"]
        H_Agh = pTCR_p1_MHC_tot/(pTCR_p1_MHC_tot+RC["p1_50"])
        #This the Hill equaiton between T regulatory cell and T helper cell.
        H_TGFb = V_T_TGFb/(V_T_TGFb+RC["TGFb_50"])
        #This the Hill equaiton between T effective cell and TGFb.
        H_TGFb_Teff = V_T_TGFb/(V_T_TGFb+RC["TGFb_50_Teff"])
        #Hill equation between PD1 and cancer (PDL1)
        H_PD1_C1 = (PD1_PDL1/RC["PD1_50"])**RC["n_PD1"] / \
            (((PD1_PDL1)/RC["PD1_50"])**RC["n_PD1"] + 1)

        #Those two equations are used to describe the cancer cell killing rate and antigen releasing rate.
        cancer_death = RC["k_C_Tcell"] * V_T_C * V_T_T/(RC["K_T_C"]*V_T_C+V_T_T+RC["cell"])*V_T_T/(
            V_T_T+RC["K_T_Treg"]*V_T_Treg+RC["cell"])*(1-H_TGFb_Teff)*(1-H_PD1_C1)
 
        R_Tcell = (RC["k_C1_death"] + k_C1_therapy) * V_T_C + RC["k_C_T1"] * V_T_C * V_T_T/(RC["K_T_C"]*V_T_C +
                                                                                            V_T_T+RC["cell"]) * (V_T_T/V_T_T+RC["K_T_Treg"]*V_T_Treg+RC["cell"])*(1-H_TGFb_Teff)*(1-H_PD1_C1)
        antigen_relasing_rate = RC["k_dep"] + (RC["kc_death"] +k_C1_therapy)*V_T_C +RC["k_C_Tcell"]*V_T_C*V_T_T/(RC["K_T_C"]*V_T_C+V_T_T+RC["cell"])*V_T_T/(V_T_T+RC["K_T_Treg"]* V_T_Treg + RC["cell"])*(1-H_TGFb_Teff)*(1-H_PD1_C1)
        
        
        
        #Those are the reactions to describe the parameters used to calculate the tumor size.
        #Clearance of dead cancer cells from tumor.
        r_dead_cancer_tumor = RC["k_cell_clear"] * Cx
        #Clearance of exhausted CD8 T cells from tumor
        r_dead_CD8_T_tumor = RC["k_cell_clear"]*T1_exh
        #Clearance of exhausted CD4 T cells from tumor
        r_dead_CD4_T_tumor = RC["k_cell_clear"]*Th_exh

        #Those are the equations needed to calculate the APCs' amount.
        #APC recruitment/death in the tumour
        #0 -> APC_tumor
        r_APC_death_tumor = RC["k_APC_death"]*(RC["APC0_T"]*V_T-APC_tumor)
        #APC maturation in the tumour
        #APC_tumor ->  mAPC_tumor
        r_APC_maturation_tumor = RC["k_APC_mat"]*c/(c+RC["c50"])*APC_tumor
        #mAPC death in the tumour
        # mAPC_tumor -> 0
        r_mAPC_death_tumor = RC["k_mAPC_death"] * mAPC_tumor
        #APC recruitment/death in LN
        #0 -> APC_LN
        r_APC_death_LN = RC["k_APC_death"]*(RC["APC0_LN"]*RC["V_LN"]-APC_LN)
        #APC migration to the lymph node
        #mAPC_tumor -> mAPC_LN
        r_mAPC_migration_LN = RC["k_APC_mig"]*mAPC_tumor
        #mAPC death in the lymph node'
        #mAPC_LN -> 0
        r_mAPC_death_LN = RC["k_mAPC_death"]*mAPC_LN
        #Baseline cytokine secretion/degradation
        #0 -> c
        r_cytokines_baseline_secreation = RC["k_c"]*(RC["c0"]-c)
        #Cytokine release in response to the tumour"
        #0 -> c
        r_cytokines_tumor_release = R_Tcell * RC["DAMPs"]

        #Antgens amount are calculated here.
        #Antigen deposition from dying cancer cells
        r_dep = RC["k_dep"]
        #Free antigen degradation
        r_deg_P = RC["k_xP_deg"]*V_T_P*V_T
        #Antigen uptake by mature antigen presenting cells
        r_uptake_T = RC["k_up"]*APC_tumor*V_T_P*V_T
        r_uptake_e = RC["k_up"]*RC["cell"]*V_T_P*RC["V_e"]
        #Antigen degradation in APC endosomes'
        r_deg_anti = RC["k_P_deg"]*V_e_P*RC["V_e"]
        #Epitope degradation in APC endosomes
        r_deg_epi = RC["k_p_deg"]*V_e_p*RC["V_e"]
        #Antigen-MHC binding in endosome
        r_an_MHC_binding_en = RC["k_on"]*V_e_p*RC["A_e"]*RC["MHC_T"]
        #Antigen-MHC unbinding in endosome
        r_an_MHC_unbinding_en = RC["k_on"]*A_e_Mp*RC["A_e"]
        #Antigen-MHC unbinding on APC surface
        r_an_MHC_unbinding_sur = RC["k_on"]*A_s_Mp*RC["A_s"]
        #Antigen-MHC translocation
        r_an_MHC_trans = RC["kout"]*A_e_Mp*RC["A_e"]
        #MHC translocation
        r_MHC_trans = RC["kout"] * RC["MHC_T"] * \
            RC["A_e"] - RC["kin"] * 1e-6 * RC["A_s"]

        # Those are the reactions for Treg cell.
        # Thymic output of naive T cell to blood
        r_nTreg_thymic = RC["Q_nCD4_thym"]/RC["nCD4_div"]
        # Naive T cell proliferation in the peripheral compartment
        r_nTreg_pro_pheripheral = RC["k_nCD4_pro"]/RC["nCD4_div"] * \
            V_P_nTreg/(RC["K_nT_pro"]/RC["nCD4_div"] + V_P_nTreg)
        # Naive T cell proliferation in the TDLN compartment
        r_nTreg_pro_TDLN = RC["k_nCD4_pro"]/RC["nCD4_div"] * \
            V_LN_nTreg/(RC["K_nT_pro"]/RC["nCD4_div"] + V_LN_nTreg)
        # Naive T cell death in the peripheral compartment
        r_nTreg_death_pheripheral = RC["k_nT_death"] * V_P_nTreg
        # Naive T cell death in the central compartment
        r_nTreg_death_central = RC["k_nT_death"] * V_C_nTreg
        # Naive T cell death in the TDLN compartment
        r_nTreg_death_TDLN = RC["k_nT_death"] * V_LN_nTreg
        # Naive T cell entry into the peripheral compartment
        r_nTreg_entry_peripheral = RC["q_nCD4_P_in"] * V_C_nTreg
        # Naive T cell exit from the peripheral compartment
        r_nTreg_exit_peripheral = RC["q_nCD4_P_out"] * V_P_nTreg
        # Naive T cell transport into the lymph node
        r_nTreg_entry_TDLN = RC["q_nCD4_LN_in"] * V_C_nTreg
        # Naive T cell exit from the lymph node
        r_nTreg_exit_TDLN = RC["q_nCD4_LN_out"] * V_LN_nTreg
        #Naive T cell activation
        if APC_tumor < V_T_P / 5:
            r_nT_APC_antigen = RC["k_nCD4_act"] * H_APC * V_LN_nTreg
            #Those reactions are for Treg cells to be activated and transported.
            #Naive T cell activation
            r_nTreg_act = RC["k_nCD4_act"] * H_APC * \
                V_LN_nTreg * RC["n_clones_tum"]
            #Naive T cell activation
            r_nT_APC_antigen = RC["k_nCD8_act"] * H_APC * V_LN_nT

            #Those reactions are for Teffective cells to be activated.
            #Naive T cell activation
            r_nT_act = RC["k_nCD8_act"] * H_APC * V_LN_nT * RC["n_clones_tum"]

            # Those are the reactions for T cell activation to be a T helper cell.
            # Naive T cell activation
            # null -> V_LN.aT
            r_nTh_act = RC["k_Th_act"] * H_APCh * V_LN_nTreg*RC["n_T1_clones"]

        else:
            r_nT_APC_antigen = RC["k_nCD4_act"] * H_Ag * V_LN_nTreg
            #Those reactions are for Treg cells to be activated and transported.
            #Naive T cell activation
            r_nTreg_act = RC["k_nCD4_act"] * H_Ag * \
                V_LN_nTreg * RC["n_clones_tum"]
            #Naive T cell activation
            r_nT_APC_antigen = RC["k_nCD8_act"] * H_Ag * V_LN_nT

            #Those reactions are for Teffective cells to be activated.
            #Naive T cell activation
            r_nT_act = RC["k_nCD8_act"] * H_Ag * V_LN_nT * RC["n_clones_tum"]

            # Those are the reactions for T cell activation to be a T helper cell.
            # Naive T cell activation
            # null -> V_LN.aT
            r_nTh_act = RC["k_Th_act"] * H_Agh * V_LN_nTreg*RC["n_T1_clones"]

        #Activated T cells proliferation
        r_aTreg_pro = RC["k_CD4_pro"]/N_aT0*V_LN_aTreg
        r_aTreg_pro_ad = (r_aTreg_pro) * 2**(N_aT0)
        #T cell death in the central compartment
        r_Treg_death_central = RC["k_CD4_death"] * V_C_Treg
        #T cell death in the lymph node compartment
        r_Treg_death_LN = RC["k_CD4_death"] * V_LN_Treg
        #T cell death in the tumor compartment
        r_Treg_death_T = RC["k_CD4_death"] * V_T_Treg
        #T cell death in the pherical compartment
        r_Treg_death_P = RC["k_CD4_death"] * V_P_Treg
        #T cell clearance upon antigen clearance
        r_Treg_clear_Ag = RC["k_cell_clear"] * V_T_Treg * \
            (RC["Kc_rec"]/(V_T_C ** 2 + RC["Kc_rec"]))
        #T cell transport into the peripheral compartment
        r_Treg_in_pher = RC["q_CD4_P_in"] * V_C_Treg
        #T cell transport out of the peripheral compartment
        r_Treg_out_pher = RC["q_CD4_P_out"]*V_P_Treg
        #T cell transport into the tumor compartment
        r_Treg_in_tumor = RC["q_CD4_T_in"] * V_C_Treg * \
            (V_T_C**2/(V_T_C**2 + RC["Kc_rec"]))
        #T cell transport out of the lymph node compartment
        r_Treg_out_LN = RC["q_CD4_LN_out"]*V_LN_Treg
        #IL2 degradation
        r_IL2_degra_reg = RC["k_IL2_deg"]*V_LN_IL2_reg*RC["V_LN"]
        #IL2 consumption by Treg cells
        r_IL2_com_reg = RC["k_IL2_cons"]*V_LN_Treg * \
            V_LN_IL2_reg/(RC["IL2_50_Treg"]+V_LN_IL2_reg)
        #IL2 secretion from activated T cells
        r_IL12_sec_reg = RC["k_IL2_sec"]*V_LN_aTreg

        #Those are the reactions for T cell.
        #Those part is for naive T cell dynamics
        #Thymic output of naive T cell to blood
        r_nT_thymic = RC["Q_nCD8_thym"]/RC["nCD8_div"]
        #Naive T cell proliferation in the peripheral compartment
        r_nT_pro_pheripheral = RC["k_nCD8_pro"]/RC["nCD8_div"] * \
            V_P_nT/(RC["K_nT_pro"]/RC["nCD8_div"]+V_P_nT)
        #Naive T cell proliferation in the TDLN compartment
        r_nT_pro_TDLN = RC["k_nCD8_pro"]/RC["nCD8_div"] * \
            V_LN_nT/(RC["K_nT_pro"]/RC["nCD8_div"]+V_LN_nT)
        #Naive T cell death in the peripheral compartment
        r_nT_death_pheripheral = RC["k_nT_death"]*V_P_nT
        #Naive T cell death in the central compartment
        r_nT_death_central = RC["k_nT_death"]*V_C_nT
        #Naive T cell death in the TDLN compartment
        r_nT_death_TDLN = RC["k_nT_death"]*V_LN_nT
        #Naive T cell entry into the peripheral compartment
        r_nT_entry_peripheral = RC["q_nCD8_P_in"]*V_C_nT
        #Naive T cell exit from the peripheral compartment
        r_nT_exit_peripheral = RC["q_nCD8_P_out"]*V_P_nT
        #Naive T cell entry into the lymph node
        r_nT_entry_TDLN = RC["q_nCD8_LN_in"]*V_C_nT
        #Naive T cell exit from the lymph node
        r_nT_exit_TDLN = RC["q_nCD8_LN_out"]*V_LN_nT

        #Activated T cells proliferation
        r_aT_pro = RC["k_CD8_pro"]/N_aT*V_LN_aT
        r_aT_pro_ad = (r_aT_pro) * 2**(N_aT)
        #T cell death in the central compartment
        r_T_death_central = RC["k_CD8_death"] * V_C_T
        #T cell death in the lymph node compartment
        r_T_death_LN = RC["k_CD8_death"] * V_LN_T
        #T cell death in the tumor compartment
        r_T_death_T = RC["k_CD8_death"] * V_T_T
        #T cell death in the pherical compartment
        r_T_death_P = RC["k_CD8_death"] * V_P_T
        #T cell clearance upon antigen clearance
        r_T_clear_Ag = RC["k_cell_clear"] * V_T_T * \
            (RC["Kc_rec"]/(V_T_C**2 + RC["Kc_rec"]))
        #T cell death from Tregs
        r_T_death_Treg = RC["k_Treg"] * V_T_T * \
            V_T_Treg/(V_T_T+V_T_Treg+RC["cell"])
        #T cell death from cancer
        r_T_death_cancer = RC["k_Tcell"]*V_T_T * \
            V_T_C/(V_T_C+V_T_T+V_T_C)*H_PD1_C1
        #T cell transport into the peripheral compartment
        r_T_in_pher = RC["q_CD8_P_in"] * V_C_T
        #T cell transport out of the peripheral compartment
        r_T_out_pher = RC["q_CD8_P_out"]*V_P_T
        #T cell transport into the tumor compartment
        r_T_in_tumor = RC["q_CD8_T_in"] * V_C_T * \
            (V_T_C**2/(V_T_C**2 + RC["Kc_rec"]))
        #T cell transport out of the lymph node compartment
        r_T_out_LN = RC["q_CD8_LN_out"]*V_LN_T
        #IL2 degradation
        r_IL12_sec = RC["k_IL2_deg"]*V_LN_IL2*RC["V_LN"]
        #IL2 consumption by T cells
        r_IL2_degra = RC["k_IL2_cons"]*V_LN_T*V_LN_IL2/(RC["IL2_50"]+V_LN_IL2)
        #IL2 secretion from activated T cells
        r_IL2_com = RC["k_IL2_sec"]*V_LN_aT

        # Helper Tcell proliferation.
        # V_LN.aT -> null
        r_Th_pro = RC["k_T0_pro"]/N_aT0 * V_LN_aTh * 2**N_aT0
        # Differentiation between Treg and Th Cells
        #V_T.T -> V_T.T0'
        r_diff_Treg_Th = RC["k_Th_Treg"]*V_T_Th*H_TGFb
        #T cell death in the central compartment
        r_Th_death_C = RC["k_T0_death"]*V_C_Th
        #T cell death in the peripheral compartment
        r_Th_death_per = RC["k_T0_death"]*V_P_Th
        #T cell death in the lymph node compartment
        r_Th_death_LN = RC["k_T0_death"]*V_LN_Th
        #T cell death in the tumor compartment
        r_Th_death_tumor = RC["k_T0_death"]*V_T_Th
        #T cell clearance upon antigen clearance
        r_Th_ag_clear = RC["k_cell_clear"]*V_T_Th * \
            (RC["Kc_rec"]/(V_T_C**2 + RC["Kc_rec"]))
        #T cell transport into the peripheral compartment
        r_Th_entry_P = RC["q_T0_P_in"]*V_C_Th
        #T cell transport into the peripheral compartment
        r_Th_exit_P = RC["q_T0_P_out"]*V_P_Th
        #T cell transport into the tumor compartment
        r_Th_entry_tumor = RC["q_T0_T_in"]*V_T * \
            V_C_Th*(V_T_C**2/(V_T_C**2 + RC["Kc_rec"]))
        #T cell transport out of the lymph node compartment
        r_Th_exit_tumor = RC["q_T0_LN_out"]*V_T_Th
        #T cell transport out of the lymph node compartment
        r_Th_exit_LN = RC["q_T0_LN_out"]*V_LN_Th
        #TGFb secretion by triple-negative breast cancer cells
        r_TGFb_sec_cancer = RC["k_TGFb_deg"]*(RC["TGFbase"] - V_T_TGFb)*V_T
        #TGFb secretion from activated T cells in tumor
        r_TGFb_sec_aT = RC["k_TGFb_Tsec"]*(V_T_Th+V_T_Treg)

        #Those are the equations for the number of cancer.
        #Cancer cell growth

        r_C_growth = RC["k_C_growth"]*V_T_C * (log(V_T_K/(V_T_C+RC["cell"])))
        #Cancer cell death
        r_C_death = RC["k_C_death"]*V_T_C

        # Those are the reactions to describe the parameters used to calculate the number of PDL1-PD1 binding.(Secreation)
        # Translocation of PDL1 between cell surface and cytoplasm
        r_PDL1_trans_out = RC["k_out_PDL1"] * V_T_IFNg / (V_T_IFNg + RC["IFNg_50_ind"])/(
            RC["C1_PDL1_base"] * RC["r_PDL1_IFNg"] / RC["A_cell"] * RC["r_PDL2C1"])
        # Translocation of PDL1 between cell surface and cytoplasm
        r_PDL1_trans_in = RC["k_in_PDL1"] * \
            (RC["PDL1_base"] / RC["A_cell"] - PDL1_total)
        # Secreation of IFNg
        r_IFNg_sec = RC["k_IFNg_sec"] * V_T_Th
        # Degratation of IFNg
        r_IFNg_degra = RC["k_IFNg_deg"] * V_T_IFNg
        # Translocation of PDL2 between cell surface and cytoplasm
     
        # Dynamics of PD1/PDL1/PDL2/aPD1/aPDL1
        # binding and unbinding of PD1 PDL1 in synapse
        r_PD1_PDL1 = RC["kon_PD1_PDL1"] * \
            (RC["PD1"]) * (PDL1) - RC["koff_PD1_PDL1"] * PD1_PDL1
        # binding and unbinding of PD1 PDL2 in synapse
        r_PD1_PDL2 = RC["kon_PD1_PDL2"] * \
            (RC["PD1"]) * (RC["PDL2"]) - RC["koff_PD1_PDL2"] * PD1_PDL2

        #Those are the equations for the number of cancer.
        #Cancer cell growth
        r_C_growth = RC["k_C_growth"]*V_T_C*log(V_T_K/(V_T_C+RC["cell"]))
        #Cancer cell death
        r_C_death = RC["k_C_death"]*V_T_C
        
       
        # Those are the one added nabp and the reactions inside human body.
        
        #Secretion of angiogenic factors by cancer cells
        r_ang_sec = RC["k_vas_Csec"]*V_T_C
        #Degradation of tumor angiogenic factors
        r_ang_deg = RC["k_vas_deg"]*V_T_c_vas
        #Growth of tumor carrying capacity
        r_tumor_cap = RC["k_K_g"]*V_T_C*V_T_c_vas/(V_T_c_vas+RC["c_vas_50"])
        #Endogenous inhibition of previously generated vasculature
        r_ang_inh = RC["k_K_d"]*V_T_K*(V_T_C*2.57e-6)**(2/3)
        


        if self.atezo != 0:
            # binding and unbinding of PD1 to aPD1 on ',Tname,' surface in synapse
            r_PD1_aPD1 = 2 * RC["kon_PD1_aPD1"] * (RC["PD1"] * aPD1/RC["gamma_aPD1"]) - RC["koff_PD1_aPD1"] * PDL1_aPDL1
            # binding and unbinding of PD1 to PD1-aPD1 on ',Tname,' surface in synapse
            r_PD1_PD1aPD1 = RC["Chi_PD1_aPD1"] * RC["kon_PD1_aPD1"] * (RC["PD1"] * PDL1_aPDL1) - 2 * RC[
                "koff_PD1_aPD1"] * PDL1_aPDL1_PDL1
            # binding and unbinding of PDL1 to aPDL1 on ',Cname,' surface in synapse
            r_PDL1_aPDL1 = 2 * RC["kon_PDL1_aPDL1"] * \
                (PDL1 * aPDL1 / RC["gamma_aPDL1"]) - \
                RC["koff_PDL1_aPDL1"] * PDL1_aPDL1
            # binding and unbinding of PDL1 to PDL1-aPDL1 on ',Cname,' surface in synapse
            r_PDL1_PDL1aPDL1 = RC["Chi_PDL1_aPDL1"] * RC["kon_PDL1_aPDL1"] * (PDL1 * PDL1_aPDL1) - 2 * RC[
                "koff_PDL1_aPDL1"] * PDL1_aPDL1_PDL1
        else:
            # binding and unbinding of PD1 to aPD1 on ',Tname,' surface in synapse
            r_PD1_aPD1 = 0
            # binding and unbinding of PD1 to PD1-aPD1 on ',Tname,' surface in synapse
            r_PD1_PD1aPD1 = 0
            # binding and unbinding of PDL1 to aPDL1 on ',Cname,' surface in synapse
            r_PDL1_aPDL1 = 0
            # binding and unbinding of PDL1 to PDL1-aPDL1 on ',Cname,' surface in synapse
            r_PDL1_PDL1aPDL1 = 0

        #This is to write an ODE to express Cx, T1_exh and Th_exh
        dCxdt = - r_dead_cancer_tumor
        dT1_exhdt = - r_dead_CD8_T_tumor
        dTh_exhdt = -  r_dead_CD4_T_tumor

        #Calculate the ODE to get the APCs values
        dAPC_tumordt = r_APC_death_tumor - r_APC_maturation_tumor
        dAPC_LNdt = r_APC_death_LN
        dmAPC_tumordt = r_APC_maturation_tumor - \
            r_mAPC_migration_LN - r_mAPC_death_tumor
        dmAPC_LNdt = r_mAPC_migration_LN - r_mAPC_death_LN
        dcdt = r_cytokines_baseline_secreation + r_cytokines_tumor_release

        #Calculate the ODE to get the antigens values
        dV_e_Pdt = (r_uptake_e - r_deg_anti)/RC["V_e"]
        dV_e_pdt = (r_deg_anti - r_deg_epi - r_an_MHC_binding_en -
                    r_an_MHC_unbinding_en)/RC["V_e"]
        dA_e_Mpdt = (r_an_MHC_binding_en -
                     r_an_MHC_unbinding_en - r_an_MHC_trans)/RC["A_e"]
        dA_s_Mpdt = (r_an_MHC_trans - r_an_MHC_unbinding_sur)/RC["A_s"]
        dA_e_Mdt = (r_an_MHC_unbinding_en -
                    r_an_MHC_binding_en - r_MHC_trans)/RC["A_e"]
        dA_s_Mdt = (r_an_MHC_unbinding_sur + r_MHC_trans)/RC["A_s"]
        dV_T_Pdt = (r_dep - r_deg_P - r_uptake_T)/V_T

        #Calculate the ODE to get the activated Treg values
        dV_LN_aTregdt = r_nTreg_act - r_Treg_out_LN
        dV_LN_Tregdt = r_aTreg_pro_ad - r_Treg_out_LN - r_Treg_death_LN
        dV_C_Tregdt = r_Treg_out_LN - r_Treg_in_pher + \
            r_Treg_out_pher - r_Treg_in_tumor - r_Treg_death_central
        dV_P_Tregdt = r_Treg_in_pher - r_Treg_out_pher - r_Treg_death_P + r_diff_Treg_Th
        dV_T_Tregdt = r_Treg_in_tumor - r_Treg_death_T - r_Treg_clear_Ag
        dV_LN_IL2_regdt = r_IL12_sec_reg - r_IL2_degra_reg - r_IL2_com_reg

        #Calculated the ODEs for the naive Treg cells.
        dV_C_nTregdt = r_nTreg_thymic - r_nTreg_entry_peripheral + r_nTreg_exit_peripheral - \
            r_nTreg_entry_TDLN + r_nTreg_exit_TDLN - r_nTreg_death_central
        dV_P_nTregdt = r_nTreg_pro_pheripheral + r_nTreg_entry_peripheral - \
            r_nTreg_exit_peripheral - r_nTreg_death_pheripheral
        dV_LN_nTregdt = r_nTreg_pro_TDLN + r_nTreg_entry_peripheral - \
            r_nTreg_exit_peripheral - r_nTreg_death_TDLN

        #Calculated the ODEs for the naive Tcells.
        dV_C_nTdt = r_nT_thymic - r_nT_entry_peripheral + r_nT_exit_peripheral - \
            r_nT_entry_TDLN + r_nT_exit_TDLN - r_nT_death_central
        dV_P_nTdt = r_nT_pro_pheripheral + r_nT_entry_peripheral - \
            r_nT_exit_peripheral - r_nT_death_pheripheral
        dV_LN_nTdt = r_nT_pro_TDLN + r_nT_entry_peripheral - \
            r_nT_exit_peripheral - r_nT_death_TDLN - r_nT_APC_antigen

        #Those are calclated to solve the parameters in the list by ODE.(T effective cells)
        dV_LN_aTdt = r_nT_act - r_T_out_LN
        dV_LN_Tdt = r_aT_pro_ad - r_T_out_LN - r_T_death_LN
        dV_C_Tdt = r_T_out_LN - r_T_in_pher + \
            r_T_out_pher - r_T_in_tumor - r_T_death_central
        dV_P_Tdt = r_T_in_pher - r_T_out_pher - r_T_death_P
        dV_T_Tdt = r_T_in_tumor - r_T_death_T - \
            r_T_death_Treg - r_T_death_cancer - r_T_clear_Ag
        dV_LN_IL2dt = r_IL12_sec - r_IL2_degra - r_IL2_com

        #Those are the ODE equations to solve the T helper cells.
        dV_LN_aThdt = r_nTh_act - r_Th_pro
        dV_T_Thdt = r_Th_entry_tumor - r_Th_death_tumor - r_Th_exit_tumor - r_diff_Treg_Th
        dV_C_Thdt = r_Th_exit_LN - r_Th_death_C - \
            r_Th_entry_tumor - r_Th_entry_P + r_Th_exit_P
        dV_P_Thdt = r_Th_entry_P - r_Th_death_per - r_Th_exit_P
        dV_LN_Thdt = r_Th_pro - r_Th_death_LN - r_Th_exit_LN - r_Th_ag_clear
        dV_T_TGFbdt = r_TGFb_sec_aT - r_TGFb_sec_cancer

        #Those are the ODE equations for the number of cancer.
        dV_T_Cdt = r_C_growth - r_C_death - r_ang_rel_nabp - cancer_death - r_cancer_death_nabp

        #Those are the ODE equations for PDL1 and PD1_PDL1
        dpd1_pdl1dt = r_PD1_PDL1
        dpd2_pdl2dt = r_PD1_PDL2
        dpdl1_atezodt = r_PD1_aPD1 - r_PD1_PD1aPD1
        dpdl1_atezo_pdl1dt = r_PD1_PD1aPD1
        dpdl1dt = r_PDL1_trans_out + r_PDL1_trans_in - \
            r_PD1_PDL1 - r_PDL1_aPDL1 - r_PDL1_PDL1aPDL1
        dV_T_IFNgdt = r_IFNg_sec - r_IFNg_degra
        
        

        #Those are the ODE equations for the number of cancer.
        dV_T_Cdt = r_C_growth - r_C_death
        dV_T_Kdt = r_tumor_cap - r_ang_inh - r_nabp_inhi 
        dV_1_NabPdt = V_1_NabP - r_V1_V2 - r_V1_V3 - r_V1
        dV_2_NabPdt = V_2_NabP + r_V1_V2
        dV_3_NabPdt = V_3_NabP + r_V1_V3
        dV_T_c_vasdt = r_ang_sec -  r_ang_deg
        
        daPD1dt = aPD1 - RC["k_cl_aPD1"] * aPD1
        daPDL1dt = aPDL1 - RC["k_cl_aPDL1"] * aPDL1
        
        self.VT_list.append(V_T)


        return [dCxdt, dT1_exhdt, dTh_exhdt, dAPC_tumordt, dAPC_LNdt, dmAPC_tumordt, dmAPC_LNdt, dcdt,
                dV_e_Pdt, dV_e_pdt, dA_e_Mpdt, dA_s_Mpdt, dA_e_Mdt, dA_s_Mdt, dV_T_Pdt, dV_P_nTregdt, dV_LN_nTregdt, dV_C_nTregdt, dV_LN_aTregdt,
                dV_LN_Tregdt, dV_C_Tregdt, dV_P_Tregdt,  dV_T_Tregdt, dV_LN_IL2_regdt, dV_C_nTdt, dV_P_nTdt, dV_LN_nTdt, dV_LN_aTdt, dV_LN_Tdt, dV_C_Tdt, dV_P_Tdt, dV_T_Tdt,
                dV_LN_IL2dt, dV_LN_aThdt, dV_T_Thdt, dV_C_Thdt, dV_P_Thdt, dV_LN_Thdt, dV_T_TGFbdt, dpd1_pdl1dt, dpd2_pdl2dt, dpdl1_atezodt, dpdl1_atezo_pdl1dt, dpdl1dt, dV_T_IFNgdt, dV_T_Cdt, 
                dV_1_NabPdt, dV_2_NabPdt, dV_3_NabPdt, dV_T_c_vasdt, dV_T_Kdt, daPD1dt, daPDL1dt] 

    def simulate(self, t_start=0, t_end=28):  
        """
        

        Parameters
        ----------
        t_start : TYPE, optional
            Initial time for reaction. The default is 0.
        t_end : TYPE, optional
            End time for reaction. The default is 50.

        Returns
        -------
        2D list to describe the values of various components with time.

        """
    
        intervals = np.linspace(t_start, t_end, 29)
        solution = solve_ivp(
            self.immune, [t_start, t_end],[1692186.37991838, 121598100.7, 21504017.63, 30088.42153, 1290617.534,
                                      7994.975389, 1119638.044, 1.50E-07, 1.08E-09, 1.03E-10, 0.000360553, 0.054346931, 697.2936537, 23.2432063, 8.99E-10, 129486.0973,
                                      36.24077612, 2599.365231, 3794.442545, 21367.73076, 765663.9591, 111909867.1, 100511499.8, 0.00066611,
                                      2038.056155, 101529.8326, 13.19419616, 103609.5265, 1577674.855, 19951265.97, 2915074105, 122043960.7, 0.00066611, 107294.4682, 603951.7795,
                                      7780485.275, 1136815835, 232109228.5, 0.121422255, 145.7332167, 21.88935217, 0, 0, 128.7655938, 0.001845759, 1085453188, 100, 0, 0, 893.4067406, 2512042491, 840, 840], t_eval=intervals)
        return solution



if __name__ == "__main__":

    No_drug = Immune_systems([1692186.37991838, 121598100.7, 21504017.63, 30088.42153, 1290617.534,
                              7994.975389, 1119638.044, 1.50E-07, 1.08E-09, 1.03E-10, 0.000360553, 0.054346931, 697.2936537, 23.2432063, 8.99E-10, 129486.0973,
                              36.24077612, 2599.365231, 3794.442545, 21367.73076, 765663.9591, 111909867.1, 100511499.8, 0.00066611,
                              2038.056155, 101529.8326, 13.19419616, 103609.5265, 1577674.855, 19951265.97, 2915074105, 122043960.7, 0.00066611, 107294.4682, 603951.7795,
                              7780485.275, 1136815835, 232109228.5, 0.121422255, 145.7332167, 21.88935217, 0, 0, 128.7655938, 0.001845759, 1085453188, 100, 0, 0, 893.4067406, 2512042491, 840, 840], 0, 0)
    Nabp = Immune_systems([1692186.37991838, 121598100.7, 21504017.63, 30088.42153, 1290617.534,
                           7994.975389, 1119638.044, 1.50E-07, 1.08E-09, 1.03E-10, 0.000360553, 0.054346931, 697.2936537, 23.2432063, 8.99E-10, 129486.0973,
                           36.24077612, 2599.365231, 3794.442545, 21367.73076, 765663.9591, 111909867.1, 100511499.8, 0.00066611,
                           2038.056155, 101529.8326, 13.19419616, 103609.5265, 1577674.855, 19951265.97, 2915074105, 122043960.7, 0.00066611, 107294.4682, 603951.7795,
                           7780485.275, 1136815835, 232109228.5, 0.121422255, 145.7332167, 21.88935217, 0, 0, 128.7655938, 0.001845759, 1085453188, 100, 0, 0, 893.4067406, 2512042491, 840, 840], 1, 0)
    Atezo = Immune_systems([1692186.37991838, 121598100.7, 21504017.63, 30088.42153, 1290617.534,
                            7994.975389, 1119638.044, 1.50E-07, 1.08E-09, 1.03E-10, 0.000360553, 0.054346931, 697.2936537, 23.2432063, 8.99E-10, 129486.0973,
                            36.24077612, 2599.365231, 3794.442545, 21367.73076, 765663.9591, 111909867.1, 100511499.8, 0.00066611,
                            2038.056155, 101529.8326, 13.19419616, 103609.5265, 1577674.855, 19951265.97, 2915074105, 122043960.7, 0.00066611, 107294.4682, 603951.7795,
                            7780485.275, 1136815835, 232109228.5, 0.121422255, 145.7332167, 21.88935217, 0, 0, 128.7655938, 0.001845759, 1085453188, 100, 0, 0, 893.4067406, 2512042491, 840, 840], 0, 1)
    Nabp_Atezo = Immune_systems([1692186.37991838, 121598100.7, 21504017.63, 30088.42153, 1290617.534,
                                 7994.975389, 1119638.044, 1.50E-07, 1.08E-09, 1.03E-10, 0.000360553, 0.054346931, 697.2936537, 23.2432063, 8.99E-10, 129486.0973,
                                 36.24077612, 2599.365231, 3794.442545, 21367.73076, 765663.9591, 111909867.1, 100511499.8, 0.00066611,
                                 2038.056155, 101529.8326, 13.19419616, 103609.5265, 1577674.855, 19951265.97, 2915074105, 122043960.7, 0.00066611, 107294.4682, 603951.7795,
                                 7780485.275, 1136815835, 232109228.5, 0.121422255, 145.7332167, 21.88935217, 0, 0, 128.7655938, 0.001845759, 1085453188, 100, 0, 0, 893.4067406, 2512042491, 840, 840], 1, 1)

    Solution_No_drug = No_drug.simulate(t_start=0, t_end=28)
    Solution_Nabp = Nabp.simulate(t_start=0, t_end=28)
    Solution_Atezo = Atezo.simulate(t_start=0, t_end=28)
    Solution_Nabp_Atezo = Nabp_Atezo.simulate(t_start=0, t_end=28)
    
    No_drug_Teff = Solution_No_drug.y[31]
    Nabp_Teff = Solution_Nabp.y[31]
    Atezo_Teff = Solution_Atezo.y[31]
    Nabp_Atezo_Teff = Solution_Nabp_Atezo.y[31]

    x = np.arange(29)
    plt.plot(x, Solution_No_drug.y[31], label="No drug")
    plt.plot(x, Solution_Nabp.y[31], label="Nabp")
    plt.plot(x, Solution_Atezo.y[31], label="Atezo")
    plt.plot(x, Solution_Nabp_Atezo.y[31], label="Combination")
    plt.xlabel("time")
    plt.ylabel("Teff")
    plt.title('Teff Comparion')
    plt.legend(loc="best")
    plt.show()

    No_drug_APC = Solution_No_drug.y[34]
    Nabp_T_APC = Solution_Nabp.y[34]
    Atezo_T_APC = Solution_Atezo.y[34]
    Nabp_Atezo_APC = Solution_Nabp_Atezo.y[34]

    x = np.arange(51)
    plt.plot(x, Solution_No_drug.y[34], label="No drug")
    plt.plot(x, Solution_Nabp.y[34], label="Nabp")
    plt.plot(x, Solution_Atezo.y[34], label="Atezo")
    plt.plot(x, Solution_Nabp_Atezo.y[34], label="Combination")
    plt.xlabel("time")
    plt.ylabel("Th")
    plt.title('Thelper Comparison')
    plt.legend(loc="best")
    plt.show()

    T_total = Solution_No_drug.y[22] + Solution_No_drug.y[34] + Solution_No_drug.y[31]
    V_T_list = ((Solution_No_drug.y[0] + Solution_No_drug.y[45]) * RC["vol_cell"] +
                (Solution_No_drug.y[1] + Solution_No_drug.y[2] + T_total) * RC["vol_Tcell"]) / RC["Ve_T"]
    No_drug_T_VT = V_T_list
    T_total = Solution_Nabp.y[22] + Solution_Nabp.y[34] +Solution_Nabp.y[31]
    V_T_list = ((Solution_Nabp.y[0] + Solution_Nabp.y[45]) * RC["vol_cell"] +
                (Solution_Nabp.y[1] + Solution_Nabp.y[2] + T_total) * RC["vol_Tcell"]) / RC["Ve_T"]
    Nabp_T_VT = V_T_list
    T_total = Solution_Atezo.y[22] + Solution_Atezo.y[34] + Solution_Atezo.y[31]
    V_T_list = ((Solution_Atezo.y[0] + Solution_Atezo.y[45]) * RC["vol_cell"] +
                (Solution_Atezo.y[1] + Solution_Atezo.y[2] + T_total) * RC["vol_Tcell"]) / RC["Ve_T"]
    Atezo_T_VT = V_T_list
    T_total = Solution_Nabp_Atezo.y[22] + Solution_Nabp_Atezo.y[34] + Solution_Nabp_Atezo.y[31]
    V_T_list = ((Solution_Nabp_Atezo.y[0] + Solution_Nabp_Atezo.y[45]) * RC["vol_cell"] +
                (Solution_Nabp_Atezo.y[1] + Solution_Nabp_Atezo.y[2] + T_total) * RC["vol_Tcell"]) / RC["Ve_T"]
    Nabp_Atezo_T_VT = V_T_list

    x = np.arange(29)
    plt.plot(x, No_drug_T_VT, label="No drug")
    plt.plot(x, Nabp_T_VT, label="Nabp")
    plt.plot(x, Atezo_T_VT, label="Atezo")
    plt.plot(x, Nabp_Atezo_T_VT, label="Combination")
    plt.xlabel("time")
    plt.ylabel("V_T")
    plt.title('VT comparison')
    plt.legend(loc="best")
    plt.show()

    
