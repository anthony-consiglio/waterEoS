"""
Literature references of the waterEoS models and reference data, with DOIs.

REFERENCES          reference key -> {'short', 'citation', 'doi', 'notes'}
                    ('notes' lists errata and corrections, each with its own DOI)
MODEL_REFERENCES    model key -> reference keys, the primary source first
REFERENCE_GROUPS    (title, reference keys) in display order

Checked against the Crossref records of each DOI (November 2026).
"""


def _ref(short, citation, doi, notes=()):
    return {'short': short, 'citation': citation, 'doi': doi,
            'notes': [{'text': text, 'doi': note_doi} for text, note_doi in notes]}


REFERENCES = {
    # Equations of state for real water
    'holten2014': _ref(
        'Holten et al. (2014)',
        'V. Holten, J. V. Sengers, and M. A. Anisimov. "Equation of state for supercooled water at pressures '
        'up to 400 MPa." J. Phys. Chem. Ref. Data 43, 043101 (2014).',
        '10.1063/1.4895593'),
    'caupin2019': _ref(
        'Caupin & Anisimov (2019)',
        'F. Caupin and M. A. Anisimov. "Thermodynamics of supercooled and stretched water: Unifying '
        'two-structure description and liquid-vapor spinodal." J. Chem. Phys. 151, 034503 (2019).',
        '10.1063/1.5100228',
        notes=[('Erratum: J. Chem. Phys. 156, 069902 (2022).', '10.1063/5.0085566'),
               ('Erratum: J. Chem. Phys. 163, 039902 (2025). Its corrected Table III is the caupin2019 '
                'model; caupin2019_kim is Table II as printed.', '10.1063/5.0284908')]),
    'kim2017': _ref(
        'Kim et al. (2017)',
        'K. H. Kim, A. Späh, H. Pathak, F. Perakis, D. Mariedahl, K. Amann-Winkel, J. A. Sellberg, J. H. Lee, '
        'S. Kim, J. Park, K. H. Nam, T. Katayama, and A. Nilsson. "Maxima in the thermodynamic response and '
        'correlation functions of deeply supercooled water." Science 358, 1589–1593 (2017). The isothermal '
        'compressibility data of the caupin2019_kim fit.',
        '10.1126/science.aap8269'),
    'duska2020': _ref(
        'Duška (2020)',
        'M. Duška. "Water above the spinodal." J. Chem. Phys. 152, 174501 (2020).',
        '10.1063/5.0006431'),
    'shi_tanaka2020': _ref(
        'Shi & Tanaka (2020)',
        'R. Shi and H. Tanaka. "The anomalies and criticality of liquid water." Proc. Natl. Acad. Sci. U.S.A. '
        '117, 26591–26599 (2020). The hierarchical two-state equation of state (Table S2) and the transport '
        'model for viscosity, self-diffusion and rotational relaxation (Table S3).',
        '10.1073/pnas.2008426117'),
    'grenke2025': _ref(
        'Grenke & Elliott (2025)',
        'J. H. Grenke and J. A. W. Elliott. "Analytic correlation for the thermodynamic properties of water '
        'at low temperatures (200–300 K) and high pressures (0.1–400 MPa)." J. Phys. Chem. B 129, '
        '1997–2012 (2025).',
        '10.1021/acs.jpcb.4c03909',
        notes=[('Correction: J. Phys. Chem. B 129, 9850–9853 (2025).', '10.1021/acs.jpcb.5c05934')]),
    'moynihan1996': _ref(
        'Moynihan (1996)',
        'C. T. Moynihan. "Two species/nonideal solution model for amorphous/amorphous phase transitions." '
        'MRS Proc. 455, 411 (1996).',
        '10.1557/PROC-455-411'),
    'mishima2023': _ref(
        'Mishima & Sumita (2023)',
        'O. Mishima and T. Sumita. "Equation of state of liquid water written by simple experimental '
        'polynomials and the liquid–liquid critical point." J. Phys. Chem. B 127, 1414–1421 (2023).',
        '10.1021/acs.jpcb.2c08342'),
    # Transport
    'singh2017': _ref(
        'Singh et al. (2017)',
        'L. P. Singh, B. Issenmann, and F. Caupin. "Pressure dependence of viscosity in supercooled water '
        'and a unified approach for thermodynamic and dynamic anomalies of water." Proc. Natl. Acad. Sci. '
        'U.S.A. 114, 4312–4317 (2017).',
        '10.1073/pnas.1619501114'),
    # PC-SAFT
    'gross2001': _ref(
        'Gross & Sadowski (2001)',
        'J. Gross and G. Sadowski. "Perturbed-chain SAFT: An equation of state based on a perturbation '
        'theory for chain molecules." Ind. Eng. Chem. Res. 40, 1244–1260 (2001).',
        '10.1021/ie0003887'),
    'diamantonis2011': _ref(
        'Diamantonis & Economou (2011)',
        'N. I. Diamantonis and I. G. Economou. "Evaluation of statistical associating fluid theory (SAFT) and '
        'perturbed chain-SAFT equations of state for the calculation of thermodynamic derivative properties '
        'of fluids related to carbon capture and sequestration." Energy Fuels 25, 3334–3343 (2011).',
        '10.1021/ef200387p'),
    'novak2024': _ref(
        'Novak et al. (2024)',
        'N. Novak, X. Liang, and G. M. Kontogeorgis. "Prediction of water anomalous properties by '
        'introducing the two-state theory in SAFT." J. Chem. Phys. 160, 104505 (2024).',
        '10.1063/5.0186752'),
    'novak2025': _ref(
        'Novak et al. (2025)',
        'N. Novak, X. Liang, and G. M. Kontogeorgis. "PC-SAFT-TS-CAF: A revised two state model for pure '
        'water and binary aqueous mixtures." Fluid Phase Equilib. 598, 114475 (2025).',
        '10.1016/j.fluid.2025.114475'),
    # Fits to simulations of model waters
    'singh2016': _ref(
        'Singh et al. (2016)',
        'R. S. Singh, J. W. Biddle, P. G. Debenedetti, and M. A. Anisimov. "Two-state thermodynamics and the '
        'possibility of a liquid-liquid phase transition in supercooled TIP4P/2005 water." J. Chem. Phys. '
        '144, 144504 (2016).',
        '10.1063/1.4944986'),
    'biddle2017': _ref(
        'Biddle et al. (2017)',
        'J. W. Biddle, R. S. Singh, E. M. Sparano, F. Ricci, M. A. González, C. Valeriani, J. L. F. Abascal, '
        'P. G. Debenedetti, M. A. Anisimov, and F. Caupin. "Two-structure thermodynamics for the TIP4P/2005 '
        'model of water covering supercooled and deeply stretched regions." J. Chem. Phys. 146, 034502 (2017).',
        '10.1063/1.4973546',
        notes=[('Erratum: J. Chem. Phys. 148, 069901 (2018).', '10.1063/1.5024509')]),
    'holten2014st2': _ref(
        'Holten et al. (2014), ST2',
        'V. Holten, J. C. Palmer, P. H. Poole, P. G. Debenedetti, and M. A. Anisimov. "Two-state '
        'thermodynamics of the ST2 model for supercooled water." J. Chem. Phys. 140, 104502 (2014).',
        '10.1063/1.4867287'),
    'holten2013': _ref(
        'Holten et al. (2013)',
        'V. Holten, D. T. Limmer, V. Molinero, and M. A. Anisimov. "Nature of the anomalies in the '
        'supercooled liquid state of the mW model of water." J. Chem. Phys. 138, 174501 (2013).',
        '10.1063/1.4802992'),
    'gartner2020': _ref(
        'Gartner et al. (2020)',
        'T. E. Gartner, L. Zhang, P. M. Piaggi, R. Car, A. Z. Panagiotopoulos, and P. G. Debenedetti. '
        '"Signatures of a liquid–liquid transition in an ab initio deep neural network model for water." '
        'Proc. Natl. Acad. Sci. U.S.A. 117, 26040–26046 (2020).',
        '10.1073/pnas.2015440117'),
    'weis2022': _ref(
        'Weis et al. (2022)',
        'J. Weis, F. Sciortino, A. Z. Panagiotopoulos, and P. G. Debenedetti. "Liquid–liquid criticality in '
        'the WAIL water model." J. Chem. Phys. 157, 024502 (2022).',
        '10.1063/5.0099520'),
    'eltareb2022': _ref(
        'Eltareb et al. (2022)',
        'A. Eltareb, G. E. Lopez, and N. Giovambattista. "Evidence of a liquid–liquid phase transition in '
        'H₂O and D₂O from path-integral molecular dynamics simulations." Sci. Rep. 12, 6004 (2022).',
        '10.1038/s41598-022-09525-x'),
    'zorzi2024': _ref(
        'Zorzi et al. (2024)',
        'N. Zorzi, A. Neophytou, and F. Sciortino. "Two state model for the ML-BOP potential." Mol. Phys. '
        '122, e2407025 (2024).',
        '10.1080/00268976.2024.2407025'),
    'yu2023': _ref(
        'Yu et al. (2023)',
        'Z. Yu, R. Shi, and H. Tanaka. "A unified description of the liquid structure, static and dynamic '
        'anomalies, and criticality of TIP4P/2005 water by a hierarchical two-state model." J. Phys. Chem. B '
        '127, 3452–3462 (2023).',
        '10.1021/acs.jpcb.3c00869'),
    'handle2018': _ref(
        'Handle & Sciortino (2018)',
        'P. H. Handle and F. Sciortino. "Potential energy landscape of TIP4P/2005 water." J. Chem. Phys. '
        '148, 134505 (2018).',
        '10.1063/1.5023894'),
    'eltareb2024': _ref(
        'Eltareb et al. (2024)',
        'A. Eltareb, G. E. Lopez, and N. Giovambattista. "Potential energy landscape of a flexible water '
        'model: Equation of state, configurational entropy, and Adam–Gibbs relationship." J. Chem. Phys. '
        '160, 154510 (2024).',
        '10.1063/5.0200306'),
    # Reference equations, ices and vapor
    'wagner2002': _ref(
        'IAPWS-95',
        'W. Wagner and A. Pruß. "The IAPWS formulation 1995 for the thermodynamic properties of ordinary '
        'water substance for general and scientific use." J. Phys. Chem. Ref. Data 31, 387–535 (2002). '
        'Reference state of every model, and the ideal-gas part and virial coefficients of the vapor.',
        '10.1063/1.1461829'),
    'journaux2020': _ref(
        'Journaux et al. (2020)',
        'B. Journaux, J. M. Brown, A. Pakhomova, I. E. Collings, S. Petitgirard, P. Espinoza, '
        'T. Boffa Ballaran, S. D. Vance, J. Ott, F. Cova, G. Garbarino, and M. Hanfland. "Holistic approach '
        'for studying planetary hydrospheres: Gibbs representation of ices thermodynamics, elasticity, and '
        'the water phase diagram to 2,300 MPa." J. Geophys. Res. Planets 125, e2019JE006176 (2020). '
        'SeaFreeze: the water1 liquid and ices Ih, II, III, V and VI.',
        '10.1029/2019JE006176'),
    'french2015': _ref(
        'French & Redmer (2015)',
        'M. French and R. Redmer. "Construction of a thermodynamic potential for the water ices VII and X." '
        'Phys. Rev. B 91, 014308 (2015). Ice VII/X in the H₂O phase diagram.',
        '10.1103/PhysRevB.91.014308'),
    'feistel2006': _ref(
        'IAPWS-06',
        'R. Feistel and W. Wagner. "A new equation of state for H₂O ice Ih." J. Phys. Chem. Ref. Data 35, '
        '1021–1047 (2006). Ice Ih melting line of the EoS phase diagram.',
        '10.1063/1.2183324'),
    'wagner2011': _ref(
        'IAPWS R14-08',
        'W. Wagner, T. Riethmann, R. Feistel, and A. H. Harvey. "New equations for the sublimation pressure '
        'and melting pressure of H₂O ice Ih." J. Phys. Chem. Ref. Data 40, 043103 (2011). Sublimation '
        'pressure and melting curves.',
        '10.1063/1.3657937'),
    'wagner1993': _ref(
        'Wagner & Pruß (1993)',
        'W. Wagner and A. Pruß. "International equations for the saturation properties of ordinary water '
        'substance. Revised according to the International Temperature Scale of 1990." J. Phys. Chem. Ref. '
        'Data 22, 783–787 (1993). Saturation pressure.',
        '10.1063/1.555926'),
    'murphy2005': _ref(
        'Murphy & Koop (2005)',
        'D. M. Murphy and T. Koop. "Review of the vapour pressures of ice and supercooled water for '
        'atmospheric applications." Q. J. R. Meteorol. Soc. 131, 1539–1565 (2005). Vapor-pressure reference '
        'lines of the Measured Data tab.',
        '10.1256/qj.04.94'),
}

MODEL_REFERENCES = {
    'duska2020': ['duska2020'],
    'holten2014': ['holten2014'],
    'caupin2019': ['caupin2019'],
    'caupin2019_kim': ['caupin2019', 'kim2017'],
    'shi_tanaka2020': ['shi_tanaka2020'],
    'shi_tanaka2020_transport': ['shi_tanaka2020'],
    'grenke2025': ['grenke2025'],
    'singh2017': ['singh2017', 'holten2014'],
    'singh2017_caupin2019': ['singh2017', 'caupin2019'],
    'singh2017_duska2020': ['singh2017', 'duska2020'],
    'moynihan1996species': ['moynihan1996'],
    'mishima2023equation': ['mishima2023'],
    'water1': ['journaux2020'],
    'IAPWS95': ['wagner2002'],
    'vapor': ['wagner2002', 'wagner2011', 'wagner1993'],
    'pcsaft_water': ['diamantonis2011', 'novak2024', 'gross2001'],
    'novak2024_pcsaft_ts_tv': ['novak2024', 'gross2001'],
    'novak2024_pcsaft_ts_t': ['novak2024', 'novak2025', 'gross2001'],
    'novak2025_pcsaft_ts_caf': ['novak2025', 'gross2001'],
    'singh2016_tip4p2005': ['singh2016'],
    'biddle2017_tip4p2005': ['biddle2017'],
    'holten2014_st2_i_mf': ['holten2014st2'],
    'holten2014_st2_ii_mf': ['holten2014st2'],
    'holten2014_st2_i_crossover': ['holten2014st2'],
    'holten2014_st2_ii_crossover': ['holten2014st2'],
    'holten2013_mw': ['holten2013'],
    'holten2013_mw_n6': ['holten2013'],
    'holten2013_mw_weak_crystallization': ['holten2013'],
    'gartner2020_scan': ['gartner2020'],
    'weis2022_wail': ['weis2022'],
    'eltareb2022_qtip4pf_h2o_classical': ['eltareb2022'],
    'eltareb2022_qtip4pf_h2o_quantum': ['eltareb2022'],
    'eltareb2022_qtip4pf_d2o_quantum': ['eltareb2022'],
    'zorzi2024_mlbop': ['zorzi2024'],
    'yu2023_tip4p2005': ['yu2023'],
    'handle2018_tip4p2005': ['handle2018'],
    'eltareb2024_qtip4pf_classical': ['eltareb2024'],
}

REFERENCE_GROUPS = [
    ('Equations of state for real water',
     ['holten2014', 'caupin2019', 'kim2017', 'duska2020', 'shi_tanaka2020', 'grenke2025', 'moynihan1996',
      'mishima2023']),
    ('Transport', ['singh2017']),
    ('PC-SAFT', ['gross2001', 'diamantonis2011', 'novak2024', 'novak2025']),
    ('Fits to simulations of model waters',
     ['singh2016', 'biddle2017', 'holten2014st2', 'holten2013', 'gartner2020', 'weis2022', 'eltareb2022',
      'zorzi2024', 'yu2023', 'handle2018', 'eltareb2024']),
    ('Reference equations, ices and vapor',
     ['wagner2002', 'journaux2020', 'french2015', 'feistel2006', 'wagner2011', 'wagner1993', 'murphy2005']),
]


def doi_url(doi):
    return f'https://doi.org/{doi}'
