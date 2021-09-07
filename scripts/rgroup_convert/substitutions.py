'''
Define common substitutions for chemical shorthand
Note: does not include R groups or halogens as X

Each tuple records the
    string_tO_substitute, smarts_to_match, probability
'''
substitutions = [
    ('[OAc]', '[OH0;X2]C(=O)[CH3]', 0.8),
    ('[Ac]', 'C(=O)[CH3]', 0.1),
    
    ('[OBz]', '[OH0;D2]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', 0.7), # Benzoyl
    ('[Bz]', 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', 0.2), # Benzoyl
    
    ('[OBn]', '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', 0.7), # Benzyl
    ('[Bn]', '[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', 0.2), # Benzyl
    
    ('[NHBoc]', '[NH1;D2]C(=O)OC([CH3])([CH3])[CH3]', 0.9),
    ('[NBoc]', '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', 0.9),
    ('[Boc]', 'C(=O)OC([CH3])([CH3])[CH3]', 0.2),
    
    ('[Cbm]', 'C(=O)[NH2;D1]', 0.2),
    ('[Cbz]', 'C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', 0.4),
    ('[Cy]', '[CH1;X3]1[CH2][CH2][CH2][CH2][CH2]1', 0.3),
    ('[Fmoc]', 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3', 0.6),
    ('[Mes]', '[cH0]1c([CH3])cc([CH3])cc([CH3])1', 0.5),
    ('[OMs]', '[OH0;D2]S(=O)(=O)[CH3]', 0.8),
    ('[Ms]', 'S(=O)(=O)[CH3]', 0.2),
    ('[Ph]', '[cH0]1[cH][cH][cH1][cH][cH]1', 0.7),
    ('[Py]', '[cH0]1[n;+0][cH1][cH1][cH1][cH1]1', 0.1),
    ('[Suc]', 'C(=O)[CH2][CH2]C(=O)[OH]', 0.2),
    ('[TBS]', '[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', 0.5),
    ('[TBZ]', 'C(=S)[cH]1[cH][cH][cH1][cH][cH]1', 0.2),
    ('[OTf]', '[OH0;D2]S(=O)(=O)C(F)(F)F', 0.8),
    ('[Tf]', 'S(=O)(=O)C(F)(F)F', 0.2),
    ('[TFA]', 'C(=O)C(F)(F)F', 0.3),
    ('[TMS]', '[Si]([CH3])([CH3])[CH3]', 0.5),
    ('[Ts]', 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', 0.6), # Tos
    
    # Alkyl chains
    ('[OMe]', '[OH0;D2][CH3;D1]', 0.3),
    ('[Me]', '[CH3;D1]', 0.1),
    ('[OEt]', '[OH0;D2][CH2;D2][CH3]', 0.5),
    ('[Et]', '[CH2;D2][CH3]', 0.2),
    ('[Pr]', '[CH2;D2][CH2;D2][CH3]', 0.1),
    ('[Bu]', '[CH2;D2][CH2;D2][CH2;D2][CH3]', 0.1),
    
    # Branched
    ('[iPr]', '[CH1;D3]([CH3])[CH3]', 0.1),
    ('[iBu]', '[CH2;D2][CH1;D3]([CH3])[CH3]', 0.1),
    ('[OtBu]', '[OH0;D2][CH0]([CH3])([CH3])[CH3]', 0.7),
    ('[tBu]', '[CH0]([CH3])([CH3])[CH3]', 0.3),
    
    # Other shorthands (MIGHT NOT WANT ALL OF THESE)
    ('[CF3]', '[CH0;D4](F)(F)F', 0.5),
    ('[CO2H]', 'C(=O)[OH]', 0.2), # COOH
    ('[COOH]', 'C(=O)[OH]', 0.2), # COOH
    ('[CN]', 'C#[ND1]', 0.1),
    ('[OCH3]', '[OH0;D2][CH3]', 0.2),
    ('[SO3H]', 'S(=O)(=O)[OH]', 0.4),
]