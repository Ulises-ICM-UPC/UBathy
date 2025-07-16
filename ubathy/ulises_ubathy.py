#
# Tue Jul 15 15:56:31 2025, extract from Ulises by Gonzalo Simarro and Daniel Calvete
#
import cv2
import datetime
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
import shutil
import string
import subprocess
import warnings
warnings.filterwarnings("ignore")

#
def AllVar2DHor_2410(allVar, nc, nr, zr, rangeC):  # lm:2025-03-21; lr:2025-07-02
    if rangeC == 'close':
        dHor = {'nc': nc, 'nr': nr, 'zr': zr, 'range': rangeC}
        rEarth, oHorizon = REarth_2410(), OHorizon_2410()
        dHor = dHor | {'rEarth': rEarth, 'oHorizon': oHorizon}
        allVarKeys = AllVarKeys(rangeC)
        dAllVar = Array2Dictionary(allVarKeys, allVar)
        ef = UnitVectors_2502(dAllVar)[-1]  # dAngVar < dAllVar
        Pa11 = ArrayPx_2410(dAllVar, dAllVar, rangeC)  # {dExtVar, dCaSVar} < dAllVar
        zr = min(zr, dAllVar['zc'] - 0.1 / rEarth)
        a0, b0 = dAllVar['zc'] - 2 * zr, np.sqrt(2 * (dAllVar['zc'] - zr) * rEarth)
        den = max(np.hypot(ef[0], ef[1]), 1.e-14)  # WATCH OUT: epsilon
        xA = dAllVar['xc'] + b0 * ef[0] / den
        yA = dAllVar['yc'] + b0 * ef[1] / den
        zA = -a0
        dHor = dHor | {'xA': xA, 'yA': yA, 'zA': zA}
        ac, bc = Pa11[0] * xA + Pa11[1] * yA + Pa11[2] * zA + Pa11[3], -Pa11[0] * ef[1] + Pa11[1] * ef[0]
        ar, br = Pa11[4] * xA + Pa11[5] * yA + Pa11[6] * zA + Pa11[7], -Pa11[4] * ef[1] + Pa11[5] * ef[0]
        ad, bd = Pa11[8] * xA + Pa11[9] * yA + Pa11[10] * zA + 1, -Pa11[8] * ef[1] + Pa11[9] * ef[0]
        ccUh1, crUh1, ccUh0 = br * ad - bd * ar, bd * ac - bc * ad, bc * ar - br * ac
        den = max(np.hypot(ccUh1, crUh1), 1.e-14)  # WATCH OUT: epsilon
        ccUh1, crUh1, ccUh0 = [item / den for item in [ccUh1, crUh1, ccUh0]]
        crUh1 = ClipWithSign(crUh1, 1.e-14, np.inf)  # WATCH OUT: epsilon
        dHor = dHor | {'ccUh1': ccUh1, 'crUh1': crUh1, 'ccUh0': ccUh0}
        cUhs = np.linspace(-0.1 * nc, +1.1 * nc, 31)
        rUhs = CUh2RUh_2410(cUhs, dHor)
        cDhs, rDhs, possG = CURU2CDRD_2410(cUhs, rUhs, dAllVar, dAllVar, rangeC, rtrnPossG=True)  # no nc nor nr for possG; just well recovered
        if len(possG) < len(cUhs):  # WATCH OUT
            dHor['ccDh'] = np.zeros(oHorizon + 1)
            dHor['ccDh'][0] = -99  # WATCH OUT: epsilon; constant
        else:
            A = np.ones((len(possG), oHorizon + 1))  # IMP*: initialize with ones
            for n in range(1, oHorizon + 1):  # IMP*: increasing
                A[:, n] = cDhs[possG] ** n
            b = rDhs[possG]
            try:
                AT = np.transpose(A)
                dHor['ccDh'] = np.linalg.solve(np.dot(AT, A), np.dot(AT, b))
                assert np.max(np.abs(b - np.dot(A, dHor['ccDh']))) < 5.e-1  # WATCH OUT: assert meant for try
            except Exception:
                dHor['ccDh'] = np.zeros(oHorizon + 1)
                dHor['ccDh'][0] = -99  # WATCH OUT: epsilon; constant
    elif rangeC == 'long':
        dHor = {}  # WATCH OUT: missing
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return dHor
def AllVar2DMCS_2410(allVar, nc, nr, rangeC, incHor=True, zr=0.):  # lm:2025-02-17; lr:2025-07-02
    dMCS = {'rangeC': rangeC, 'nc': nc, 'nr': nr}
    allVarKeys = AllVarKeys(rangeC)
    dAllVar = Array2Dictionary(allVarKeys, allVar)
    dMCS = dMCS | {'allVar': allVar, 'allVarKeys': allVarKeys, 'dAllVar': dAllVar}
    dMCS = dMCS | dAllVar  # IMP*: keys
    if rangeC == 'close':
        px = np.asarray([dAllVar['xc'], dAllVar['yc'], dAllVar['zc']])
        dMCS = dMCS | {'px': px, 'pc': px}
    elif rangeC == 'long':
        px = np.asarray([dAllVar['x0'], dAllVar['y0'], dAllVar['z0']])
        dMCS = dMCS | {'px': px, 'p0': px}
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    R = MatrixR_2502(dAllVar)  # dAngVar < dAllVar
    dMCS = dMCS | {'R': R}
    eu, ev, ef = UnitVectors_2502(dAllVar)  # dAngVar < dAllVar
    dMCS = dMCS | {'eu': eu, 'eux': eu[0], 'euy': eu[1], 'euz': eu[2]}
    dMCS = dMCS | {'ev': ev, 'evx': ev[0], 'evy': ev[1], 'evz': ev[2]}
    dMCS = dMCS | {'ef': ef, 'efx': ef[0], 'efy': ef[1], 'efz': ef[2]}
    Px = ArrayPx_2410(dAllVar, dAllVar, rangeC)  # dExtVar, dCaSVar < dAllVar
    if rangeC == 'close':
        dMCS = dMCS | {'Px': Px, 'Pa11': Px}
    elif rangeC == 'long':
        dMCS = dMCS | {'Px': Px, 'Po8': Px}
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    if incHor:
        dHor = AllVar2DHor_2410(allVar, nc, nr, zr, rangeC)
        dMCS = dMCS | {'dHor': dHor}
        dMCS = dMCS | dHor  # IMP*: keys
    return dMCS
def AllVarKeys(rangeC):  # 2010-01-01; lr:2025-04-28; lr:2025-07-11
    if rangeC == 'close':
        allVarKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']  # WATCH OUT: cannot be changed
        if True:  # avoidable check for readability
            assert len(allVarKeys) == 14
    elif rangeC == 'long':
        allVarKeys = ['x0', 'y0', 'z0', 'ph', 'sg', 'ta', 'sc', 'sr']  # WATCH OUT: cannot be changed
        if True:  # avoidable check for readability
            assert len(allVarKeys) == 8
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return allVarKeys
def ApplyAffineA01_2504(A01, xs0, ys0):  # 1900-01-01; lm:2025-05-06; lm:2025-06-27
    xs1 = A01[0] * xs0 + A01[1] * ys0 + A01[2]
    ys1 = A01[3] * xs0 + A01[4] * ys0 + A01[5]
    return xs1, ys1
def Array2Dictionary(keys, theArray):  # 1900-01-01; lm:2025-05-01; lr:2025-07-11
    if not (len(set(keys)) == len(keys) == len(theArray)):
        raise Exception("Invalid input: 'keys' and 'theArray' must have the same length")
    if isinstance(theArray, (list, np.ndarray)):
        theDictionary = dict(zip(keys, theArray))
    else:
        raise Exception("Invalid input: 'theArray' must be a list or a NumPy ndarray")
    return theDictionary
def ArrayPx_2410(dExtVar, dCaSVar, rangeC):  # 1900-01-01; lm:2025-04-30; lm:2025-06-21
    Rt = MatrixRt_2410(dExtVar, rangeC)
    K = MatrixK_2410(dCaSVar, rangeC)
    P = np.dot(K, Rt)
    if False:  # avoidable check for readability
        assert P.shape == (3, 4)
    if rangeC == 'close':
        P23 = ClipWithSign(P[2, 3], 1.e-14, np.inf)  # WATCH OUT: epsilon
        Pa = P / P23
        Px = np.hstack([Pa[0, :4], Pa[1, :4], Pa[2, :3]])
    elif rangeC == 'long':
        Po = P[:2, :]
        Px = np.hstack([Po[0, :4], Po[1, :4]])
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return Px
def CDRD2CURU_2410(cDs, rDs, dCaSVar, dDtrVar, rangeC, rtrnPossG=False, nc=None, nr=None, margin=0):  # undistort; potentially expensive; 2010-01-01; lm:2025-05-01; lm:2025-07-01
    uDas, vDas = CR2UaVa_2410(cDs, rDs, dCaSVar, rangeC)
    uUas, vUas, possG = UDaVDa2UUaVUa_2410(uDas, vDas, dDtrVar, rangeC, rtrnPossG=rtrnPossG)  # WATCH OUT: potentially expensive
    cUs, rUs = UaVa2CR_2410(uUas, vUas, dCaSVar, rangeC)
    if len(possG) > 0 and rtrnPossG and nc is not None and nr is not None:
        cDsG, rDsG = [item[possG] for item in [cDs, rDs]]
        possGInPossG = CR2PossWithinImage_2502(cDsG, rDsG, nc, nr, margin=margin, case='')
        possG = possG[possGInPossG]
    return cUs, rUs, possG
def CDRD2XYZ_2410(cDs, rDs, planes, dMCS, rtrnPossG=False, margin=0):  # potentially expensive; 2010-01-01; lm:2025-05-05; lr:2025-07-02
    Px, rangeC, dAllVar, ef, nc, nr = [dMCS[item] for item in ['Px', 'rangeC', 'dAllVar', 'ef', 'nc', 'nr']]
    cUs, rUs, possG = CDRD2CURU_2410(cDs, rDs, dAllVar, dAllVar, rangeC, rtrnPossG=rtrnPossG, nc=nc, nr=nr, margin=margin)
    xs, ys, zs, possGH = CURU2XYZ_2410(cUs, rUs, planes, Px, rangeC, rtrnPossG=rtrnPossG, dCamVar=dAllVar, ef=ef, nc=nc, nr=nr)
    possG = np.intersect1d(possG, possGH, assume_unique=True)
    return xs, ys, zs, possG
def CDRDZ2XY_2410(cDs, rDs, zs, dMCS, rtrnPossG=False, margin=0):  # potentially expensive; 2010-01-01; lm:2025-05-05; lr:2025-07-02
    planes = {'pxs': np.zeros(zs.shape), 'pys': np.zeros(zs.shape), 'pzs': np.ones(zs.shape), 'pts': -zs}
    xs, ys, _, possG = CDRD2XYZ_2410(cDs, rDs, planes, dMCS, rtrnPossG=rtrnPossG, margin=margin)
    return xs, ys, possG
def CR2CRIntegerWithinImage_2502(cs, rs, nc, nr, margin=0, case='round'):  # 1900-01-01; lm:2025-05-28; lr:2025-07-07
    csI, rsI = CR2CRInteger_2504(cs, rs, case=case)
    possW = CR2PossWithinImage_2502(csI, rsI, nc, nr, margin=margin, case='')
    csIW, rsIW = [item[possW] for item in [csI, rsI]]
    return csIW, rsIW
def CR2CRInteger_2504(cs, rs, case='round'):  # 1900-01-01; lm:2025-05-28; lr:2025-07-13
    if case == 'round':
        csI = np.round(cs).astype(int)
        rsI = np.round(rs).astype(int)
    elif case == 'floor':
        csI = np.floor(cs).astype(int)
        rsI = np.floor(rs).astype(int)
    else:
        raise Exception("Invalid input: 'case' ('{}') must be 'round' or 'floor'".format(case))
    return csI, rsI
def CR2PossWithinImage_2502(cs, rs, nc, nr, margin=0, case=''):  # 1900-01-01; lm:2025-05-28; lr:2025-07-13
    if len(cs) == 0 or len(rs) == 0:
        return np.asarray([], dtype=int)
    if case in ['round', 'floor']:
        cs, rs = CR2CRInteger_2504(cs, rs, case=case)
    cMin, cMax = -1/2 + margin, nc-1/2 - margin  # IMP*
    rMin, rMax = -1/2 + margin, nr-1/2 - margin  # IMP*
    possW = np.where((cs > cMin) & (cs < cMax) & (rs > rMin) & (rs < rMax))[0]  # WATCH OUT: "<" and ">" for safety
    return possW
def CR2UaVa_2410(cs, rs, dCaSVar, rangeC):  # lm:1900-01-01; lr:2025-05-01; lr:2025-06-30
    if rangeC == 'close':
        sca, sra = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sca', 'sra']]  # WATCH OUT: epsilon
        uas = (cs - dCaSVar['oc']) * sca
        vas = (rs - dCaSVar['or']) * sra
    elif rangeC == 'long':
        sc, sr = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sc', 'sr']]  # WATCH OUT: epsilon
        uas = (cs - dCaSVar['oc']) * sc  # uas are actually us in this case
        vas = (rs - dCaSVar['or']) * sr  # vas are actually vs in this case
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return uas, vas
def CURU2CDRD_2410(cUs, rUs, dCaSVar, dDtrVar, rangeC, rtrnPossG=False, nc=None, nr=None, margin=0):  # distort; 2010-01-01; lm:2025-05-01; lm:2025-07-01
    uUas, vUas = CR2UaVa_2410(cUs, rUs, dCaSVar, rangeC)
    uDas, vDas, possG = UUaVUa2UDaVDa_2410(uUas, vUas, dDtrVar, rangeC, rtrnPossG=rtrnPossG)
    cDs, rDs = UaVa2CR_2410(uDas, vDas, dCaSVar, rangeC)
    if len(possG) > 0 and rtrnPossG and nc is not None and nr is not None:
        cDsG, rDsG = [item[possG] for item in [cDs, rDs]]
        possGInPossG = CR2PossWithinImage_2502(cDsG, rDsG, nc, nr, margin=margin, case='')
        possG = possG[possGInPossG]
    return cDs, rDs, possG
def CURU2XYZ_2410(cUs, rUs, planes, Px, rangeC, rtrnPossG=False, dCamVar=None, ef=None, nc=None, nr=None):  # 2000-01-01; lm:2025-05-01; lr:2025-07-02
    if rangeC == 'close':
        A11s, A12s, A13s, bb1s = Px[0] - cUs * Px[8], Px[1] - cUs * Px[9], Px[2] - cUs * Px[10], cUs - Px[3]
        A21s, A22s, A23s, bb2s = Px[4] - rUs * Px[8], Px[5] - rUs * Px[9], Px[6] - rUs * Px[10], rUs - Px[7]
        A31s, A32s, A33s, bb3s = planes['pxs'], planes['pys'], planes['pzs'], -planes['pts']
    elif rangeC == 'long':
        A11s, A12s, A13s, bb1s = Px[0], Px[1], Px[2], cUs - Px[3]
        A21s, A22s, A23s, bb2s = Px[4], Px[5], Px[6], rUs - Px[7]
        A31s, A32s, A33s, bb3s = planes['pxs'], planes['pys'], planes['pzs'], -planes['pts']
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    dens = A11s * (A22s * A33s - A23s * A32s) + A12s * (A23s * A31s - A21s * A33s) + A13s * (A21s * A32s - A22s * A31s)
    dens = ClipWithSign(dens, 1.e-14, np.inf)  # WATCH OUT: epsilon
    xs = (bb1s * (A22s * A33s - A23s * A32s) + A12s * (A23s * bb3s - bb2s * A33s) + A13s * (bb2s * A32s - A22s * bb3s)) / dens
    ys = (A11s * (bb2s * A33s - A23s * bb3s) + bb1s * (A23s * A31s - A21s * A33s) + A13s * (A21s * bb3s - bb2s * A31s)) / dens
    zs = (A11s * (A22s * bb3s - bb2s * A32s) + A12s * (bb2s * A31s - A21s * bb3s) + bb1s * (A21s * A32s - A22s * A31s)) / dens
    if rtrnPossG:
        cUsR, rUsR, possG = XYZ2CURU_2410(xs, ys, zs, Px, rangeC, rtrnPossG=rtrnPossG, dCamVar=dCamVar, ef=ef, nc=nc, nr=nr)
        cUsG, rUsG, cUsRG, rUsRG = [item[possG] for item in [cUs, rUs, cUsR, rUsR]]
        possGInPossG = np.where(np.hypot(cUsRG - cUsG, rUsRG - rUsG) < 1.e-6)[0]  # WATCH OUT: epsilon
        possG = possG[possGInPossG]
        xsG, ysG, zsG = [item[possG] for item in [xs, ys, zs]]
        pxsG, pysG, pzsG, ptsG = [planes[item][possG] for item in ['pxs', 'pys', 'pzs', 'pts']]
        possGInPossG = np.where(np.abs(pxsG * xsG + pysG * ysG + pzsG * zsG + ptsG) < 1.e-6)[0]  # WATCH OUT: epsilon
        possG = possG[possGInPossG]
    else:
        possG = np.asarray([], dtype=int)
    return xs, ys, zs, possG
def CUh2RUh_2410(cUhs, dHor, eps=1.e-12):  # 2000-01-01; lm:2025-04-25; lr:2025-07-02
    crUh1 = ClipWithSign(dHor['crUh1'], eps, np.inf)
    rUhs = - (dHor['ccUh1'] * cUhs + dHor['ccUh0']) / crUh1
    return rUhs
def CleanAFld_2504(pathFld):  # 2000-01-01; lm:2025-04-19; lr:2025-06-27
    if not os.path.exists(pathFld):
        return None
    for item in os.listdir(pathFld):
        pathItem = os.path.join(pathFld, item)
        if os.path.isfile(pathItem) or os.path.islink(pathItem):
            os.remove(pathItem)
        elif os.path.isdir(pathItem):
            shutil.rmtree(pathItem)
    return None
def ClipWithSign(xs, x0, x1):  # 1900-01-01; lm:2025-05-28; lr:2025-07-14
    if not (0 < x0 < x1):
        raise Exception("Invalid input: invalid 'x0' and/or 'x1'; must satisfy 0 < x0 < x1")
    signs = np.where(np.sign(xs) == 0, 1, np.sign(xs))
    xs = signs * np.clip(np.abs(xs), x0, x1)
    return xs
def CloudOfPoints2PossInsidePolyline_2508(xsC, ysC, polyline, eps_xy=1e-9):  # 1900-01-01; lm:2025-06-12; lr:2025-07-02
    xsP, ysP = polyline['xs'], polyline['ys']
    nP = len(xsP)
    if nP < 3:
        raise Exception("Invalid input: polyline must have at least 3 points")
    if np.hypot(xsP[0] - xsP[-1], ysP[0] - ysP[-1]) < eps_xy:  # open the polyline if it comes closed
        xsP, ysP = xsP[:-1], ysP[:-1]
        nP = len(xsP)
    possIn = []
    for posC, (xC, yC) in enumerate(zip(xsC, ysC)):
        count = 0
        for posP in range(nP):
            x0, y0 = xsP[posP], ysP[posP]
            x1, y1 = xsP[(posP + 1) % nP], ysP[(posP + 1) % nP]
            if ((y0 > yC) != (y1 > yC)):
                xcross = (x1 - x0) * (yC - y0) / (y1 - y0 + 1e-14) + x0  # WATCH OUT; epsilon; it was 1.e-20
                if xC < xcross + eps_xy:
                    count += 1
            dx, dy = x1 - x0, y1 - y0
            num = abs(dy * (xC - x0) - dx * (yC - y0))
            den = np.hypot(dx, dy)
            if den > 0 and num < eps_xy * den:
                dot = (xC - x0) * (x1 - x0) + (yC - y0) * (y1 - y0)
                if 0 <= dot <= den ** 2:
                    count = 1
                    break
        if count % 2 == 1:
            possIn.append(posC)
    possIn = np.asarray(possIn, dtype=int)
    return possIn
def CloudOfPoints2RectangleAux_2504(angle, xs, ys, margin=0.):  # 1900-01-01; lm:2025-05-06; lr:2025-06-30
    lDs = - np.sin(angle) * xs + np.cos(angle) * ys  # signed-distances to D-line dir = (+cos, +sin) through origin (0, 0); positive above
    lD0 = {'lx': -np.sin(angle), 'ly': +np.cos(angle), 'lt': -(np.min(lDs)-margin)}
    lD1 = {'lx': -np.sin(angle), 'ly': +np.cos(angle), 'lt': -(np.max(lDs)+margin)}
    lPs = + np.cos(angle) * xs + np.sin(angle) * ys  # signed-distances to P-line dir = (+sin, -cos) through origin (0, 0); positive right
    lP0 = {'lx': +np.cos(angle), 'ly': +np.sin(angle), 'lt': -(np.min(lPs)-margin)}
    lP1 = {'lx': +np.cos(angle), 'ly': +np.sin(angle), 'lt': -(np.max(lPs)+margin)}
    xcs, ycs = [np.zeros(4) for _ in range(2)]
    xcs[0], ycs[0] = IntersectionOfTwoLines_2506(lD0, lP0)[:2]
    xcs[1], ycs[1] = IntersectionOfTwoLines_2506(lP0, lD1)[:2]
    xcs[2], ycs[2] = IntersectionOfTwoLines_2506(lD1, lP1)[:2]
    xcs[3], ycs[3] = IntersectionOfTwoLines_2506(lP1, lD0)[:2]
    area = (np.max(lDs) - np.min(lDs) + 2 * margin) * (np.max(lPs) - np.min(lPs) + 2 * margin)  # IMP*
    return xcs, ycs, area
def CloudOfPoints2Rectangle_2504(xs, ys, margin=0.):  # 1900-01-01; lm:2025-07-07; lr:2025-07-10
    xcs, ycs, area = None, None, np.inf
    for angleH in np.linspace(0, np.pi / 2, 1000):  # WATCH OUT; a capon
        xcsH, ycsH, areaH = CloudOfPoints2RectangleAux_2504(angleH, xs, ys, margin=margin)  # already oriented clockwise
        if areaH < area:
            xcs, ycs, area = xcsH, ycsH, areaH
    pos0 = np.argmin(np.hypot(xcs - xs[0], ycs - ys[0]))
    xcs, ycs = np.roll(xcs, -pos0), np.roll(ycs, -pos0)
    if True:  # avoidable check for readability
        assert len(xcs) == len(ycs) == 4
    return xcs, ycs, area
def Date2Datenum_2504(date):  # lm:2010-01-01; lr:2025-07-08
    datenum = Datetime2Datenum_2504(Date2Datetime_2504(date))
    return datenum
def Date2Datetime_2504(date):  # lm:2025-04-09; lr:2025-07-08
    year, month, day = int(date[:4]), int(date[4:6]), int(date[6:8])
    hour, minute, second = int(date[8:10]), int(date[10:12]), int(date[12:14])
    if len(date) == 17:
        microsecond = int(date[14:17]) * 1000
    elif len(date) == 20:
        microsecond = int(date[14:])
    else:
        raise Exception("Invalid input: 'date' ('{}') must have length 17 or 20".format(date))
    theDatetime = datetime.datetime(year, month, day, hour, minute, second, microsecond)
    if False:  # avoidable check for readability
        assert Datetime2Date_2504(theDatetime, length=len(date)) == date
    return theDatetime
def Datetime2Date_2504(theDatetime, length=17):  # lm:2025-04-09; lr:2025-07-08
    year, month, day = str(theDatetime.year).zfill(4), str(theDatetime.month).zfill(2), str(theDatetime.day).zfill(2)
    hour, minute, second = str(theDatetime.hour).zfill(2), str(theDatetime.minute).zfill(2), str(theDatetime.second).zfill(2)
    if length == 17:
        millisecond = str(int(theDatetime.microsecond / 1.e+3)).zfill(3)  # IMP*; do not round
        date = '{}{}{}{}{}{}{}'.format(year, month, day, hour, minute, second, millisecond)
    elif length == 20:
        microsecond = str(int(theDatetime.microsecond)).zfill(6)
        date = '{}{}{}{}{}{}{}'.format(year, month, day, hour, minute, second, microsecond)
    else:
        raise Exception("Invalid input: 'datetime' has invalid value ('{}')".format(theDatetime))
    return date
def Datetime2Datenum_2504(theDatetime):  # lm:2025-04-09; lr:2025-07-08
    base = theDatetime.toordinal()
    frac = (theDatetime - datetime.datetime(theDatetime.year, theDatetime.month, theDatetime.day)).total_seconds()/86400
    datenum = base + frac + 366  # the 366 is to start in year 0, as matlab
    return datenum
def DisplayCRInImage_2504(img, cs, rs, margin=0, factor=1., colors=None, pathOut=None):  # 1900-01-01; lm:2025-05-28; lr:2025-06-27
    img = PathImgOrImg2Img(img)
    nr, nc = img.shape[:2]
    imgOut = img.copy()  # IMP*: otherwise, if imgNew = DisplayCRInImage_2504(img, ...) then img is also modified
    csIW, rsIW = CR2CRIntegerWithinImage_2502(cs, rs, nc, nr, margin=margin)
    if len(csIW) == 0:
        return imgOut
    if colors is None:
        colors = [[0, 0, 0]]
    colors = (colors * ((len(csIW) + len(colors) - 1) // len(colors)))[:len(csIW)]
    radius = int(factor * np.sqrt(nc * nr) / 2.e+2 + 1)
    for pos in range(len(csIW)):
        cv2.circle(imgOut, (csIW[pos], rsIW[pos]), radius, colors[pos], -1)
    if pathOut is not None:
        os.makedirs(os.path.dirname(pathOut), exist_ok=True)
        cv2.imwrite(pathOut, imgOut)
    return imgOut
def EstimateScatterSize_2506(xs, ys, fw, fh, dpi):  # 2000-01-01; lm:2025-07-02; lm:2025-07-14
    dx, dy = np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)
    ipux, ipuy = fw / dx, fh / dy  # inches per unit x/y
    dpuxy = dpi * min(ipux, ipuy)  # dots per unit x/y = dots per inch (dpi) * inches per unit x/y
    size = (dpuxy * min(dx, dy)) ** 2 / len(xs)
    size = size * 0.49  # a capon; 0.7**2
    return size
def FindAffineA01_2504(xs0, ys0, xs1, ys1):  # 1900-01-01; lm:2025-05-06; lr:2025-06-26
    minNOfPoints = 3
    if not (len(xs0) == len(ys0) == len(xs1) == len(ys1) >= minNOfPoints):
        return None
    A, b = np.zeros((2 * len(xs0), 6)), np.zeros(2 * len(xs0))  # IMP*; initialize with zeroes
    poss0, poss1 = Poss0AndPoss1InFind2DTransform_2504(len(xs0))
    A[poss0, 0], A[poss0, 1], A[poss0, 2], b[poss0] = xs0, ys0, np.ones(xs0.shape), xs1
    A[poss1, 3], A[poss1, 4], A[poss1, 5], b[poss1] = xs0, ys0, np.ones(xs0.shape), ys1
    try:
        AT = np.transpose(A)
        A01 = np.linalg.solve(np.dot(AT, A), np.dot(AT, b))
    except Exception:  # aligned points
        return None
    return A01
def GHBathyExtractVideoToPathFldFrames(pathFldMain, video, active=True, extsVids=['mp4', 'avi', 'mov'], fps=0., round=True, stamp='millisecond', extImg='png', overwrite=False):  # lm:2025-06-27; lr:2025-07-14
    pathFldVid = os.path.join(pathFldMain, 'data', 'videos', video)
    pathFldDPlanviews, pathFldDFrames = [os.path.join(pathFldVid, item) for item in ['planviews', 'frames']]
    if os.path.exists(pathFldDPlanviews) and os.path.exists(pathFldDFrames):
        raise Exception("Invalid input: both 'frames' and 'planviews' found at '{}'".format(os.sep.join(pathFldVid.split(os.sep)[-2:])))  # WATCH OUT: [-2:] formatting
    pathFldFrames = next((item for item in [pathFldDPlanviews, pathFldDFrames] if os.path.exists(item)), None)
    pathVid = GHLookFor0Or1PathVideoOrFail(pathFldVid, extsVids=extsVids)
    if pathVid == '':  # no video, there must be frames in pathFldFrames
        if pathFldFrames is None or len(os.listdir(pathFldFrames)) == 0:
            raise Exception("Unexpected condition: neither video nor frames found at '{}'".format(os.sep.join(pathFldVid.split(os.sep)[-2:])))  # WATCH OUT: [-2:] formatting
    else:
        pathFldFrames = os.path.join(pathFldMain, 'scratch', 'frames', video)  # IMP*: nomenclature
        if active and (not os.path.exists(pathFldFrames) or len(os.listdir(pathFldFrames)) == 0 or overwrite):
            GHExtractVideo(pathVid, pathFldFrames, fps=fps, round=round, stamp=stamp, extImg=extImg, overwrite=overwrite)
    return pathFldFrames
def GHBathyLoadVideos4Dates(pathFldMain):  # lm:2025-07-08; lr:2025-07-14
    pathJson = os.path.join(pathFldMain, 'data', 'videos4dates.json')  # IMP*: nomenclature
    try:
        with open(pathJson, 'r') as f:
            videos4dates = json.load(f)
    except Exception as eTMP:
        raise Exception("Invalid input: failed to read '{}': {}".format(pathJson, eTMP))
    return videos4dates
def GHBathyParAndVideos(pathFldMain):  # lm:2025-07-08; lr:2025-07-14
    par = GHLoadPar(pathFldMain) | {'videos4dates': GHBathyLoadVideos4Dates(pathFldMain)}
    pathFldDVideos = os.path.join(pathFldMain, 'data', 'videos')
    if not os.path.exists(pathFldDVideos) or len([item.name for item in os.scandir(pathFldDVideos) if item.is_dir()]) == 0:
        raise Exception("Unexpected condition: no videos available to obtain bathymetries")
    videos_available = sorted([item.name for item in os.scandir(pathFldDVideos) if item.is_dir()])
    videos = sorted(set([item for sublist in par['videos4dates'].values() for item in sublist]))  # videos required
    videos_missing = set(videos) - set(videos_available)  # 2025-07-10
    if videos_missing:
        raise Exception("Invalid input: missing videos {}".format(videos_missing))
    if len(videos) == 0:
        raise Exception("Unexpected condition: no videos available to obtain bathymetries")
    return par, videos
def GHBathyPlotBath(pathImg, date, xs, ys, zbs, ezbs, min_depth, max_depth, fw, fh, fontsize, dpi=100, thereIsGT=False, zbsGT=None, title2='self error [m]'):  # lm:2025-07-09; lr:2025-07-13
    fw, fh = GHFwFh2FwFh(fw, fh, xs, ys)
    if thereIsGT:
        nOfC = 3
    else:
        nOfC = 2
    plt.figure(figsize=(nOfC*fw, fh))
    size = EstimateScatterSize_2506(xs, ys, fw, fh, dpi)
    plt.rcParams.update(LoadParamsC(fontsize=fontsize))
    plt.suptitle('date = {}'.format(date))
    plt.subplot(1, nOfC, 1)
    plt.title(r'$z_b$ [m]')
    plt.plot(np.min(xs), np.min(ys), 'w.'); plt.plot(np.max(xs), np.max(ys), 'w.')
    sct = plt.scatter(xs, ys, marker='o', c=zbs, edgecolor='none', vmin=-max_depth, vmax=-min_depth, s=size, cmap='gist_earth')
    plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal')
    plt.subplot(1, nOfC, 2)
    plt.title(title2)
    plt.plot(np.min(xs), np.min(ys), 'w.'); plt.plot(np.max(xs), np.max(ys), 'w.')
    sct = plt.scatter(xs, ys, marker='o', c=ezbs, vmin=0, vmax=1, edgecolor='none', s=size, cmap='jet')  # IMP*: vmax
    plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal')
    if thereIsGT:
        plt.subplot(1, nOfC, nOfC)
        plt.title('error [m]')
        plt.plot(np.min(xs), np.min(ys), 'w.'); plt.plot(np.max(xs), np.max(ys), 'w.')
        sct = plt.scatter(xs, ys, marker='o', c=zbs-zbsGT, vmin=-2, vmax=2, edgecolor='none', s=size, cmap='seismic_r')  # IMP*: vmin, vmax
        plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal')
    plt.tight_layout()
    os.makedirs(os.path.dirname(pathImg), exist_ok=True)
    plt.savefig(pathImg, dpi=dpi)
    plt.close()
    return None
def GHBathyPlotMesh(pathImg, xs, ys, fw, fh, fontsize, dpi=100, xsBoun=None, ysBoun=None):  # lm:2025-07-05; lr:2025-07-13
    if xsBoun is not None and ysBoun is not None and len(xsBoun) == len(ysBoun):  # boundary
        xsAll, ysAll = np.concatenate((xs, xsBoun)), np.concatenate((ys, ysBoun))
    else:
        xsAll, ysAll = xs, ys
    fw, fh = GHFwFh2FwFh(fw, fh, xsAll, ysAll)
    plt.figure(figsize=(fw, fh))
    plt.rcParams.update(LoadParamsC(fontsize=fontsize))
    size = np.sqrt(EstimateScatterSize_2506(xs, ys, fw, fh, dpi)) / 10  # IMP*: np.sqrt() and 10
    plt.plot(xs, ys, 'k.', markersize=size)
    if xsBoun is not None and ysBoun is not None and len(xsBoun) == len(ysBoun):  # boundary
        plt.plot(list(xsBoun) + [xsBoun[0]], list(ysBoun) + [ysBoun[0]], 'm-', lw=3*size)  # IMP*: 'm-' and 3*
    plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal')
    plt.tight_layout()
    os.makedirs(os.path.dirname(pathImg), exist_ok=True)
    plt.savefig(pathImg, dpi=dpi)
    plt.close()
    return None
def GHBathyPlotModes(pathImg, xs, ys, phases, amplitudes, T, fw, fh, fontsize, dpi=100):  # lm:2025-06-30; lr:2025-07-13
    fw, fh = GHFwFh2FwFh(fw, fh, xs, ys)
    size = EstimateScatterSize_2506(xs, ys, fw, fh, dpi)
    plt.figure(figsize=(2*fw, fh))
    plt.rcParams.update(LoadParamsC(fontsize=fontsize))
    plt.suptitle('T = {:.2f} s'.format(T))  # WATCH OUT: formatting
    plt.subplot(1, 2, 1)
    plt.title('phase [rad]')
    plt.plot(np.min(xs), np.min(ys), 'w.'); plt.plot(np.max(xs), np.max(ys), 'w.')
    sct = plt.scatter(xs, ys, marker='o', c=phases, edgecolor='none', s=size, vmin=-np.pi, vmax=np.pi, cmap='jet')
    plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal')
    plt.subplot(1, 2, 2)
    plt.title('amplitude [-]')
    plt.plot(np.min(xs), np.min(ys), 'w.'); plt.plot(np.max(xs), np.max(ys), 'w.')
    sct = plt.scatter(xs, ys, marker='o', c=amplitudes, edgecolor='none', vmin=0, vmax=0.10, s=size, cmap='jet')  # IMP*: from 0 to 0.10
    plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal')
    plt.tight_layout()
    os.makedirs(os.path.dirname(pathImg), exist_ok=True)
    plt.savefig(pathImg, dpi=dpi)
    plt.close()
    return None
def GHBathyPlotWavenumbers(pathImg, xsM, ysM, T, phasesM, xsK, ysK, RKs, ztsK, ks, gammas, stdGs, max_depth, fw, fh, fontsize, dpi=100):  # lm:2025-07-05; lr:2025-07-14
    fw, fh = GHFwFh2FwFh(fw, fh, xsK, ysK)  # IMP*: mesh_K
    plt.figure(figsize=(3*fw, len(RKs)*fh))
    fontsize = min(fontsize + 2 * (len(RKs) - 2), 2 * fontsize)  # WATCH OUT: epsilon
    sizeM = EstimateScatterSize_2506(xsM, ysM, fw, fh, dpi)  # mesh_M
    sizeK = EstimateScatterSize_2506(xsK, ysK, fw, fh, dpi)  # mesh_K
    plt.rcParams.update(LoadParamsC(fontsize=fontsize))
    plt.suptitle('T = {:.2f} s'.format(T))
    for posRK in range(len(RKs)):
        try:
            plt.subplot(len(RKs), 3, 3*posRK+1)
            plt.title('phase [rad]')
            plt.plot(np.min(xsK), np.min(ysK), 'w.'); plt.plot(np.max(xsK), np.max(ysK), 'w.')
            sct = plt.scatter(xsM, ysM, marker='o', c=phasesM, vmin=-np.pi, vmax=np.pi, edgecolor='none', s=sizeM, cmap='jet')
            possTMP = np.where(np.hypot(xsM - np.mean(xsM), ysM - np.mean(ysM)) <= RKs[posRK])[0]
            plt.plot(xsM[possTMP], ysM[possTMP], 'wo', markersize=sizeM/2)  # WATCH OUT: epsilon, formatting
            plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal'); plt.xlim([np.min(xsK), np.max(xsK)])
            plt.subplot(len(RKs), 3, 3*posRK+2)
            plt.title(r'$z_b$ [m]')
            plt.plot(np.min(xsK), np.min(ysK), 'w.'); plt.plot(np.max(xsK), np.max(ysK), 'w.')
            zbs = np.mean(ztsK) - np.arctanh(np.clip(gammas[:, posRK], 0, 1-1.e-6)) / ks[:, posRK]  # gammas and ks have np.nan
            sct = plt.scatter(xsK, ysK, marker='o', c=zbs, vmin=-max_depth, vmax=0, edgecolor='none', s=sizeK, cmap='gist_earth')  # WATCH OUT: vmax, formatting
            plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal'); plt.xlim([np.min(xsK), np.max(xsK)])
            plt.subplot(len(RKs), 3, 3*posRK+3)
            plt.title(r'$\sigma_\gamma$ [-]')
            plt.plot(np.min(xsK), np.min(ysK), 'w.'); plt.plot(np.max(xsK), np.max(ysK), 'w.')
            sct = plt.scatter(xsK, ysK, marker='o', c=stdGs[:, posRK], vmin=0, vmax=0.2, edgecolor='none', s=sizeK, cmap='CMRmap_r')  # WATCH OUT: from 0 to 0.2, formatting
            plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal'); plt.xlim([np.min(xsK), np.max(xsK)])
        except Exception:  # WATCH OUT
            continue
    plt.tight_layout()
    os.makedirs(os.path.dirname(pathImg), exist_ok=True)
    plt.savefig(pathImg, dpi=dpi)
    plt.close()
    return None
def GHExtractVideo(pathVid, pathFldFrames, fps=0., round=True, stamp='millisecond', extImg='png', overwrite=False):  # 2010-01-01; lm:2025-06-26; lr:2025-07-11
    if os.path.exists(pathFldFrames):
        if overwrite:
            CleanAFld_2504(pathFldFrames)
        else:
            return None
    pathFldVid, fnVid = os.path.split(pathVid)
    fpsVid, nOfFramesVid = PathVid2FPS(pathVid), PathVid2NOfFrames(pathVid)
    lenVid = (nOfFramesVid - 1) / fpsVid
    fps = RecomputeFPS_2502(fps, fpsVid, round=round)
    isFpsTheSame = np.isclose(fps, fpsVid, rtol=1.e-6)
    if not isFpsTheSame:
        nOfFramesApp = lenVid * fps + 1  # ~ approximation of the number of frames of the new video
        if nOfFramesApp <= 3:  # WATCH OUT: epsilon
            raise Exception("Invalid input: requested fps is too low for video '{}'; please specify fps > {:.4f}".format(fnVid, 4 / lenVid))  # WATCH OUT: formatting, epsilon
    fnVidTMP = '000_{}{}'.format(''.join(random.choices(string.ascii_letters, k=20)), os.path.splitext(fnVid)[1])
    pathVidTMP = os.path.join(pathFldVid, fnVidTMP)
    if os.path.exists(pathVidTMP):  # miracle
        os.remove(pathVidTMP)
    if not isFpsTheSame:
        shutil.move(pathVid, pathVidTMP)  # pathVidTMP is a backup of the original
        Vid2VidModified_2504(pathVidTMP, pathVid, fps=fps, round=round)  # IMP*: this pathVid has the desired fps
    PathVid2AllFrames_2506(pathVid, pathFldFrames, stamp=stamp, extImg=extImg)
    if not isFpsTheSame:
        shutil.move(pathVidTMP, pathVid)  # pathVid is the original
    return None
def GHFwFh2FwFh(fw, fh, xs, ys):  # lm:2025-07-05; lr:2025-07-11
    dx, dy = np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)
    fh = max(fh, dy / (dx + 1.e-12) * fw)  # WATCH OUT: epsilon
    fw = dx / (dy + 1.e-12) * fh  # WATCH OUT: epsilon
    return fw, fh
def GHInform_2506(UCode, pathFldMain, par, pos, margin=0, sB='*', nFill=10):  # lm:2025-06-19; lr:2025-07-14
    text0 = 'session started for'
    text1 = 'session finished successfully for' 
    textC = os.sep.join(pathFldMain.split(os.sep)[-2:])  # WATCH OUT: [-2:] formatting
    if len(text1) > len(text0):
        l0, l1 = len(text1) - len(text0), 0
    else:
        l0, l1 = 0, len(text0) - len(text1)
    if pos == 0:
        msg = '{} {} /{}'.format(UCode, text0, textC)
        print("\n{:s}\n".format(msg.center(len(msg)+nFill+l0, '_')))
        print("Parameters:")
        PrintDictionary_2506(par, margin=margin, sB=sB)
    else:
        msg = '{} {} /{}'.format(UCode, text1, textC)
        print("\n{:s}\n".format(msg.center(len(msg)+nFill+l1, '_')))
    return None
def GHLoadDGit():  # lm:2025-06-30; lr:2025-07-13
    dGit = {}
    dGit |= {'ind': 2}
    dGit |= {'sOK': '\033[92m\u25CF\033[0m'}  # '\033[92m\u25CF\033[0m' '\U0001F7E2' '\033[92m✔\033[0m'
    dGit |= {'sWO': '\033[93m\u25CF\033[0m'}  # '\033[93m\u25CF\033[0m' '\U0001F7E0'
    dGit |= {'sKO': '\033[91m\u25CF\033[0m'}  # '\033[91m\u25CF\033[0m' '\U0001F534' '\033[31m✘\033[0m'
    dGit |= {'sB1': '\u2022'}  # bullet
    dGit |= {'sB2': '\u2023'}  # triangular bullet
    dGit |= {'sB3': '\u25E6'}  # white bullet
    dGit |= {'sB4': '\u2043'}  # hyphen bullet
    dGit |= {'sB5': '\u2219'}  # bullet operator
    dGit |= {'sB6': '\u25AA'}  # small black square
    dGit |= {'sB7': '\u25AB'}  # small white square
    dGit |= {'sB8': '\u25CF'}  # black circle
    dGit |= {'sB9': '\u25CB'}  # white circle
    dGit |= {'fontsize': 20}
    dGit |= {'fs': 8}  # figure size for scatter plot
    dGit |= {'fw': 10}  # figure width
    dGit |= {'fh': 4}  # figure height
    dGit |= {'dpiLQ': 100}
    dGit |= {'dpiHQ': 200}
    return dGit
def GHLoadPar(pathFldMain):  # lm:2025-07-08; lr:2025-07-14
    pathJson = os.path.join(pathFldMain, 'data', 'parameters.json')  # IMP*: nomenclature
    try:
        with open(pathJson, 'r') as f:
            par = json.load(f)
    except Exception as eTMP:
        raise Exception("Invalid input: failed to read '{}': {}".format(pathJson, eTMP))
    return par
def GHLookFor0Or1PathVideoOrFail(pathFld, extsVids=['mp4', 'avi', 'mov']):  # 2010-01-01; lm:2025-06-15; lr:2025-07-14
    pathsVids = [item.path for item in os.scandir(pathFld) if item.is_file() and os.path.splitext(item.name)[1][1:].lower() in extsVids]
    if len(pathsVids) == 0:
        pathVid = ''
    elif len(pathsVids) == 1:
        pathVid = pathsVids[0]
    else:
        raise Exception("Unexpected condition: more than one video found at '{}'".format(pathFld))
    return pathVid
def GetWPhaseFitting_2506(ts, fs, Rt):  # lm:2025-07-14; lr:2025-07-14
    if not np.min(np.diff(ts)) > 0:
        raise Exception("Invalid input: 'ts' must be strictly increasing")
    Rt_max = (np.max(ts) - np.min(ts)) / 2.1
    if Rt > Rt_max:
        return -999., -999.
    tsIn, wsIn = [[] for _ in range(2)]
    for posT, t in enumerate(ts):
        if not (np.min(ts)+Rt <= t <= np.max(ts)-Rt):  # we want the whole neighborhood within ts
            continue
        possA = np.where((ts >= t-Rt) & (ts <= t+Rt))[0]
        A = np.ones((len(possA), 2))  # IMP*: initialize with ones
        A[:, 1] = ts[possA] - ts[posT]  # sol[0] + sol[1] * t
        b = np.angle(fs[possA] * np.conj(fs[posT]))  # IMP*: real value; np.angle(real_value) = 0
        AT = np.transpose(A)
        sol = np.linalg.solve(np.dot(AT, A), np.dot(AT, b))
        tsIn.append(t); wsIn.append(np.abs(sol[1]))
    tsIn, wsIn = map(np.asarray, [tsIn, wsIn])
    w, wStd = np.mean(wsIn), np.std(wsIn)
    if w <= 0:
        return -999., -999.
    return w, wStd
def IntersectionOfTwoLines_2506(line0, line1, eps=1.e-14):  # 2000-01-01; lm:2025-06-11; lr:2025-06-30
    den = ClipWithSign(line0['lx'] * line1['ly'] - line1['lx'] * line0['ly'], eps, np.inf)  # WATCH OUT; epsilon
    xI = (line1['lt'] * line0['ly'] - line0['lt'] * line1['ly']) / den
    yI = (line0['lt'] * line1['lx'] - line1['lt'] * line0['lx']) / den
    return xI, yI
def IsFlModified_2506(pathFl, t0=None, margin=datetime.timedelta(seconds=2)):  # 2000-01-01; lm:2025-06-26; lr:2025-07-13
    if t0 is None:
        t0 = datetime.datetime.now()    
    tM = datetime.datetime.fromtimestamp(os.path.getmtime(pathFl))
    tC = datetime.datetime.fromtimestamp(os.path.getctime(pathFl))
    isFlModified = abs(tM - t0) <= margin or abs(tC - t0) <= margin
    return isFlModified
def IsFldModified_2506(pathFld, t0=None, margin=datetime.timedelta(seconds=2), recursive=False):  # 2000-01-01; lm:2025-06-26; lr:2025-07-13
    if t0 is None:
        t0 = datetime.datetime.now()    
    isFldModified = False
    if recursive:
        for root, _, fns in os.walk(pathFld):
            for fn in fns:
                pathFl = os.path.join(root, fn)
                if os.path.isfile(pathFl):
                    if IsFlModified_2506(pathFl, t0=t0, margin=margin):
                        return True
    else:
        for fn in os.listdir(pathFld):
            pathFl = os.path.join(pathFld, fn)
            if os.path.isfile(pathFl):
                if IsFlModified_2506(pathFl, t0=t0, margin=margin):
                    return True
    return isFldModified
def IsImg_2504(img):  # 2000-01-01; lm:2025-05-27; lr:2025-07-01
    isImg = True
    if not isinstance(img, np.ndarray):
        return False
    if not img.ndim >= 2:
        return False
    nr, nc = img.shape[:2]
    if not (nr > 0 and nc > 0):
        return False
    if img.dtype.kind not in ('u', 'i') or img.min() < 0 or img.max() > 255:
        return False
    return isImg
def Kappa2MuOneStep(kappa):  # 1900-01-01; lm:2025-05-27; lr:2025-07-02
    kappa = np.clip(kappa, 1.e-14, np.inf)  # WATCH OUT; epsilon
    mu = kappa / np.sqrt(np.tanh(kappa)) * (1. + kappa ** 1.09 * np.exp(-(1.56 + 1.28 * kappa + 0.219 * kappa ** 2 + 0.00371 * kappa ** 3)))  # error = 0.047%
    return mu
def LoadDMCSFromCalTxt_2502(pathCalTxt, rangeC, incHor=True, zr=0.):  # 2010-01-01; lm:2025-05-01; lr:2025-07-11
    allVar, nc, nr = ReadCalTxt_2410(pathCalTxt, rangeC)[:3]
    dMCS = AllVar2DMCS_2410(allVar, nc, nr, rangeC, incHor=incHor, zr=zr)
    return dMCS
def LoadParamsC(fontsize=24):  # lm:20205-06-24; lr:2025-06-24
    paramsC = {'font.size': fontsize, 'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize, 'legend.fontsize': fontsize, 'figure.titlesize': fontsize}
    return paramsC
def MatrixK_2410(dCaSVar, rangeC):  # 1900-01-01; lm:2025-05-01; lr:2025-06-21
    if rangeC == 'close':  # K*
        sca, sra = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sca', 'sra']]
        K = np.asarray([[1/sca, 0, dCaSVar['oc']], [0, 1/sra, dCaSVar['or']], [0, 0, 1]])
    elif rangeC == 'long':  # K
        sc, sr = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sc', 'sr']]
        K = np.asarray([[1/sc, 0, dCaSVar['oc']], [0, 1/sr, dCaSVar['or']], [0, 0, 1]])
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    if False:  # avoidable check for readability
        assert np.isclose(K[0, 2], dCaSVar['oc']) and np.isclose(K[2, 0], 0)
        assert np.isclose(K[1, 2], dCaSVar['or']) and np.isclose(K[2, 1], 0)
    return K
def MatrixR_2502(dAngVar):  # 1900-01-01; lr:2025-04-30; lr:2025-06-21
    eu, ev, ef = UnitVectors_2502(dAngVar)
    R = np.asarray([eu, ev, ef])
    if False:  # avoidable check for readability
        assert np.allclose(eu, R[0, :]) and np.allclose(ev, R[1, :]) and np.allclose(ef, R[2, :])
    return R
def MatrixRt_2410(dExtVar, rangeC):  # 1900-01-01; lm:2025-04-30; lm:2025-06-21
    if rangeC == 'close':
        xH, yH, zH = [dExtVar[item] for item in ['xc', 'yc', 'zc']]
    elif rangeC == 'long':
        xH, yH, zH = [dExtVar[item] for item in ['x0', 'y0', 'z0']]
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    R = MatrixR_2502(dExtVar)  # dAngVar < dExtVar
    t = -np.dot(R, np.asarray([xH, yH, zH]))  # the rows of R are eu, ev and ef
    if False:  # avoidable check for readability
        eu, ev, ef = UnitVectors_2502(dExtVar)
        assert np.allclose(t, np.asarray([-xH*item[0]-yH*item[1]-zH*item[2] for item in [eu, ev, ef]]))
    Rt = np.zeros((3, 4))
    Rt[:, :3], Rt[:, 3] = R, t
    if False:  # avoidable check for readability
        assert Rt.shape == (3, 4) and np.allclose(Rt[:, :3], R) and np.allclose(Rt[:, 3], t)
    return Rt
def NForRANSAC(pOutlier, pDesired, nOfPoints, eps=1.e-12):  # 1900-01-01; lm:2025-05-28; lr:2025-06-23
    num = np.log(np.clip(1 - pDesired, eps, 1-eps))
    den = np.log(np.clip(1 - (1 - pOutlier) ** nOfPoints, eps, 1-eps))
    N = int(np.ceil(num / den))
    return N
def OHorizon_2410():  # 2000-01-01; lm:2025-05-27; lr:2025-07-11
    oHorizon = 5
    return oHorizon
def OpenVideoOrFail_2504(pathVid):  # 2000-01-01; lm:2025-05-28; lr:2025-07-14
    vid = cv2.VideoCapture(pathVid)
    if not vid.isOpened():
        raise Exception("Invalid input: failed to read '{}'".format(pathVid))
    return vid
def PathImgOrImg2Img(img):  # 2000-01-01; lm:2025-05-27; lr:2025-07-01
    if isinstance(img, str):
        img = cv2.imread(img)  # WATCH OUT: can return None without raising an error
    if not IsImg_2504(img):
        raise Exception("Invalid input: invalid path or image")
    return img
def PathVid2AllFrames_2506(pathVid, pathFldFrames, stamp='millisecond', extImg='png'):  # 2010-01-01; lm:2025-06-11; lr:2025-06-30
    pathFldFramesP = Path(pathFldFrames)
    pathFldFramesP.mkdir(parents=True, exist_ok=True)
    pattern = str(pathFldFramesP / f"frame_%06d.{extImg}")
    cmd = ["ffmpeg", "-i", pathVid, "-vsync", "0", pattern]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    fnVidWE = os.path.splitext(os.path.split(pathVid)[1])[0]
    if stamp == 'millisecond':
        fpsVid = PathVid2FPS(pathVid)
    fns = sorted([item for item in os.listdir(pathFldFrames)])  # IMP*; sorted
    for posFn, fn in enumerate(fns):
        if stamp == 'millisecond':
            millisecond = int(posFn * 1000 / fpsVid)  # WATCH OUT; IMP*; int()
            fnNew = '{:}_{:}.{:}'.format(fnVidWE, str(millisecond).zfill(12), extImg)  # IMP*; nomenclature
        elif stamp == 'counter':  # counter
            fnNew = '{:}_{:}.{:}'.format(fnVidWE, str(posFn).zfill(12), extImg)  # IMP*; nomenclature
        else:
            raise Exception("Invalid input: 'stamp' must be 'millisecond' or 'counter'")
        os.rename(os.path.join(pathFldFrames, fn), os.path.join(pathFldFrames, fnNew))
    return None
def PathVid2FPS(pathVid):  # 2000-01-01; lm:2025-05-28; lr:2025-07-14
    vid = OpenVideoOrFail_2504(pathVid)
    fps = vid.get(cv2.CAP_PROP_FPS)
    vid.release()
    return fps
def PathVid2NOfFrames(pathVid):  # 2000-01-01; lm:2025-05-28; lr:2025-07-14
    vid = OpenVideoOrFail_2504(pathVid)
    nOfFrames = int(np.round(vid.get(cv2.CAP_PROP_FRAME_COUNT)))
    vid.release()
    return nOfFrames
def PathVid2NcNr(pathVid):  # 2000-01-01; lm:2025-05-28; lr:2025-07-09
    vid = OpenVideoOrFail_2504(pathVid)
    nc = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    nr = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid.release()
    return nc, nr
def Polyline2InnerHexagonalMesh_2506(polyline, d, eps_xs=1.e-9):  # 1900-01-01; lm:2025-06-12; lr:2025-07-01
    xsC, ysC = CloudOfPoints2Rectangle_2504(polyline['xs'], polyline['ys'], margin=2.1*d)[:2]
    v1, v2 = [np.asarray([xsC[pos+1] - xsC[pos], ysC[pos+1] - ysC[pos]]) for pos in [0, 1]]
    l1, l2 = [np.sqrt(np.sum(item ** 2)) for item in [v1, v2]]
    u1, u2 = v1 / l1, v2 / l2
    j1, j2 = d, 2 * d * np.cos(np.pi / 6)  # WATCH OUT; 2 * d * cos(30); jump between same-indented rows
    n1, n2 = int(l1 / j1 + 1), int(l2 / j2 + 1)
    xsM, ysM = [[] for _ in range(2)]
    for indented in [False, True]:
        x0 = xsC[0]
        y0 = ysC[0]
        if indented:
            x0 += j1 / 2 * u1[0] + j2 / 2 * u2[0]  # [0] -> x
            y0 += j1 / 2 * u1[1] + j2 / 2 * u2[1]  # [1] -> y
        for iN1, iN2 in itertools.product(range(n1), range(n2)):
            xsM.append(x0 + iN1 * j1 * u1[0] + iN2 * j2 * u2[0])
            ysM.append(y0 + iN1 * j1 * u1[1] + iN2 * j2 * u2[1])
    xsM, ysM = map(np.asarray, [xsM, ysM])
    possI = CloudOfPoints2PossInsidePolyline_2508(xsM, ysM, polyline, eps_xy=eps_xs)
    xsM, ysM = [item[possI] for item in [xsM, ysM]]
    return xsM, ysM
def Poss0AndPoss1InFind2DTransform_2504(n): # 1900-01-01; lm:2025-05-28; lr:2025-06-27
    aux = np.arange(n)  # array([], dtype=int64) if n == 0
    poss0 = 2 * aux + 0
    poss1 = 2 * aux + 1
    return poss0, poss1
def PrintDictionary_2506(theDictionary, margin=0, sB='*'):  # 2020-01-01; lm:2025-06-19; lr:2025-07-09
    if not theDictionary:
        print("{:{}}{} <empty dictionary>".format('', margin, sB))
        return None
    lMax = max(max(len(str(item)) for item in theDictionary) + 5, 30)  # WATCH OUT; epsilon
    for key in theDictionary:
        print("{:{}}{} __{:_<{}} {}".format('', margin, sB, key, lMax, theDictionary[key]))
    return None
def RANSACPlane(xs, ys, zs, errorC, pDesired=1-1.e-9, margin=0.5, max_nForRANSAC=np.inf):  # lm:2025-06-17; lm:2025-07-02
    minNOfPoints = 3
    if not (len(xs) == len(ys) == len(zs) >= minNOfPoints):
        raise Exception("Invalid input: 'xs', 'ys', and 'zs' must be arrays of the same length >= {}".format(minNOfPoints))
    A, b = np.column_stack((xs, ys, np.ones(len(xs)))), zs
    iForRANSAC, nForRANSAC, possG = 0, np.inf, []
    while iForRANSAC < nForRANSAC:
        possH = np.random.choice(range(len(xs)), size=3, replace=False)
        AH, bH = A[possH, :], b[possH]
        try:
            AHT = np.transpose(AH)
            sol = np.linalg.solve(np.dot(AHT, AH), np.dot(AHT, bH))
            errors = np.abs(np.dot(A, sol) - b)
        except Exception:
            continue
        possGH = np.where(errors <= errorC)[0]
        if len(possGH) > len(possG):
            possG = possGH
            pOutlier = 1 - len(possG) / len(xs) + margin * len(possG) / len(xs)  # margin in [0, 1), the higher the safer
            nForRANSAC = min(max_nForRANSAC, NForRANSAC(pOutlier, pDesired, minNOfPoints))
        if len(possG) == len(xs):
            break
        iForRANSAC += 1
    A, b = A[possG, :], b[possG]
    AT = np.transpose(A)
    px, py, pl = np.linalg.solve(np.dot(AT, A), np.dot(AT, b))
    return px, py, pl, possG
def REarth_2410():  # 2000-01-01; lm:2025-05-27; lr:2025-07-11
    rEarth = 6.371e+6
    return rEarth
def RMSE1D_2506(xs, xsR):  # lm:2025-06-18; lr:2025-07-13
    rmse = np.sqrt(np.mean((xs - xsR) ** 2))
    return rmse
def ReadCalTxt_2410(pathCalTxt, rangeC):  # 1900-01-01; lm:2025-05-01; lr:2025-07-14
    data = np.loadtxt(pathCalTxt, usecols=0, dtype=float, ndmin=1)
    if rangeC == 'close':
        lenAllVar = 14
    elif rangeC == 'long':
        lenAllVar = 8
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    if len(data) < lenAllVar+3:
        raise Exception("Invalid input: unable to read file at '{}'".format(pathCalTxt))
    allVar, nc, nr, errorT = data[:lenAllVar], int(np.round(data[lenAllVar])), int(np.round(data[lenAllVar+1])), data[lenAllVar+2]
    return allVar, nc, nr, errorT
def RecomputeFPS_2502(fpsGoal, fpsAvailable, round=True):  # 1900-01-01; lm:2025-06-06; lr:2025-07-11
    if not (0 < fpsGoal < fpsAvailable):
        fps = fpsAvailable
    else:
        factor = fpsAvailable / fpsGoal  # IMP*; >1
        if round:
            factor = int(np.round(factor))
        else:
            factor = int(factor)
        if True:  # avoidable check for readability
            assert isinstance(factor, int) and factor >= 1
        fps = fpsAvailable / factor  # float, but so that fpsAvailable / fps = factor = integer
    if True:  # avoidable check for readability
        assert 0 < fps <= fpsAvailable
    return fps
def TH2LOneStep(T, h, g):  # 2010-01-01; lm:2025-05-27; lr:2025-07-02
    w = 2 * np.pi / T
    kappa = w ** 2 * h / g
    mu = Kappa2MuOneStep(kappa)
    k = mu / h
    L = 2 * np.pi / k
    return L
def UDaVDa2UUaVUaAux_2410(uDas, vDas, uUas, vUas, dDtrVar, rangeC):  # 1900-01-01; lm:2025-05-01; lr:2025-07-01
    uDasN, vDasN = UUaVUa2UDaVDa_2410(uUas, vUas, dDtrVar, rangeC)[:2]  # WATCH OUT: all positions; distorted from the current undistorted
    uerrors, verrors = uDas - uDasN, vDas - vDasN  # IMP*: direction
    errors = np.hypot(uerrors, verrors)
    aux1s = uUas ** 2 + vUas ** 2
    aux1suUa = 2 * uUas
    aux1svUa = 2 * vUas
    aux2s = 1 + dDtrVar['k1a'] * aux1s + dDtrVar['k2a'] * aux1s ** 2
    aux2suUa = dDtrVar['k1a'] * aux1suUa + dDtrVar['k2a'] * 2 * aux1s * aux1suUa
    aux2svUa = dDtrVar['k1a'] * aux1svUa + dDtrVar['k2a'] * 2 * aux1s * aux1svUa
    aux3suUa = 2 * vUas
    aux3svUa = 2 * uUas
    aux4suUa = aux1suUa + 4 * uUas
    aux4svUa = aux1svUa
    aux5suUa = aux1suUa
    aux5svUa = aux1svUa + 4 * vUas
    JuUauUas = aux2s + uUas * aux2suUa + dDtrVar['p2a'] * aux4suUa + dDtrVar['p1a'] * aux3suUa
    JuUavUas = uUas * aux2svUa + dDtrVar['p2a'] * aux4svUa + dDtrVar['p1a'] * aux3svUa
    JvUauUas = vUas * aux2suUa + dDtrVar['p1a'] * aux5suUa + dDtrVar['p2a'] * aux3suUa
    JvUavUas = aux2s + vUas * aux2svUa + dDtrVar['p1a'] * aux5svUa + dDtrVar['p2a'] * aux3svUa
    dens = JuUauUas * JvUavUas - JuUavUas * JvUauUas
    dens = ClipWithSign(dens, 1.e-14, np.inf)  # WATCH OUT: epsilon
    duUas = (+JvUavUas * uerrors - JuUavUas * verrors) / dens
    dvUas = (-JvUauUas * uerrors + JuUauUas * verrors) / dens
    return duUas, dvUas, errors
def UDaVDa2UUaVUaParabolicDistortion_2410(uDas, vDas, k1a, rtrnPossG=False):  # undistort; Cardano; 1900-01-01; lm:2025-05-01; lr:2025-07-01
    xiDs = k1a * (uDas ** 2 + vDas ** 2)
    xiUs, possG = XiD2XiUCubicEquation_2410(xiDs, rtrnPossG=rtrnPossG)
    uUas, vUas = [item / (1 + xiUs) for item in [uDas, vDas]]
    return uUas, vUas, possG
def UDaVDa2UUaVUa_2410(uDas, vDas, dDtrVar, rangeC, rtrnPossG=False):  # undistort; potentially expensive; 1900-01-01; lm:2025-05-01; lm:2025-07-01
    if rangeC == 'long':
        if rtrnPossG:
            possG = np.arange(len(uDas))
        else:
            possG = np.asarray([], dtype=int)
        return uDas, vDas, possG
    elif rangeC == 'close':
        if len(uDas) == 0 or len(vDas) == 0:
            return np.full(uDas.shape, np.nan), np.full(uDas.shape, np.nan), np.asarray([], dtype=int)  # WATCH OUT
        uUas, vUas, possG = UDaVDa2UUaVUaParabolicDistortion_2410(uDas, vDas, dDtrVar['k1a'], rtrnPossG=rtrnPossG)
        if np.allclose([dDtrVar['k2a'], dDtrVar['p1a'], dDtrVar['p2a']], 0, atol=1.e-9):  # WATCH OUT: epsilon
            return uUas, vUas, possG
        errors, hasConverged, counter = 1.e+6 * np.ones(uDas.shape), False, 0
        while not hasConverged and counter <= 50:
            duUas, dvUas, errorsN = UDaVDa2UUaVUaAux_2410(uDas, vDas, uUas, vUas, dDtrVar, rangeC)
            possB = np.where(np.isnan(errorsN) | (errorsN > 4 * errors))[0]  # WATCH OUT: some points do not converge; epsilon
            if len(possB) == len(uDas):  # not hasConverged
                break
            uUas, vUas, errors, hasConverged, counter = uUas + duUas, vUas + dvUas, errorsN, np.nanmax(errorsN) < 1.e-9, counter + 1  # WATCH OUT: epsilon
            uUas[possB], vUas[possB] = np.nan, np.nan  # IMP*
        if not hasConverged:
            return np.full(uDas.shape, np.nan), np.full(uDas.shape, np.nan), np.asarray([], dtype=int)  # WATCH OUT
        if rtrnPossG:  # WATCH OUT: necessarily different than for parabolic: now it includes something like the solution xiU < -4/3; handles np.nan
            uDasR, vDasR = UUaVUa2UDaVDa_2410(uUas, vUas, dDtrVar, rangeC)[:2]  # WATCH OUT: all positions
            possG = np.where(np.hypot(uDasR - uDas, vDasR - vDas) < 1.e-9)[0]  # WATCH OUT: epsilon; this is also a check
        else:
            possG = np.asarray([], dtype=int)
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return uUas, vUas, possG
def UUaVUa2UDaVDaParabolicDistortion_2410(uUas, vUas, k1a, rtrnPossG=False):  # distort; 1900-01-01; lm:2025-05-01; lm:2025-07-01
    xiUs = k1a * (uUas ** 2 + vUas ** 2)
    uDas, vDas = [item * (1 + xiUs) for item in [uUas, vUas]]
    xiDs, possG = XiU2XiDCubicEquation_2410(xiUs, rtrnPossG=rtrnPossG)
    if False:  # avoidable check for readability
        assert np.allclose(xiDs, k1a * (uDas ** 2 + vDas ** 2))
    return uDas, vDas, possG
def UUaVUa2UDaVDa_2410(uUas, vUas, dDtrVar, rangeC, rtrnPossG=False):  # distort; 1900-01-01; lm:2025-05-01; lr:2025-07-01
    if rangeC == 'long':
        if rtrnPossG:
            possG = np.arange(len(uUas))
        else:
            possG = np.asarray([], dtype=int)
        return uUas, vUas, possG
    elif rangeC == 'close':
        if np.allclose([dDtrVar['k2a'], dDtrVar['p1a'], dDtrVar['p2a']], 0, atol=1.e-9):  # WATCH OUT: epsilon
            uDas, vDas, possG = UUaVUa2UDaVDaParabolicDistortion_2410(uUas, vUas, dDtrVar['k1a'], rtrnPossG=rtrnPossG)
            return uDas, vDas, possG
        aux1s = uUas ** 2 + vUas ** 2  # = dUas**2 = d_{U*}**2
        aux2s = 1 + dDtrVar['k1a'] * aux1s + dDtrVar['k2a'] * aux1s ** 2
        aux3s = 2 * uUas * vUas
        aux4s = aux1s + 2 * uUas ** 2
        aux5s = aux1s + 2 * vUas ** 2
        uDas = uUas * aux2s + dDtrVar['p2a'] * aux4s + dDtrVar['p1a'] * aux3s
        vDas = vUas * aux2s + dDtrVar['p1a'] * aux5s + dDtrVar['p2a'] * aux3s
        if rtrnPossG:  # WATCH OUT: necessarily different than for parabolic: now it includes something like the solution xiU < -4/3
            uUasR, vUasR = UDaVDa2UUaVUa_2410(uDas, vDas, dDtrVar, rangeC)[:2]  # WATCH OUT: all positions
            possG = np.where(np.hypot(uUasR - uUas, vUasR - vUas) < 1.e-6)[0]  # WATCH OUT: epsilon
        else:
            possG = np.asarray([], dtype=int)
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return uDas, vDas, possG
def UaVa2CR_2410(uas, vas, dCaSVar, rangeC):  # 1900-01-01; lm:2025-04-30; lr:2025-06-30
    if rangeC == 'close':
        sca, sra = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sca', 'sra']]  # WATCH OUT: epsilon
        cs = uas / sca + dCaSVar['oc']
        rs = vas / sra + dCaSVar['or']
    elif rangeC == 'long':
        sc, sr = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sc', 'sr']]  # WATCH OUT: epsilon
        cs = uas / sc + dCaSVar['oc']  # uas are actually us in this case
        rs = vas / sr + dCaSVar['or']  # vas are actually vs in this case
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return cs, rs
def UnitVectors_2502(dAngVar):  # 1900-01-01; lm:2025-04-30; lr:2025-06-23
    sph, cph = np.sin(dAngVar['ph']), np.cos(dAngVar['ph'])
    ssg, csg = np.sin(dAngVar['sg']), np.cos(dAngVar['sg'])
    sta, cta = np.sin(dAngVar['ta']), np.cos(dAngVar['ta'])
    eux = +csg * cph - ssg * sph * cta
    euy = -csg * sph - ssg * cph * cta
    euz = -ssg * sta
    eu = np.asarray([eux, euy, euz])
    evx = -ssg * cph - csg * sph * cta
    evy = +ssg * sph - csg * cph * cta
    evz = -csg * sta
    ev = np.asarray([evx, evy, evz])
    efx = +sph * sta
    efy = +cph * sta
    efz = -cta
    ef = np.asarray([efx, efy, efz])
    if False:  # avoidable check for readability
        R = np.asarray([eu, ev, ef])
        assert np.allclose(np.dot(R, np.transpose(R)), np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    return eu, ev, ef
def Vid2VidModified_2504(pathVid0, pathVid1, fps=0., round=True, scl=1., t0InSeconds=0., t1InSeconds=np.inf, overwrite=False):  # 2010-01-01; lm:2025-06-23; lr:2025-07-11
    if pathVid1 == pathVid0:
        raise Exception("Invalid input: 'pathVid0' and 'pathVid1' must be different")
    pathFld0, fnVid0 = os.path.split(pathVid0)
    pathFld1, fnVid1 = os.path.split(pathVid1)
    os.makedirs(pathFld1, exist_ok=True)
    if os.path.exists(pathVid1):
        if overwrite:
            os.remove(pathVid1)
        else:
            return None
    fnVidTMP = '000_{:}{:}'.format(''.join(random.choices(string.ascii_letters, k=20)), os.path.splitext(fnVid1)[1])
    pathVidTMP = os.path.join(pathFld0, fnVidTMP)  # IMP*; we always run in pathFld0
    if os.path.exists(pathVidTMP):  # miracle
        os.remove(pathVidTMP)
    (nc0, nr0), fps0, nOfFrames0 = PathVid2NcNr(pathVid0), PathVid2FPS(pathVid0), PathVid2NOfFrames(pathVid0)
    t0InSeconds0, t1InSeconds0 = 0, (nOfFrames0 - 1) / fps0  # IMP*
    fps1 = RecomputeFPS_2502(fps, fps0, round=round)
    if np.isclose(fps0, fps1):
        shutil.copy2(pathVid0, pathVidTMP)  # IMP*
    else:
        cmd = ['ffmpeg', '-i', fnVid0, '-filter:v', 'fps=fps={:.8f}'.format(fps1), fnVidTMP]
        subprocess.run(cmd, cwd=pathFld0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if not os.path.exists(pathVidTMP):
        raise Exception("Unexpected condition: unable to change fps of '{}'".format(pathVid0))
    t0InSeconds1, t1InSeconds1 = max(t0InSeconds, t0InSeconds0), min(t1InSeconds, t1InSeconds0)
    if not (np.isclose(t0InSeconds0, t0InSeconds1) and np.isclose(t1InSeconds0, t1InSeconds1)):
        t0 = '{:02}:{:02}:{:02}'.format(*map(int, [t0InSeconds1 // 3600, (t0InSeconds1 % 3600) // 60, t0InSeconds1 % 60]))
        t1 = '{:02}:{:02}:{:02}'.format(*map(int, [t1InSeconds1 // 3600, (t1InSeconds1 % 3600) // 60, t1InSeconds1 % 60]))
        fnAux = '{:}_aux{:}'.format(os.path.splitext(fnVidTMP)[0], os.path.splitext(fnVidTMP)[1])
        cmd = ['ffmpeg', '-ss', t0, '-i', fnVidTMP, '-to', t1, fnAux]
        subprocess.run(cmd, cwd=pathFld0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        shutil.move(os.path.join(pathFld0, fnAux), pathVidTMP)
    nc1, nr1 = int(np.round(scl * nc0)), int(np.round(scl * nr0))
    if not (np.isclose(nc0, nc1) and np.isclose(nr0, nr1)):
        fnAux = '{:}_aux{:}'.format(os.path.splitext(fnVidTMP)[0], os.path.splitext(fnVidTMP)[1])
        cmd = ['ffmpeg', '-i', fnVidTMP, '-vf', f'scale={nc1}:{nr1}', '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', fnAux]
        subprocess.run(cmd, cwd=pathFld0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        shutil.move(os.path.join(pathFld0, fnAux), pathVidTMP)
    shutil.move(pathVidTMP, pathVid1)
    return None
def WGH2KOneStep(w, g, h):  # lm:2022-07-09; lr:2025-06-26
    if np.any(np.asarray(h) <= 0):
        raise Exception("Invalid input: 'h' must be positive")
    kappa = w ** 2 * h / g
    mu = Kappa2MuOneStep(kappa)
    k = mu / h
    return k
def X2DMD_2506(X, rank, dt=1):  # lm:2025-06-16; lm:2025-07-03
    X0, X1 = X[:, :-1], X[:, 1:]
    U, S, Vstar = np.linalg.svd(X0, full_matrices=False)
    Ustar, V = [np.conj(np.transpose(item)) for item in [U, Vstar]]
    U, S, V = U[:, :rank], S[:rank], V[:, :rank]
    Ustar, Vstar = [np.conj(np.transpose(item)) for item in [U, V]]
    Sinv = np.diag([1/s if s > 1e-14 else 0 for s in S])  # it was Sinv = np.diag(1 / S)
    B = np.dot(Ustar, np.dot(X1, np.dot(V, Sinv)))
    Lambda, W = np.linalg.eig(B)
    Phi = np.dot(X1, np.dot(V, np.dot(Sinv, W)))
    Ts = 2 * np.pi * dt / np.imag(np.log(Lambda))
    phases = [np.angle(Phi[:, pos]) for pos in range(Phi.shape[1])]
    amplitudes = [np.abs(Phi[:, pos]) for pos in range(Phi.shape[1])]
    if True:  # avoidable check for readability
        assert len(Ts) == len(phases) == len(amplitudes) == rank
    return Ts, phases, amplitudes
def X2EOF_2502(X):  # 2000-01-01; lm:2020-02-16; lr:2025-06-25
    (nx, nt), nq = X.shape, np.min(X.shape)
    XMean, XHat = X2XMean_2502(X), X2XHat_2502(X)
    U, S, VH = np.linalg.svd(XHat, full_matrices=False)  # (nx x nq)-ndarray, nq array, nq x nt ndarray
    SVH = np.dot(np.diag(S), VH)  # nq x nt ndarray
    expVariances = np.real(np.diag(np.dot(SVH, np.conj(np.transpose(SVH)))))  # nq x nq ndarray
    if True:  # avoidable check for readability
        assert np.allclose(expVariances, S**2)
        assert U.shape == (nx, nq) and S.shape == (nq, ) and VH.shape == (nq, nt) 
        assert np.allclose(XHat, np.dot(U, SVH))
        assert np.allclose(np.dot(np.conj(U.transpose()), U), np.eye(nq)) 
        assert np.allclose(np.dot(VH, np.conj(VH.transpose())), np.eye(nq))
    spatialMean = XMean
    spatialEOFs = [U[:, item] for item in range(nq)]
    temporalEOFs = [SVH[item, :] for item in range(nq)]
    expVariances = expVariances / np.sum(expVariances)
    cumVariances = np.cumsum(expVariances)
    return nx, nt, nq, U, SVH, spatialMean, spatialEOFs, temporalEOFs, expVariances, cumVariances
def X2LSByCandes_2506(X, max_iter=10000, tol=None, mu=None, lmbda=None):  # 2000-01-01; lm:2025-07-03; lr:2025-07-13
    m, n = X.shape
    normX = np.linalg.norm(X, 'fro')
    if mu is None:
        mu = (m * n) / (4 * np.sum(np.abs(X) + 1e-8))  # WATCH OUT: epsilon
    mu_inv = 1 / mu
    if lmbda is None:
        lmbda = 1 / np.sqrt(max(m, n))
    if tol is None:
        tol = 1e-7 * normX  # WATCH OUT: epsilon
    L, S, Y = [np.zeros_like(X, dtype=X.dtype) for _ in range(3)]
    for _ in range(max_iter):
        U, s, Vh = np.linalg.svd(X - S + mu_inv * Y, full_matrices=False)
        s_thresh = np.maximum(np.abs(s) - mu_inv, 0)
        L = (U @ np.diag(s_thresh)) @ Vh
        D = X - L + mu_inv * Y
        S = np.sign(D) * np.maximum(np.abs(D) - mu_inv * lmbda, 0)
        Z = X - L - S
        Y += mu * Z
        err = np.linalg.norm(Z, 'fro')
        if err < tol:
            break
    return L, S
def X2XHat_2502(X):  # 1900-01-01; lm:2025-04-11; lr:2025-06-25
    XMean = np.mean(X, axis=1, keepdims=True)
    XHat = X - XMean
    return XHat
def X2XMean_2502(X):  # 1900-01-01; lm:2025-04-11; lm:2025-06-25
    if X.ndim != 2:
        raise Exception("Invalid input: 'X' must be a 2D float ndarray")
    XMean = np.mean(X, axis=1)
    return XMean    
def XYZ2CDRD_2410(xs, ys, zs, dMCS, rtrnPossG=False, margin=0):  # explicit if not rtrnPossG; 2010-01-01; lm:2025-05-05; lr:2025-06-23
    Px, rangeC, dAllVar, ef, nc, nr = [dMCS[item] for item in ['Px', 'rangeC', 'dAllVar', 'ef', 'nc', 'nr']]
    cUs, rUs, possG = XYZ2CURU_2410(xs, ys, zs, Px, rangeC, rtrnPossG=rtrnPossG, dCamVar=dAllVar, ef=ef, nc=nc, nr=nr)
    cDs, rDs, possGH = CURU2CDRD_2410(cUs, rUs, dAllVar, dAllVar, rangeC, rtrnPossG=rtrnPossG, nc=nc, nr=nr, margin=margin)
    possG = np.intersect1d(possG, possGH, assume_unique=True)
    if rtrnPossG and len(possG) > 0:
        xsG, ysG, zsG, cDsG, rDsG = [item[possG] for item in [xs, ys, zs, cDs, rDs]]
        xsGR, ysGR = CDRDZ2XY_2410(cDsG, rDsG, zsG, dMCS)[:2]  # WATCH OUT: potentially expensive
        possGInPossG = np.where(np.hypot(xsG - xsGR, ysG - ysGR) < 1.e-6)[0]  # WATCH OUT: epsilon; could be 1.e-3 also
        possG = possG[possGInPossG]
    return cDs, rDs, possG
def XYZ2CURU_2410(xs, ys, zs, Px, rangeC, rtrnPossG=False, dCamVar=None, ef=None, nc=None, nr=None):  # 2010-01-01; lm:2025-05-28; lr:2025-07-05
    if rangeC == 'close':
        dens = Px[8] * xs + Px[9] * ys + Px[10] * zs + 1
        dens = ClipWithSign(dens, 1.e-14, np.inf)  # WATCH OUT: epsilon
        cUs = (Px[0] * xs + Px[1] * ys + Px[2] * zs + Px[3]) / dens
        rUs = (Px[4] * xs + Px[5] * ys + Px[6] * zs + Px[7]) / dens
    elif rangeC == 'long':
        cUs = Px[0] * xs + Px[1] * ys + Px[2] * zs + Px[3]
        rUs = Px[4] * xs + Px[5] * ys + Px[6] * zs + Px[7]
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    if rtrnPossG:
        possG = XYZ2PossRightSideOfCamera_2410(xs, ys, zs, rangeC, dCamVar=dCamVar, ef=ef)
        if len(possG) > 0 and nc is not None and nr is not None:
            cUsG, rUsG = [item[possG] for item in [cUs, rUs]]
            possGInPossG = CR2PossWithinImage_2502(cUsG, rUsG, nc, nr, margin=-max(nc, nr), case='')  # WATCH OUT: undistorted, large negative margin to relax
            possG = possG[possGInPossG]
    else:
        possG = np.asarray([], dtype=int)
    return cUs, rUs, possG
def XYZ2PossRightSideOfCamera_2410(xs, ys, zs, rangeC, dCamVar=None, ef=None):  # 2000-01-01; lr:2025-05-28; lr:2025-06-22
    if rangeC == 'close':
        xas, yas, zas = xs - dCamVar['xc'], ys - dCamVar['yc'], zs - dCamVar['zc']
        possRightSideOfCamera = np.where(xas * ef[0] + yas * ef[1] + zas * ef[2] > 0)[0]
    elif rangeC == 'long':
        possRightSideOfCamera = np.arange(len(xs))
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return possRightSideOfCamera
def XiD2XiUCubicEquation_2410(xiDs, rtrnPossG=False):  # undistort; Cardano; 1900-01-01; lr:2025-05-28; lr:2025-06-23
    p, qs = -1 / 3, -(xiDs + 2 / 27)
    Deltas = (xiDs + 4 / 27) * xiDs  # Deltas <= 0 -> -4/27 <= xiD <= 0 -> several solutions
    possN = np.where(Deltas <= 0)[0]  # possN -> several solutions
    possP = np.asarray([item for item in np.arange(len(xiDs)) if item not in possN], dtype=int)  # possP -> unique solution
    auxsN = (qs[possN] + 1j * np.sqrt(np.abs(Deltas[possN]))) / 2
    auxsP = (qs[possP] + np.sqrt(Deltas[possP])) / 2
    ns = np.zeros(xiDs.shape) + 1j * np.zeros(xiDs.shape)
    ns[possN] = np.abs(auxsN) ** (1 / 3) * np.exp(1j * (np.abs(np.angle(auxsN)) + 2 * np.pi * 1) / 3)  # + 2 * pi * j for j = 0, *1*, 2
    ns[possP] = np.sign(auxsP) * (np.abs(auxsP) ** (1 / 3))
    xiUs = np.real(p / (3 * ns) - ns - 2 / 3)  # WATCH OUT
    if rtrnPossG:
        possG = np.where(xiDs >= -4 / 27)[0]  # works also if len(xiDs) = 0
    else:
        possG = np.asarray([], dtype=int)
    return xiUs, possG
def XiU2XiDCubicEquation_2410(xiUs, rtrnPossG=False):  # distort; 1900-01-01; lr:2025-05-28; lr:2025-07-01
    xiDs = xiUs ** 3 + 2 * xiUs ** 2 + xiUs
    if rtrnPossG:
        possG = np.where(xiUs >= -1 / 3)[0]  # works also if len(xiUs) = 0
    else:
        possG = np.asarray([], dtype=int)
    return xiDs, possG
