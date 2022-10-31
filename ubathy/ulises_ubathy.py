#
# Fri Oct 28 15:42:08 2022, extract from Ulises by Gonzalo Simarro and Daniel Calvete
#
import copy
import cv2
import datetime
import itertools
import numpy as np
import os
#
def AllVariables2MainSet(allVariables, nc, nr, options={}): # 202109141500 (last read 2022-07-06)
    ''' comments:
    .- input allVariables is a 14-float-ndarray (xc, yc, zc, ph, sg, ta, k1a, k2a, p1a, p2a, sca, sra, oc, or)
    .- input nc and nr are integers or floats
    .- output mainSet is a dictionary
    '''
    keys, defaultValues = ['orderOfTheHorizonPolynomial', 'radiusOfEarth', 'z0'], [5, 6.371e+6, 0.]
    options = CompleteADictionary(options, keys, defaultValues)
    allVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # WATCH OUT order matters
    mainSet = {'nc':nc, 'nr':nr, 'orderOfTheHorizonPolynomial':options['orderOfTheHorizonPolynomial'], 'radiusOfEarth':options['radiusOfEarth'], 'z0':options['z0']}
    allVariablesDictionary = Array2Dictionary(allVariablesKeys, allVariables)
    allVariablesDictionary['sca'] = ClipWithSign(allVariablesDictionary['sca'], 1.e-8, 1.e+8)
    allVariablesDictionary['sra'] = ClipWithSign(allVariablesDictionary['sra'], 1.e-8, 1.e+8)
    allVariables = Dictionary2Array(allVariablesKeys, allVariablesDictionary)
    mainSet['allVariablesDictionary'] = allVariablesDictionary
    mainSet.update(allVariablesDictionary) # IMP* (absorb in mainSet all the keys which are in allVariablesDictionary)
    mainSet['allVariables'] = allVariables
    mainSet['pc'] = np.asarray([mainSet['xc'], mainSet['yc'], mainSet['zc']])
    R = EulerianAngles2R(mainSet['ph'], mainSet['sg'], mainSet['ta'])
    eu, ev, ef = R2UnitVectors(R)
    mainSet['R'] = R
    mainSet['eu'], (mainSet['eux'], mainSet['euy'], mainSet['euz']) = eu, eu
    mainSet['ev'], (mainSet['evx'], mainSet['evy'], mainSet['evz']) = ev, ev
    mainSet['ef'], (mainSet['efx'], mainSet['efy'], mainSet['efz']) = ef, ef
    Pa = MainSet2Pa(mainSet)
    mainSet['Pa'], mainSet['Pa11'] = Pa, Pa2Pa11(Pa)
    horizonLine = MainSet2HorizonLine(mainSet)
    mainSet['horizonLine'] = horizonLine
    mainSet.update(horizonLine) # IMP* (absorb in mainSet all the keys which are in horizonLine)
    return mainSet
def ApplyAffineA01(A01, xs0, ys0): # 202111241134 (last read 2022-07-11)
    ''' comments:
    .- input A01 is a 6-float-ndarray (allows to transform from 0 to 1)
    .- input xs0 and ys0 are float-ndarrays of the same length
    .- output xs1 and ys1 are float-ndarrays of the same length of xs0 and ys0
    '''
    xs1 = A01[0] * xs0 + A01[1] * ys0 + A01[2]
    ys1 = A01[3] * xs0 + A01[4] * ys0 + A01[5]
    return xs1, ys1
def AreImgMarginsOK(nc, nr, imgMargins): # 202109101200 (last read 2022-07-06)
    ''' comments:
    .- input nc and nr are integers
    .- input imgMargins is a dictionary
    .- output areImgMarginsOK is a boolean
    '''
    imgMargins = CompleteImgMargins(imgMargins)
    condC = min([imgMargins['c0'], imgMargins['c1'], nc-1-(imgMargins['c0']+imgMargins['c1'])]) >= 0
    condR = min([imgMargins['r0'], imgMargins['r1'], nr-1-(imgMargins['r0']+imgMargins['r1'])]) >= 0
    areImgMarginsOK = condC and condR
    return areImgMarginsOK
def Array2Dictionary(keys, theArray): # 202206291320 (last read 2022-06-29)
    ''' comments:
    .- input keys is a string-list
    .- input theArray is a list or ndarray of the same length of keys
    .- output theDictionary is a dictionary
    '''
    theDictionary = {keys[posKey]:theArray[posKey] for posKey in range(len(keys))}
    return theDictionary
def CDRD2CURU(mainSet, cDs, rDs): # 202109101200 (last read 2022-07-06) potentially expensive
    ''' comments:
    .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc' and 'or')
    .- input cDs and rDs are float-ndarrays of the same length
    .- output cUs and rUs are float-ndarrays of the same length of cDs and rDs or Nones (if it does not succeed)
    '''
    uDas, vDas = CR2UaVa(mainSet, cDs, rDs)
    uUas, vUas = UDaVDa2UUaVUa(mainSet, uDas, vDas) # potentially expensive
    if uUas is None or vUas is None:
        cUs, rUs = None, None
    else:
        cUs, rUs = UaVa2CR(mainSet, uUas, vUas)
    return cUs, rUs
def CDRDZ2XY(mainSet, cDs, rDs, zs, options={}): # 202109231442 (last read 2022-07-12) potentially expensive
    ''' comments:
    .- input mainSet is a dictionary (including at least 'nc' and 'nr')
    .- input cDs, rDs and zs are float-ndarrays of the same length (zs can be just a float)
    .- output xs and ys are float-ndarrays of the same length or Nones (if it does not succeed)
    .- output possGood is a integer-list or None (if it does not succeed or not options['returnGoodPositions'])
    '''
    keys, defaultValues = ['returnGoodPositions', 'imgMargins'], [False, {'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}]
    options = CompleteADictionary(options, keys, defaultValues)
    cUs, rUs = CDRD2CURU(mainSet, cDs, rDs) # potentially expensive
    if cUs is None or rUs is None:
        return None, None, None
    uUas, vUas = CR2UaVa(mainSet, cUs, rUs)
    if isinstance(zs, np.ndarray): # float-ndarray
        planes = {'pxs':np.zeros(zs.shape), 'pys':np.zeros(zs.shape), 'pzs':np.ones(zs.shape), 'pts':-zs}
    else: # float
        planes = {'pxs':0., 'pys':0., 'pzs':1., 'pts':-zs}
    xs, ys, zsR, possGood = UUaVUa2XYZ(mainSet, planes, uUas, vUas, options={'returnPositionsRightSideOfCamera':options['returnGoodPositions']})
    if isinstance(zs, np.ndarray): # float-ndarray
        assert np.allclose(zsR, zs)
    else: # float
        assert np.allclose(zsR, zs*np.ones(len(xs)))
    if options['returnGoodPositions']:
        if len(possGood) > 0:
            nc, nr, cDsGood, rDsGood = mainSet['nc'], mainSet['nr'], cDs[possGood], rDs[possGood]
            possGoodInGood = CR2PositionsWithinImage(nc, nr, cDsGood, rDsGood, options={'imgMargins':options['imgMargins']})
            possGood = [possGood[item] for item in possGoodInGood]
    else: # possGood is None from UUaVUa2XYZ above
        assert possGood is None
    return xs, ys, possGood
def CR2CRInteger(cs, rs): # 202109131000 (last read 2022-07-06)
    ''' comments:
    .- input cs and rs are integer- or float-ndarrays
    .- output cs and rs are integer-ndarrays
    '''
    cs = np.round(cs).astype(int)
    rs = np.round(rs).astype(int)
    return cs, rs
def CR2CRIntegerWithinImage(nc, nr, cs, rs, options={}): # 202109141700 (last read 2022-07-12)
    ''' comments:
    .- input nc and nr are integers or floats
    .- input cs and rs are float-ndarrays
    .- output csIW and rsIW are integer-ndarrays
    '''
    keys, defaultValues = ['imgMargins'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}]
    options = CompleteADictionary(options, keys, defaultValues)
    possWithin = CR2PositionsWithinImage(nc, nr, cs, rs, options={'imgMargins':options['imgMargins'], 'rounding':True}) # IMP*
    csIW, rsIW = cs[possWithin].astype(int), rs[possWithin].astype(int)
    return csIW, rsIW
def CR2PositionsWithinImage(nc, nr, cs, rs, options={}): # 202109131400 (last read 2022-07-06)
    ''' comments:
    .- input nc and nr are integers
    .- input cs and rs are integer- or float-ndarrays
    .- output possWithin is an integer-list
    '''
    keys, defaultValues = ['imgMargins', 'rounding'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
    options = CompleteADictionary(options, keys, defaultValues)
    imgMargins = CompleteImgMargins(options['imgMargins'])
    assert AreImgMarginsOK(nc, nr, imgMargins)
    if options['rounding']:
        cs, rs = CR2CRInteger(cs, rs)
    cMin, cMax = imgMargins['c0'], nc-1-imgMargins['c1'] # recall that img[:, nc-1, :] is OK, but not img[:, nc, :]
    rMin, rMax = imgMargins['r0'], nr-1-imgMargins['r1'] # recall that img[nr-1, :, :] is OK, but not img[nr, :, :]
    possWithin = np.where((cs >= cMin) & (cs <= cMax) & (rs >= rMin) & (rs <= rMax))[0]
    return possWithin
def CR2UaVa(mainSet, cs, rs): # 202109101200 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'sca', 'sra', 'oc' and 'or')
        .- mainSet['sca'] and mainSet['sra'] are non-zero, but allowed to be negative
    .- input cs and rs are floats or float-ndarrays of the same length
    .- output uas and vas are floats or float-ndarrays of the same length of cs and rs
    '''
    uas = (cs - mainSet['oc']) * mainSet['sca']
    vas = (rs - mainSet['or']) * mainSet['sra']
    return uas, vas
def CURU2CDRD(mainSet, cUs, rUs): # 202109101200 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary
    .- input cUs and rUs are floats or float-ndarrays of the same length
    .- output cDs and rDs are floats or float-ndarrays of the same length of cUs and rUs
    '''
    uUas, vUas = CR2UaVa(mainSet, cUs, rUs)
    uDas, vDas = UUaVUa2UDaVDa(mainSet, uUas, vUas)
    cDs, rDs = UaVa2CR(mainSet, uDas, vDas)
    return cDs, rDs
def CUh2RUh(horizonLine, cUhs): # 202109101200 (last read 2022-07-06)
    ''' comments:
    .- input horizonLine is a dictionary (including at least 'ccUh1', 'crUh1' and 'ccUh0')
        .- the horizonLine is 'ccUh1' * cUhs + 'crUh1' * rUhs + 'ccUh0' = 0, i.e., rUhs = - ('ccUh1' * cUhs + 'ccUh0') / 'crUh1'
    .- input cUhs is a float-ndarray
    .- output rUhs is a float-ndarray
    '''
    crUh1 = ClipWithSign(horizonLine['crUh1'], 1.e-8, 1.e+8)
    rUhs = - (horizonLine['ccUh1'] * cUhs + horizonLine['ccUh0']) / crUh1
    return rUhs
def ClipWithSign(xs, x0, x1): # 202109101200 (last read 2022-07-06)
    ''' comments:
    .- input xs is a float of a float-ndarray
    .- input x0 and x1 are floats so that 0 <= x0 <= x1
    .- output xs is a float of a float-ndarray
        .- output xs is in [-x1, -x0] U [x0, x1] and retains the signs of input xs
    '''
    assert x1 >= x0 >= 0.
    signs = np.sign(xs)
    if isinstance(signs, np.ndarray): # ndarray
        signs[signs == 0] = 1
    elif signs == 0: # float and 0
        signs = 1
    xs = signs * np.clip(np.abs(xs), x0, x1)
    return xs
def CloudOfPoints2InnerPositionsForPolyline(xs, ys, polyline, options={}): # 202207021000 (last read 2022-07-02, checked graphically with auxiliar code)
    ''' comments:
    .- input xs and ys float-ndarrays
    .- input polyline is a dictionary (including at least 'xs' and 'ys')
        .- input polyline can be open or closed, and its orientation is irrelevant
    .- output posIns is a integer-list os the positions within the polyline
        .- this function does not call segment-related functions: it is written careful and independently
    '''
    keys, defaultValues = ['epsilon'], [1.e-11]
    options = CompleteADictionary(options, keys, defaultValues)
    eps = options['epsilon']
    xsP, ysP = polyline['xs'], polyline['ys']
    if DistanceFromAPointToAPoint(xsP[0], ysP[0], xsP[-1], ysP[-1]) < eps:
        xsP, ysP = [item[0:-1] for item in [xsP, ysP]]
    xsP, ysP, xsM, ysM = xsP - np.mean(xsP), ysP - np.mean(ysP), xs - np.mean(xsP), ys - np.mean(ysP)
    possIn0 = [item for item in range(len(xsM)) if np.min(np.sqrt((xsM[item]-xsP)**2+(ysM[item]-ysP)**2)) < 1.e+2 * eps] # IMP* 1.e+2
    xsC, ysC, possMR0 = np.concatenate((xsP, xsM)), np.concatenate((ysP, ysM)), [item for item in range(len(xsM)) if item not in possIn0]
    while True:
        xO, yO = [np.max(item) + (0.5 + np.random.random()) * (1.0 + np.std(item)) for item in [xsC, ysC]]
        anglesP, anglesMR0 = np.angle((xO - xsP) + 1j * (yO - ysP)), np.angle((xO - xsM[possMR0]) + 1j * (yO - ysM[possMR0]))
        anglesP, anglesMR0 = np.meshgrid(anglesP, anglesMR0)
        anglesP, anglesMR0 = [np.reshape(item, -1) for item in [anglesP, anglesMR0]]
        if np.min(np.abs(anglesP - anglesMR0)) > 1.e+1 * eps: # IMP*
            break
    mx1, my1, mx2, my2, mx3, my3, mx4, my4 = [np.zeros((len(xsM), len(xsP))) for item in range(8)] # row: point of the mesh, column: segment of the boundary polyline
    for pos0B in range(len(xsP)): # run through the points of the boundary polyline
        pos1B = (pos0B + 1) % len(xsP)
        mx1[:, pos0B], my1[:, pos0B] = xsP[pos0B] * np.ones(len(xsM)), ysP[pos0B] * np.ones(len(xsM)) # point0 of the segment of the boundary
        mx2[:, pos0B], my2[:, pos0B] = xsP[pos1B] * np.ones(len(xsM)), ysP[pos1B] * np.ones(len(xsM)) # point1 of the segment of the boundary
    for posM in range(len(xsM)): # run through the points of the mesh
        mx3[posM, :], my3[posM, :] = xsM[posM] * np.ones(len(xsP)), ysM[posM] * np.ones(len(xsP)) # point of the mesh
        mx4[posM, :], my4[posM, :] = xO * np.ones(len(xsP)), yO * np.ones(len(xsP)) # outer point
    mA11, mA12, mA21, mA22, mb1, mb2 = my1 - my2, mx2 - mx1, my3 - my4, mx4 - mx3, mx2 * my1 - mx1 * my2, mx4 * my3 - mx3 * my4
    mDets = mA11 * mA22 - mA12 * mA21
    mDets[np.abs(mDets) < eps] = np.NaN
    mxI = (mb1 * mA22 - mb2 * mA12) / mDets
    myI = (mb2 * mA11 - mb1 * mA21) / mDets
    mxLim0, myLim0 = np.maximum(np.minimum(mx1, mx2), np.minimum(mx3, mx4)), np.maximum(np.minimum(my1, my2), np.minimum(my3, my4))
    mxLim1, myLim1 = np.minimum(np.maximum(mx1, mx2), np.maximum(mx3, mx4)), np.minimum(np.maximum(my1, my2), np.maximum(my3, my4))
    dsI = np.sqrt((mx3 - mxI) ** 2 + (my3 - myI) ** 2)
    auxs = np.sum((mxI >= mxLim0-eps) & (mxI <= mxLim1+eps) & (myI >= myLim0-eps) & (myI <= myLim1+eps) & (dsI < eps), axis=1)
    possIn1 = list(np.where(auxs > 0)[0])
    auxs = np.sum((mxI >= mxLim0-eps) & (mxI <= mxLim1+eps) & (myI >= myLim0-eps) & (myI <= myLim1+eps), axis=1)
    possIn2 = list(np.where(auxs % 2 == 1)[0])
    possIn = list(set(possIn0 + possIn1 + possIn2))
    return possIn
def CloudOfPoints2Rectangle(xs, ys, options={}): # 202110281047 (last read 2022-07-02, checked graphically with auxiliar code)
    ''' comments:
    .- input xs and ys are float-ndarrays of the same length
    .- output xcs and ycs are 4-float-ndarrays (4 corners, oriented clockwise, the first corner is the closest to the first point of the cloud)
    .- output area is a float
    '''
    keys, defaultValues = ['margin'], [0.]
    options = CompleteADictionary(options, keys, defaultValues)
    xcs, ycs, area = None, None, 1.e+11
    for angleH in np.linspace(0, np.pi / 2., 1000):
        xcsH, ycsH, areaH = CloudOfPoints2RectangleAnalysisForAnAngle(angleH, xs, ys, options={'margin':options['margin']})
        if areaH < area:
            xcs, ycs, area = [copy.deepcopy(item) for item in [xcsH, ycsH, areaH]]
    pos0 = np.argmin(np.sqrt((xcs - xs[0]) ** 2 + (ycs - ys[0]) ** 2)) # the first corner is the closest to the first point (they are oriented clockwise)
    xcs = np.asarray([xcs[pos0], xcs[(pos0+1)%4], xcs[(pos0+2)%4], xcs[(pos0+3)%4]])
    ycs = np.asarray([ycs[pos0], ycs[(pos0+1)%4], ycs[(pos0+2)%4], ycs[(pos0+3)%4]])
    return xcs, ycs, area
def CloudOfPoints2RectangleAnalysisForAnAngle(angle, xs, ys, options={}): # 202110280959 (last read 2022-07-02, checked graphically with auxiliar code)
    ''' comments:
    .- input angle is a float (0 = East; pi/2 = North)
    .- input xs and ys are float-ndarrays of the same length
    .- output xcs and ycs are 4-float-ndarrays (4 corners, oriented clockwise)
    .- output area is a float
    '''
    keys, defaultValues = ['margin'], [0.]
    options = CompleteADictionary(options, keys, defaultValues)
    lDs = - np.sin(angle) * xs + np.cos(angle) * ys # signed-distances to D-line dir = (+cos, +sin) through origin (0, 0)
    lPs = + np.cos(angle) * xs + np.sin(angle) * ys # signed-distances to P-line dir = (+sin, -cos) through origin (0, 0)
    area = (np.max(lDs) - np.min(lDs) + 2 * options['margin']) * (np.max(lPs) - np.min(lPs) + 2 * options['margin'])
    lD0 = {'lx':-np.sin(angle), 'ly':+np.cos(angle), 'lt':-(np.min(lDs)-options['margin'])}
    lD1 = {'lx':-np.sin(angle), 'ly':+np.cos(angle), 'lt':-(np.max(lDs)+options['margin'])}
    lP0 = {'lx':+np.cos(angle), 'ly':+np.sin(angle), 'lt':-(np.min(lPs)-options['margin'])}
    lP1 = {'lx':+np.cos(angle), 'ly':+np.sin(angle), 'lt':-(np.max(lPs)+options['margin'])}
    xcs, ycs = [np.zeros(4) for item in range(2)]
    xcs[0], ycs[0] = IntersectionOfTwoLines(lD0, lP0, options={})[0:2] 
    xcs[1], ycs[1] = IntersectionOfTwoLines(lP0, lD1, options={})[0:2]
    xcs[2], ycs[2] = IntersectionOfTwoLines(lD1, lP1, options={})[0:2]
    xcs[3], ycs[3] = IntersectionOfTwoLines(lP1, lD0, options={})[0:2]
    return xcs, ycs, area
def CompleteADictionary(theDictionary, keys, defaultValues): # 202109101200 (last read 2022-06-29)
    ''' comments:
    .- input theDictionary is a dictionary
    .- input keys is a string-list
    .- input defaultValues is a list of the same length of keys or a single value
    .- output theDictionary is a dictionary that includes keys and defaultValues for the keys not in input theDictionary
    '''
    if set(keys) <= set(theDictionary.keys()):
        pass
    else:
        if isinstance(defaultValues, (list)): # defaultValues is a list
            assert len(keys) == len(defaultValues)
            for posKey, key in enumerate(keys):
                if key not in theDictionary.keys(): # only assigns if there is no key
                    theDictionary[key] = defaultValues[posKey]
        else: # defaultValues is a single value
            for key in keys:
                if key not in theDictionary.keys(): # only assigns if there is no key
                    theDictionary[key] = defaultValues
    return theDictionary
def CompleteImgMargins(imgMargins): # 202109101200 (last read 2022-07-05)
    ''' comments:
    .- input imgMargins is a dictionary or None
        .- if imgMargins['isComplete'], then it does nothing
        .- if imgMargins is None, then it is initialized to {'c':0, 'r':0}
        .- if imgMargins includes 'c', then generates 'c0' and 'c1' (if not included); otherwise, 'c0' and 'c1' must already be included
        .- if imgMargins includes 'r', then generates 'r0' and 'r1' (if not included); otherwise, 'r0' and 'r1' must already be included
        .- imgMargins['c*'] and imgMargins['r*'] are integers
    .- output imgMargins is a dictionary (including at least 'c0', 'c1', 'r0' and 'r1' and 'isComplete'; not necessarily 'c' and 'r')
        .- imgMargins['c*'] and imgMargins['r*'] are integers
    '''
    if imgMargins is not None and 'isComplete' in imgMargins.keys() and imgMargins['isComplete']:
        return imgMargins
    if imgMargins is None:
        imgMargins = {'c':0, 'r':0}
    for letter in ['c', 'r']:
        try:
            assert int(imgMargins[letter]) == imgMargins[letter]
        except:
            assert all([int(imgMargins[letter+number]) == imgMargins[letter+number] for number in ['0', '1']])
            continue # go to the next letter
        for number in ['0', '1']:
            try:
                assert int(imgMargins[letter+number]) == imgMargins[letter+number]
            except:
                imgMargins[letter+number] = imgMargins[letter]
    imgMargins['isComplete'] = True
    return imgMargins
def Date2Datenum(date): # 202109131100
    ''' comments:
    .- input date is a 17-integer-string
    .- output datenum is a float (the unit is one day)
    '''
    datenum = Datetime2Datenum(Date2Datetime(date))
    return datenum
def Date2Datetime(date): # 202109131100 (last read 2022-07-09)
    ''' comments:
    .- input date is a 17-integer-string
    .- output theDatetime is a datetime.datetime
    '''
    theDatetime = datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), int(date[8:10]), int(date[10:12]), int(date[12:14]), int(date[14:17]) * 1000)
    return theDatetime
def Datetime2Datenum(theDatetime): # 202109131100 (last read 2022-07-09)
    ''' comment:
    .- input theDatetime is a datetime.datetime
    .- output datenum is a float (the unit is one day)
    '''
    datenum = 366. + datetime.date.toordinal(datetime.date(theDatetime.year, theDatetime.month, theDatetime.day)) + (theDatetime.hour + (theDatetime.minute + (theDatetime.second + theDatetime.microsecond / 1.e+6) / 60.) / 60.) / 24.
    return datenum
def Dictionary2Array(keys, theDictionary): # 202206291320 (last read 2022-06-29)
    ''' comments:
    .- input keys is a string-list
    .- input theDictionary is a dictionary
    .- output theArray is a ndarray
    '''
    theArray = np.asarray([theDictionary[key] for key in keys])
    return theArray
def DisplayCRInImage(img, cs, rs, options={}): # 202109141700 (last read 2022-07-12)
    ''' comments:
    .- input img is a cv2-image
    .- input cs and rs are integer- or float-ndarrays of the same length (not necessarily within the image)
    .- output img is a cv2-image
    '''
    keys, defaultValues = ['colors', 'imgMargins', 'size'], [[[0, 0, 0]], {'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, None]
    options = CompleteADictionary(options, keys, defaultValues)
    csIW, rsIW = CR2CRIntegerWithinImage(img.shape[1], img.shape[0], cs, rs, {'imgMargins':options['imgMargins']})
    if len(csIW) == len(rsIW) == 0:
        return img
    if len(options['colors']) == 1:
        colors = [options['colors'][0] for item in range(len(csIW))]
    else:
        assert len(options['colors']) >= len(csIW) == len(rsIW)
        colors = options['colors']
    if options['size'] is not None:
        size = int(options['size'])
    else:
        size = int(np.sqrt(img.shape[0]*img.shape[1])/150) + 1
    for pos in range(len(csIW)):
        img = cv2.circle(img, (csIW[pos], rsIW[pos]), size, colors[pos], -1)
    return img
def DistanceFromAPointToAPoint(x0, y0, x1, y1): # 202206201445 (last read 2022-06-20)
    ''' comments:
    .- input x0, y0, x1 and y1 are floats or float-ndarrays of the same length
    .- output distance is a float or a float-ndarray
    '''
    distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return distance
def EulerianAngles2R(ph, sg, ta): # 202109131100 (last read 2022-06-29)
    ''' comments:
    .- input ph, sg and ta are floats
    .- output R is a orthonormal 3x3-float-ndarray positively oriented
    '''
    eu, ev, ef = EulerianAngles2UnitVectors(ph, sg, ta)
    R = UnitVectors2R(eu, ev, ef)
    return R
def EulerianAngles2UnitVectors(ph, sg, ta): # 202109231415 (last read 2022-06-29)
    ''' comments:
    .- input ph, sg and ta are floats
    .- output eu, ev and ef are 3-float-ndarrays which are orthonormal and positively oriented
    '''
    sph, cph = np.sin(ph), np.cos(ph)
    ssg, csg = np.sin(sg), np.cos(sg)
    sta, cta = np.sin(ta), np.cos(ta)
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
    return eu, ev, ef
def FindAffineA01(xs0, ys0, xs1, ys1): # 202111241134 (last read 2022-07-11)
    ''' comments:
    .- input xs0, ys0, xs1 and ys1 are float-ndarrays of the same length (>= 3)
    .- output A01 is a 6-float-ndarray or None (if it does not succeed)
    '''
    assert len(xs0) == len(ys0) == len(xs1) == len(ys1) >= 3
    A, b = np.zeros((2 * len(xs0), 6)), np.zeros(2 * len(xs0))
    poss0, poss1 = Poss0AndPoss1InFind2DTransform(len(xs0))
    A[poss0, 0], A[poss0, 1], A[poss0, 2], b[poss0] = xs0, ys0, np.ones(xs0.shape), xs1
    A[poss1, 3], A[poss1, 4], A[poss1, 5], b[poss1] = xs0, ys0, np.ones(xs0.shape), ys1
    try:
        A01 = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
    except:
        A01 = None
    return A01
def GetWPhaseFitting(ts, fs, Rt, options={}): # 202105281430 (last read 2022-07-10)
    ''' comments:
    .- input ts is a float-ndarray
    .- input fs is a complex-ndarray (time component of the EOF/PCA mode)
    .- input Rt is a float
    .- output w and wStd are floats (-999. if it does not succed)
    '''
    keys, defaultValues = ['pathFigureOut'], [None]
    options = CompleteADictionary(options, keys, defaultValues)
    assert np.min(np.diff(ts)) > 0
    tsIn, wsIn = [np.asarray([]) for item in range(2)]
    for posT, t in enumerate(ts):
        if not (np.min(ts)+Rt <= ts[posT] <= np.max(ts)-Rt): # we want the whole neighborhood within ts
            continue
        optionsTMP = {'radius':Rt, 'ordered':'byPosition'}
        possA = [posA for posA in range(len(ts)) if ts[posT]-Rt <= ts[posA] <= ts[posT]+Rt]
        A = np.ones((len(possA), 2)); A[:, 1] = ts[possA] - ts[posT]
        b = np.angle(fs[possA] * np.conj(fs[posT]))
        sol = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
        tsIn, wsIn = np.append(tsIn, ts[posT]), np.append(wsIn, np.abs(sol[1]))
    w, wStd = np.mean(wsIn), np.std(wsIn)
    if w <= 0.:
        w, wStd = -999., -999.
    if options['pathFigureOut'] is not None:
        plt.subplot(1, 2, 1)
        plt.plot(ts, np.angle(fs), 'k+')
        plt.subplot(1, 2, 2)
        if w > 0:
            plt.plot(tsIn, wsIn, 'k+')
            plt.plot(tsIn, w * np.ones(len(tsIn)), 'b-')
        plt.savefig(options['pathFigureOut'], dpi=200)
        plt.close('all')
    return w, wStd
def IntersectionOfTwoLines(line0, line1, options={}): # 202205260841 (last read 2022-07-02)
    ''' comments:
    .- input line0 and line1 are dictionaries (including at least 'lx', 'ly', 'lt')
    .- output xI and yI are floats or None (if the lines are parallel)
        .- output xI and yI is the point closest to the origin if the lines are coincident
    .- output case is a string ('point', 'coincident' or 'parallel')
    '''
    keys, defaultValues = ['epsilon'], [1.e-11]
    options = CompleteADictionary(options, keys, defaultValues)
    line0, line1 = [NormalizeALine(item) for item in [line0, line1]]
    detT = + line0['lx'] * line1['ly'] - line0['ly'] * line1['lx']
    detX = - line0['lt'] * line1['ly'] + line0['ly'] * line1['lt']
    detY = - line0['lx'] * line1['lt'] + line0['lt'] * line1['lx']
    if np.abs(detT) > options['epsilon']: # point
        xI, yI, case = detX / detT, detY / detT, 'point'
    elif max([np.abs(detX), np.abs(detY)]) < options['epsilon']: # coincident
        (xI, yI), case = PointInALineClosestToAPoint(line0, 0., 0.), 'coincident'
    else: # parallel
        xI, yI, case = None, None, 'parallel'
    return xI, yI, case
def Kappa2MuOneStep(kappa): # 202207091901 (last read 2022-07-09)
    ''' comments:
    .- input kappa is a float or a float-ndarray (kappa = w**2*h/g)
    .- ouput mu is a float or a float-ndarray (mu = k*h)
    '''
    kappa = np.clip(kappa, 1.e-12, 1.e+12)
    mu = kappa / np.sqrt(np.tanh(kappa)) * (1. + kappa ** 1.21272008 * np.exp(-1.32530358 + (-1.5943765 -0.14628616 * kappa) * kappa))
    return mu
def MainSet2HorizonLine(mainSet): # 202109141400 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'nc', 'nr, 'zc', 'z0', 'radiusOfEarth', 'ef', 'xc', 'efx', 'yc', 'efy', 'Pa', 'orderOfTheHorizonPolynomial')
    .- output horizonLine is a dictionary
    '''
    horizonLine = {key:mainSet[key] for key in ['nc', 'nr']}
    bp = np.sqrt(2. * max([1.e-2, mainSet['zc'] - mainSet['z0']]) * mainSet['radiusOfEarth']) / np.sqrt(np.sum(mainSet['ef'][0:2] ** 2))
    px, py, pz, vx, vy, vz = mainSet['xc'] + bp * mainSet['efx'], mainSet['yc'] + bp * mainSet['efy'], -max([1.e-2, mainSet['zc']-2.*mainSet['z0']]), -mainSet['efy'], +mainSet['efx'], 0.
    dc, cc = np.sum(mainSet['Pa'][0, 0:3] * np.asarray([vx, vy, vz])), np.sum(mainSet['Pa'][0, 0:3] * np.asarray([px, py, pz])) + mainSet['Pa'][0, 3]
    dr, cr = np.sum(mainSet['Pa'][1, 0:3] * np.asarray([vx, vy, vz])), np.sum(mainSet['Pa'][1, 0:3] * np.asarray([px, py, pz])) + mainSet['Pa'][1, 3]
    dd, cd = np.sum(mainSet['Pa'][2, 0:3] * np.asarray([vx, vy, vz])), np.sum(mainSet['Pa'][2, 0:3] * np.asarray([px, py, pz])) + 1.
    ccUh1, crUh1, ccUh0 = dr * cd - dd * cr, dd * cc - dc * cd, dc * cr - dr * cc
    TMP = max([np.sqrt(ccUh1 ** 2 + crUh1 ** 2), 1.e-8])
    horizonLine['ccUh1'], horizonLine['crUh1'], horizonLine['ccUh0'] = [item / TMP for item in [ccUh1, crUh1, ccUh0]]
    horizonLine['crUh1'] = ClipWithSign(horizonLine['crUh1'], 1.e-8, 1.e+8)
    cUhs = np.linspace(-0.1 * mainSet['nc'], +1.1 * mainSet['nc'], 31, endpoint=True)
    rUhs = CUh2RUh(horizonLine, cUhs)
    cDhs, rDhs = CURU2CDRD(mainSet, cUhs, rUhs) # explicit
    A = np.ones((len(cDhs), mainSet['orderOfTheHorizonPolynomial'] + 1))
    for n in range(1, mainSet['orderOfTheHorizonPolynomial'] + 1):
        A[:, n] = cDhs ** n
    b = rDhs
    try:
        horizonLine['ccDh'] = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
        if np.max(np.abs(b - np.dot(A, horizonLine['ccDh']))) > 5e-1: # IMP* WATCH OUT
            horizonLine['ccDh'] = np.zeros(mainSet['orderOfTheHorizonPolynomial'] + 1)
            horizonLine['ccDh'][0] = 1.e+2 # IMP* WATCH OUT
    except:
        horizonLine['ccDh'] = np.zeros(mainSet['orderOfTheHorizonPolynomial'] + 1)
        horizonLine['ccDh'][0] = 1.e+2 # IMP* WATCH OUT
    return horizonLine
def MainSet2Pa(mainSet): # 202207061434 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'pc', 'eu', 'ev', 'ef', 'sca', 'sra', 'oc' and 'or'
    '''
    tu, tv, tf = [-np.sum(mainSet['pc'] * mainSet[item]) for item in ['eu', 'ev', 'ef']]
    P = np.zeros((3, 4))
    P[0, 0:3], P[0, 3] = mainSet['eu'] / mainSet['sca'] + mainSet['oc'] * mainSet['ef'], tu / mainSet['sca'] + mainSet['oc'] * tf
    P[1, 0:3], P[1, 3] = mainSet['ev'] / mainSet['sra'] + mainSet['or'] * mainSet['ef'], tv / mainSet['sra'] + mainSet['or'] * tf
    P[2, 0:3], P[2, 3] = mainSet['ef'], tf
    Pa = P / P[2, 3]
    return Pa
def MakeFolder(pathFolder): # 202109131100 (last read 2022-06-29)
    ''' comments:
    .- input pathFolder is a string
        .- pathFolder is created if it does not exist
    '''
    if not os.path.exists(pathFolder):
        os.makedirs(pathFolder)
    return None
def NormalizeALine(line, options={}): # 202206201459 (last read 2022-06-29)
    ''' comments:
    .- input line is a dictionary (including at least 'lx', 'ly' and 'lt')
        .- a line is so that line['lx'] * x + line['ly'] * y + line['lt'] = 0
        .- a normalized line is so that line['lx'] ** 2 + line['ly'] ** 2 = 1
    .- output line includes key 'isNormalized' (=True)
    .- output line maintains the orientation of input line
    '''
    keys, defaultValues = ['forceToNormalize'], [False]
    options = CompleteADictionary(options, keys, defaultValues)
    if options['forceToNormalize'] or 'isNormalized' not in line.keys() or not line['isNormalized']:
        lm = np.sqrt(line['lx'] ** 2 + line['ly'] ** 2)
        line = {item:line[item]/lm for item in ['lx', 'ly', 'lt']}
        line['isNormalized'] = True
    return line
def Pa2Pa11(Pa): # 202201250835 (last read 2022-07-06)
    ''' comments:
    .- input Pa is a 3x4-float-ndarray
    .- output Pa11 is a 11-float-ndarray
    '''
    Pa11 = np.ones(11)
    Pa11[0:4], Pa11[4:8], Pa11[8:11] = Pa[0, 0:4], Pa[1, 0:4], Pa[2, 0:3]
    return Pa11
def PointInALineClosestToAPoint(line, x, y): # 202205260853 (last read 2022-06-20, checked graphically with auxiliar code)
    ''' comments:
    .- input line is a dictionary (including at least 'lx', 'ly' and 'lt')
    .- input x and y are floats or float-ndarrays of the same length
    .- output xC and yC are floats or float-ndarrays
    '''
    line = NormalizeALine(line)
    xC = line['ly'] * (line['ly'] * x - line['lx'] * y) - line['lx'] * line['lt']
    yC = line['lx'] * (line['lx'] * y - line['ly'] * x) - line['ly'] * line['lt']
    return xC, yC
def Polyline2InnerHexagonalMesh(polyline, d, options={}): # 202205111346 (last read 2022-07-02)
    ''' comments:
    .- input polyline is a dictionary (including at least 'xs' and 'ys')
    .- input d is a float
    .- output xsM and ysM are float-ndarrays of the same length
    '''
    keys, defaultValues = ['epsilon'], [1.e-11]
    options = CompleteADictionary(options, keys, defaultValues)
    xsC, ysC = CloudOfPoints2Rectangle(polyline['xs'], polyline['ys'], options={'margin':2.1*d})[0:2]
    v1 = np.asarray([xsC[1] - xsC[0], ysC[1] - ysC[0]]); l1 = np.sqrt(np.sum(v1 ** 2)); u1 = v1 / l1; n1 = int(l1 / d) + 1
    v2 = np.asarray([xsC[2] - xsC[1], ysC[2] - ysC[1]]); l2 = np.sqrt(np.sum(v2 ** 2)); u2 = v2 / l2; n2 = int(l2 / (2. * d * np.cos(np.pi / 6.))) + 1 # cos(30º)
    xsM, ysM = [[] for item in range(2)]
    for i1, i2 in itertools.product(range(n1), range(n2)):
        x0 = xsC[0]
        y0 = ysC[0]
        xsM.append(x0 + i1 * d * u1[0] + i2 * 2. * d * np.cos(np.pi / 6.) * u2[0])
        ysM.append(y0 + i1 * d * u1[1] + i2 * 2. * d * np.cos(np.pi / 6.) * u2[1])
        x0 = xsC[0] + d / 2. * u1[0] + d * np.cos(np.pi / 6.) * u2[0]
        y0 = ysC[0] + d / 2. * u1[1] + d * np.cos(np.pi / 6.) * u2[1]
        xsM.append(x0 + i1 * d * u1[0] + i2 * 2. * d * np.cos(np.pi / 6.) * u2[0])
        ysM.append(y0 + i1 * d * u1[1] + i2 * 2. * d * np.cos(np.pi / 6.) * u2[1])
    xsM, ysM = [np.asarray(item) for item in [xsM, ysM]]
    posIns = CloudOfPoints2InnerPositionsForPolyline(xsM, ysM, polyline, options={'epsilon':options['epsilon']})
    xsM, ysM = [item[posIns] for item in [xsM, ysM]]
    return xsM, ysM
def Poss0AndPoss1InFind2DTransform(n): # 202207112001 (last read 2022-07-11)
    ''' comments:
    .- input n is an integer
    .- output poss0 and poss1 are integer-lists
    '''
    poss0 = [2*pos+0 for pos in range(n)]
    poss1 = [2*pos+1 for pos in range(n)]
    return poss0, poss1
def R2UnitVectors(R): # 202109131100 (last read 2022-06-29)
    ''' comments:
    .- input R is a 3x3-float-ndarray
        .- the rows of R are eu, ev and ef
    .- output eu, ev and ef are 3-float-ndarrays
    '''
    assert R.shape == (3, 3)
    eu, ev, ef = R[0, :], R[1, :], R[2, :]
    return eu, ev, ef
def ReadCalTxt(pathCalTxt): # 202110131422 (last read 2022-07-01)
    ''' comments:
    .- input pathCalTxt is a string
    .- output allVariables is a 14-float-ndarray
    .- output nc and nr are integers
    .- output errorT is a float
    '''
    rawData = np.asarray(ReadRectangleFromTxt(pathCalTxt, {'c1':1, 'valueType':'float'}))
    allVariables, nc, nr, errorT = rawData[0:14], int(np.round(rawData[14])), int(np.round(rawData[15])), rawData[16]
    return allVariables, nc, nr, errorT
def ReadMainSetFromCalTxt(pathCalTxt, options={}): # 202111171714 (read 2022-07-06)
    ''' comments:
    .- input pathCalTxt is a string
    .- output mainSet is a dictionary
    '''
    keys, defaultValues = ['orderOfTheHorizonPolynomial', 'radiusOfEarth', 'z0'], [5, 6.371e+6, 0.]
    options = CompleteADictionary(options, keys, defaultValues)
    allVariables, nc, nr = ReadCalTxt(pathCalTxt)[0:3]
    mainSet = AllVariables2MainSet(allVariables, nc, nr, options={item:options[item] for item in ['orderOfTheHorizonPolynomial', 'radiusOfEarth', 'z0']})
    return mainSet
def ReadRectangleFromTxt(pathFile, options={}): # 202109141200
    assert os.path.isfile(pathFile)
    keys, defaultValues = ['c0', 'c1', 'r0', 'r1', 'valueType', 'nullLine'], [0, 0, 0, 0, 'str', None]
    options = CompleteADictionary(options, keys, defaultValues)
    openedFile = open(pathFile, 'r')
    listOfLines = openedFile.readlines()
    if options['nullLine'] is not None:
        listOfLines = [item for item in listOfLines if item[0] != options['nullLine']]
    if not (options['r0'] == 0 and options['r1'] == 0): # if r0 == r1 == 0 it loads all the rows
        listOfLines = [listOfLines[item] for item in range(options['r0'], options['r1'])]
    for posOfLine in range(len(listOfLines)-1, -1, -1):
        if listOfLines[posOfLine] == '\n':
            print('... line {:5} is empty'.format(posOfLine))
            del listOfLines[posOfLine]
    stringsInLines = [item.split() for item in listOfLines]
    rectangle = stringsInLines
    if not (options['c0'] == options['c1'] == 0): # if c0 == c1 == 0 it loads all the columns
        rectangle = [item[options['c0']:options['c1']] for item in rectangle]
    if options['valueType'] == 'str':
        pass
    elif options['valueType'] == 'float':
        rectangle = [[float(item) for item in line] for line in rectangle]
    elif options['valueType'] == 'int':
        rectangle = [[int(item) for item in line] for line in rectangle]
    else:
        assert False
    if options['c1'] - options['c0'] == 1: # one column
        rectangle = [item[0] for item in rectangle]
    if options['r1'] - options['r0'] == 1: # one row
        rectangle = rectangle[0]
    return rectangle
def TH2LOneStep(T, h, g): # 202207091907 (last read 2022-07-09)
    ''' comments:
    .- input T, h and are floats or float-ndarrays of the same length
    .- output L is a float or a float-ndarray
    '''
    w = 2. * np.pi / T
    kappa = w ** 2 * h / g
    mu = Kappa2MuOneStep(kappa)
    k = mu / h
    L = 2. * np.pi / k
    return L
def UDaVDa2UUaVUa(mainSet, uDas, vDas): # uD* and vD* -> uU* and vU* 202207061139 (last read 2022-07-06, checked graphically with auxiliar code) potentially expensive
    ''' comments:
    .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a', 'p2a')
    .- input uDas and vDas are float-ndarrays of the same length
    .- output uUas and vUas are float-ndarrays of the same length of uDas and vDas or Nones (if it does not succeed)
    .- the funcion is implicit unless k2a = p1a = p2a = 0
    '''
    def DeltaAndError220706(mainSet, uDas, vDas, uUas, vUas): # 202109131500
        uDasN, vDasN = UUaVUa2UDaVDa(mainSet, uUas, vUas)
        fxs, fys = uDasN - uDas, vDasN - vDas # errors
        error = np.max([np.max(np.abs(fxs)), np.max(np.abs(fys))])
        aux1s = uUas ** 2 + vUas ** 2
        aux1suUa = 2. * uUas
        aux1svUa = 2. * vUas
        aux2s = 1. + mainSet['k1a'] * aux1s + mainSet['k2a'] * aux1s ** 2
        aux2suUa = mainSet['k1a'] * aux1suUa + mainSet['k2a'] * 2. * aux1s * aux1suUa
        aux2svUa = mainSet['k1a'] * aux1svUa + mainSet['k2a'] * 2. * aux1s * aux1svUa
        aux3suUa = 2. * vUas
        aux3svUa = 2. * uUas
        aux4suUa = aux1suUa + 4. * uUas
        aux4svUa = aux1svUa
        aux5suUa = aux1suUa
        aux5svUa = aux1svUa + 4. * vUas
        JuUasuUa = aux2s + uUas * aux2suUa + mainSet['p2a'] * aux4suUa + mainSet['p1a'] * aux3suUa
        JuUasvUa = uUas * aux2svUa + mainSet['p2a'] * aux4svUa + mainSet['p1a'] * aux3svUa
        JvUasuUa = vUas * aux2suUa + mainSet['p1a'] * aux5suUa + mainSet['p2a'] * aux3suUa
        JvUasvUa = aux2s + vUas * aux2svUa + mainSet['p1a'] * aux5svUa + mainSet['p2a'] * aux3svUa
        determinants = JuUasuUa * JvUasvUa - JuUasvUa * JvUasuUa
        determinants = ClipWithSign(determinants, 1.e-8, 1. / 1.e-8)
        JinvuUasuUa = + JvUasvUa / determinants
        JinvvUasvUa = + JuUasuUa / determinants
        JinvuUasvUa = - JuUasvUa / determinants
        JinvvUasuUa = - JvUasuUa / determinants
        duUas = - JinvuUasuUa * fxs - JinvuUasvUa * fys
        dvUas = - JinvvUasuUa * fxs - JinvvUasvUa * fys
        return duUas, dvUas, error
    possZero = np.where(np.sqrt(uDas ** 2 + vDas ** 2) < 1.e-11)[0]
    if len(possZero) > 0:
        uDas[possZero], vDas[possZero] = [0.1 * np.ones(len(possZero)) for item in range(2)] # give another value (0.1) to proceed
    if np.allclose([mainSet['k2a'], mainSet['p1a'], mainSet['p2a']], [0., 0., 0.]): # explicit
        if np.allclose(mainSet['k1a'], 0.):
            uUas, vUas = uDas * 1., vDas * 1.
        else: # Cardano's solution
            xis = mainSet['k1a'] * (uDas ** 2 + vDas ** 2)
            xs = Xi2XForParabolicDistortion(xis) # = mainSet['k1a'] * (uUas ** 2 + vUas ** 2)
            uUas, vUas = [item / (1 + xs) for item in [uDas, vDas]]
        converged = True
    else: # implicit (Newton using DeltaAndError220706)
        uUas, vUas, error, converged, counter, speed = 1. * uDas, 1. * vDas, 1.e+11, False, 0, 1. # initialize undistorted with distorted
        while not converged and counter <= 20:
            duUas, dvUas, errorN = DeltaAndError220706(mainSet, uDas, vDas, uUas, vUas)
            if errorN > 2. * error:
                break
            uUas, vUas, error = uUas + speed * duUas, vUas + speed * dvUas, 1. * errorN
            converged, counter = error <= 1.e-11, counter + 1
    if not converged:
        uUas, vUas = None, None
    else:
        if len(possZero) > 0:
            uDas[possZero], vDas[possZero] = 0., 0.
            uUas[possZero], vUas[possZero] = 0., 0.
        uDasR, vDasR = UUaVUa2UDaVDa(mainSet, uUas, vUas)
        assert max([np.max(np.abs(uDasR - uDas)), np.max(np.abs(vDasR - vDas))]) < 5. * 1.e-11
    return uUas, vUas
def UUaVUa2UDaVDa(mainSet, uUas, vUas): # uU* and vU* -> uD* and vD* 202109131500 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'k1a', 'k2a', 'p1a' and 'p2a')
    .- input uUas and vUas are floats or float-ndarrays of the same length
    .- output uDas and vDas are floats or float-ndarrays of the same length of uUas and vUas
    '''
    aux1s = uUas ** 2 + vUas ** 2
    aux2s = 1. + mainSet['k1a'] * aux1s + mainSet['k2a'] * aux1s ** 2
    aux3s = 2. * uUas * vUas
    aux4s = aux1s + 2. * uUas ** 2
    aux5s = aux1s + 2. * vUas ** 2
    uDas = uUas * aux2s + mainSet['p2a'] * aux4s + mainSet['p1a'] * aux3s
    vDas = vUas * aux2s + mainSet['p1a'] * aux5s + mainSet['p2a'] * aux3s
    return uDas, vDas
def UUaVUa2XYZ(mainSet, planes, uUas, vUas, options={}): # 202109141800 (last read 2022-07-12) potentially expensive
    ''' comments:
    .- input mainSet is a dictionary (including at least 'eu', 'ev', 'ef' and 'pc')
    .- input planes is a dictionary (including at least 'pxs', 'pys', 'pzs' and 'pts')
        .- input planes['pxs'/'pys'/'pzs'/'pts'] is a float or a float-ndarray of the same length of uUas and vUas
    .- input uUas and vUas are float-ndarrays of the same length
    .- output xs, ys, zs are float-ndarrays of the same length of uUas and vUas
    .- output possRightSideOfCamera is a list of integers or None (if not options['returnPositionsRightSideOfCamera'])
    '''
    keys, defaultValues = ['returnPositionsRightSideOfCamera'], [False]
    options = CompleteADictionary(options, keys, defaultValues)
    A11s = mainSet['ef'][0] * uUas - mainSet['eu'][0]
    A12s = mainSet['ef'][1] * uUas - mainSet['eu'][1]
    A13s = mainSet['ef'][2] * uUas - mainSet['eu'][2]
    bb1s = uUas * np.sum(mainSet['pc'] * mainSet['ef']) - np.sum(mainSet['pc'] * mainSet['eu'])
    A21s = mainSet['ef'][0] * vUas - mainSet['ev'][0]
    A22s = mainSet['ef'][1] * vUas - mainSet['ev'][1]
    A23s = mainSet['ef'][2] * vUas - mainSet['ev'][2]
    bb2s = vUas * np.sum(mainSet['pc'] * mainSet['ef']) - np.sum(mainSet['pc'] * mainSet['ev'])
    A31s = + planes['pxs'] # float or float-ndarray
    A32s = + planes['pys'] # float or float-ndarray
    A33s = + planes['pzs'] # float or float-ndarray
    bb3s = - planes['pts'] # float or float-ndarray
    dens = A11s * (A22s * A33s - A23s * A32s) + A12s * (A23s * A31s - A21s * A33s) + A13s * (A21s * A32s - A22s * A31s)
    dens = ClipWithSign(dens, 1.e-11, 1.e+11) # it was 1.e-8, 1.e+8
    xs = (bb1s * (A22s * A33s - A23s * A32s) + A12s * (A23s * bb3s - bb2s * A33s) + A13s * (bb2s * A32s - A22s * bb3s)) / dens
    ys = (A11s * (bb2s * A33s - A23s * bb3s) + bb1s * (A23s * A31s - A21s * A33s) + A13s * (A21s * bb3s - bb2s * A31s)) / dens
    zs = (A11s * (A22s * bb3s - bb2s * A32s) + A12s * (bb2s * A31s - A21s * bb3s) + bb1s * (A21s * A32s - A22s * A31s)) / dens
    poss = np.where(np.max(np.asarray([np.abs(xs), np.abs(ys), np.abs(zs)]), axis=0) < 1.e+8)[0]
    if isinstance(planes['pxs'], np.ndarray):
        auxs = planes['pxs'][poss] * xs[poss] + planes['pys'][poss] * ys[poss] + planes['pzs'][poss] * zs[poss] + planes['pts'][poss]
    else:
        auxs = planes['pxs'] * xs[poss] + planes['pys'] * ys[poss] + planes['pzs'] * zs[poss] + planes['pts']
    assert np.allclose(auxs, np.zeros(len(poss)))
    poss = np.where(np.max(np.asarray([np.abs(xs), np.abs(ys), np.abs(zs)]), axis=0) < 1.e+8)[0]
    uUasR, vUasR = XYZ2UUaVUa(mainSet, xs[poss], ys[poss], zs[poss], options={})[0:2]
    assert (np.allclose(uUasR, uUas[poss]) and np.allclose(vUasR, vUas[poss]))
    if options['returnPositionsRightSideOfCamera']:
        possRightSideOfCamera = XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs)
    else:
        possRightSideOfCamera = None
    return xs, ys, zs, possRightSideOfCamera
def UaVa2CR(mainSet, uas, vas): # 202109101200 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'sca', 'sra', 'oc' and 'or')
        .- mainSet['sca'] and mainSet['sra'] are non-zero, but allowed to be negative
    .- input uas and vas are floats or float-ndarrays of the same length
    .- output cs and rs are floats or float-ndarrays of the same length of uas and vas
    '''
    cs = uas / mainSet['sca'] + mainSet['oc']
    rs = vas / mainSet['sra'] + mainSet['or']
    return cs, rs
def UnitVectors2R(eu, ev, ef): # 202109231416 (last read 2022-06-29)
    ''' comments:
    .- input eu, ev and ef are 3-float-ndarrays
    .- output R is a 3x3-float-ndarray
        .- the rows of R are eu, ev and ef
    '''
    R = np.asarray([eu, ev, ef])
    return R
def Video2Snaps(pathVideo, pathFldSnaps, fps, options={}): # 202109271245 (last read 2022-07-10)
    ''' comments:
    .- input pathVideo is a string
    .- input pathFldSnaps is a string
    .- input fps is a float (desired frames per second)
    '''
    keys, defaultValues = ['extension'], ['png']
    options = CompleteADictionary(options, keys, defaultValues)
    MakeFolder(pathFldSnaps)
    fnVideo = os.path.splitext(os.path.split(pathVideo)[1])[0]
    video = cv2.VideoCapture(pathVideo)
    fpsOfVideo = video.get(cv2.CAP_PROP_FPS)
    fps = min([fpsOfVideo / int(fpsOfVideo / fps), fpsOfVideo])
    milisecond = 0.
    hasFrame = WriteASnap(video, fnVideo, milisecond, pathFldSnaps, options={'extension':options['extension']})
    while hasFrame:
        milisecond = milisecond + 1000. / fps
        hasFrame = WriteASnap(video, fnVideo, milisecond, pathFldSnaps, options={'extension':options['extension']})
    return None
def WGH2KOneStep(w, g, h): # 202207091928 (last read 2022-07-09)
    ''' comments:
    .- input w, g and h are floats or float-ndarrays of the same length
    .- output k is a float or a float-ndarray
    '''
    kappa = w ** 2 * h / g
    mu = Kappa2MuOneStep(kappa)
    k = mu / h
    return k
def WriteASnap(video, fnVideo, milisecond, pathFldSnaps, options={}): # 202109271244 (last read 2022-07-10)
    ''' comments:
    .- input video is a cv2.VideoCapture
    .- input fnVideo is a string
    .- input milisecond is a float
    .- input pathFldSnaps is a string
    .- output hasFrame is a boolean
    '''
    keys, defaultValues = ['extension'], ['png']
    options = CompleteADictionary(options, keys, defaultValues)
    video.set(cv2.CAP_PROP_POS_MSEC, milisecond)
    pathSnap = pathFldSnaps + os.sep + '{:}_{:}.{:}'.format(fnVideo, str(int(milisecond)).zfill(12), options['extension']) # IMP* milisecond
    hasFrame, img = video.read()
    if hasFrame:
        MakeFolder(os.path.split(pathSnap)[0])
        cv2.imwrite(pathSnap, img)
    return hasFrame
def X2DMD(X, r): # 202207121101 (last read 2022-07-12)
    ''' comments:
    .- input X is a  (nx x nt)-float-ndarray
    .- input r is an integer
    .- output Phi and Lambda are float-ndarrays
        .- output Phi[:, posOfMode] are the space-modes
        .- output np.imag(np.log(Lambda[posOfMode]) / dt) is the period (dt is the time step)
    '''
    X0, X1 = X[:, 0:-1], X[:, 1:]
    U, S, Vstar = np.linalg.svd(X0, full_matrices=False)
    Ustar, V = [np.conj(np.transpose(item)) for item in [U, Vstar]]
    U, S, V = U[:, 0:r], S[0:r], V[:, 0:r]
    Ustar, Vstar = [np.conj(np.transpose(item)) for item in [U, V]]
    Sinv = np.diag(1 / S)
    B = np.dot(Ustar, np.dot(X1, np.dot(V, Sinv)))
    Lambda, W = np.linalg.eig(B)
    Phi = np.dot(X1, np.dot(V, np.dot(Sinv, W)))
    return Phi, Lambda
def X2EOF(X): # 202002161020 (last read 2022-07-12)
    ''' comments:
    .- input X is a (nx x nt)-float-ndarray
    .- output EOF is a dictionary (including 'EOFs', 'amplitudesForEOFs', 'explainedVariances', 'cumulativeExplainedVariances')
    '''
    (nx, nt), nq = X.shape, np.min(X.shape)
    XMean, XHat = X2XMean(X), X2XHat(X)
    U, S, VH = np.linalg.svd(XHat, full_matrices=False) # (nx x nq)-ndarray, nq array, nq x nt ndarray
    SVH = np.dot(np.diag(S), VH) # nq x nt ndarray
    variances = np.dot(SVH, np.conj(SVH.T)) # nq x nq ndarray
    assert U.shape == (nx, nq) and S.shape == (nq, ) and VH.shape == (nq, nt) 
    EOF = {'nx':nx, 'nt':nt, 'nq':nq, 'U':U, 'SVH':SVH}
    EOF['EOF0'] = XMean # nx-array of floats
    EOF['EOFs'] = [U[:, item] for item in range(0, nq)] # nq-list of nx-arrays of floats
    EOF['amplitudesForEOFs'] = [SVH[item, :] for item in range(0, nq)] # nq-list of nt-arrays of floats
    EOF['explainedVariances'] = np.diag(variances) / np.sum(np.diag(variances)) # nq-array of floats
    EOF['cumulativeExplainedVariances'] = np.cumsum(EOF['explainedVariances']) # nq-array of floats
    return EOF
def X2LSByCandes(X, options={}): # ROBUST PCA by Candes
    class RPCAByCandes:
        def __init__(self, D, mu=None, lmbda=None):
            self.D = D
            self.S = np.zeros(self.D.shape)
            self.Y = np.zeros(self.D.shape)
            if mu:
                self.mu = mu
            else:
                self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))
            self.mu_inv = 1 / self.mu
            if lmbda:
                self.lmbda = lmbda
            else:
                self.lmbda = 1 / np.sqrt(np.max(self.D.shape))
        @staticmethod
        def frobenius_norm(M):
            return np.linalg.norm(M, ord='fro')
        @staticmethod
        def shrink(M, tau):
            return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))
        def svd_threshold(self, M, tau):
            U, S, V = np.linalg.svd(M, full_matrices=False)
            return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))
        def fit(self, tol=None, max_iter=10000, iter_print=100):
            iter = 0
            err = np.Inf
            Sk = self.S
            Yk = self.Y
            Lk = np.zeros(self.D.shape)
            if tol:
                _tol = tol
            else:
                _tol = 1E-7 * self.frobenius_norm(self.D)
            while (err > _tol) and iter < max_iter:
                Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv) # this line implements step 3
                Sk = self.shrink(self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda) # this line implements step 4
                Yk = Yk + self.mu * (self.D - Lk - Sk) # this line implements step 5
                err = self.frobenius_norm(self.D - Lk - Sk)
                iter += 1
            self.L = Lk
            self.S = Sk
            return Lk, Sk
    keys, defaultValues = ['max_iter', 'iter_print', 'mu'], [10000, 100, None]
    options = CompleteADictionary(options, keys, defaultValues)
    rpca = RPCAByCandes(X, mu=options['mu'])
    L, S = rpca.fit(max_iter=options['max_iter'], iter_print=options['iter_print'])
    return L, S
def X2XHat(X): # 202002161020 (last read 2022-07-12)
    ''' comments:
    .- input X is a (nx x nt)-float-ndarray
    .- output XHat is a (nx x nt)-float-ndarray is so that each row sums 0
    '''
    XHat = X - np.outer(X2XMean(X), np.ones(X.shape[1]))
    assert np.allclose(XHat.mean(axis=1), np.zeros(X.shape[0]))
    return XHat
def X2XMean(X): # 202002161020 (last read 2022-07-12)
    ''' comments:
    .- input X is a (nx x nt)-float-ndarray
    .- output XMean is a nx-float-ndarray
    '''
    XMean = np.mean(X, axis=1)
    assert len(XMean) == X.shape[0]
    return XMean    
def XYZ2CDRD(mainSet, xs, ys, zs, options={}): # 202109131600 (last read 2022-07-12)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'nc' and 'nr')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output cDs and rDs are float-ndarrays of the same length of xs, ys and zs
    .- output possGood is a list of integers or None (if not options['returnGoodPositions'])
    '''
    keys, defaultValues = ['imgMargins', 'returnGoodPositions'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
    options = CompleteADictionary(options, keys, defaultValues)
    uUas, vUas, possGood = XYZ2UUaVUa(mainSet, xs, ys, zs, options={'returnPositionsRightSideOfCamera':options['returnGoodPositions']})
    uDas, vDas = UUaVUa2UDaVDa(mainSet, uUas, vUas)
    cDs, rDs = UaVa2CR(mainSet, uDas, vDas)
    if options['returnGoodPositions']: # so far possGood are at the right side of the camera
        if len(possGood) > 0:
            nc, nr, cDsGood, rDsGood = mainSet['nc'], mainSet['nr'], cDs[possGood], rDs[possGood]
            possGoodInGood = CR2PositionsWithinImage(nc, nr, cDsGood, rDsGood, options={'imgMargins':options['imgMargins']})
            possGood = [possGood[item] for item in possGoodInGood]
        if len(possGood) > 0:
            xsGood, ysGood, zsGood, cDsGood, rDsGood = [item[possGood] for item in [xs, ys, zs, cDs, rDs]]
            xsGoodR, ysGoodR = CDRDZ2XY(mainSet, cDsGood, rDsGood, zsGood, options={})[0:2] # all, not only good positions; potentially expensive
            possGoodInGood = np.where(np.sqrt((xsGood - xsGoodR) ** 2 + (ysGood - ysGoodR) ** 2) < 1.e-5)[0] # 1.e-5 could be changed
            possGood = [possGood[item] for item in possGoodInGood]
    else: # possGood is None from XYZ2UUaVUa above
        assert possGood is None
    return cDs, rDs, possGood
def XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs): # 202109231412 (last read 2022-07-06)
    '''
    .- input mainSet is a dictionary (including 'xc', 'yc', 'zc' and 'ef')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output possRightSideOfCamera is a integer-list
    '''
    xas, yas, zas = xs - mainSet['xc'], ys - mainSet['yc'], zs - mainSet['zc']
    possRightSideOfCamera = np.where(xas * mainSet['ef'][0] + yas * mainSet['ef'][1] + zas * mainSet['ef'][2] > 0)[0]
    return possRightSideOfCamera
def XYZ2UUaVUa(mainSet, xs, ys, zs, options={}): # 202109231411 (last read 2022-07-06)
    ''' comments:
    .- input mainSet is a dictionary (including at least 'xc', 'yc', 'zc', 'eu', 'ev' and 'ef')
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output uUas and vUas are float-ndarrays of the same length of xs, ys and zs
    .- output possRightSideOfCamera is a list of integers or None (if not options['returnPositionsRightSideOfCamera'])
    '''
    keys, defaultValues = ['returnPositionsRightSideOfCamera'], [False]
    options = CompleteADictionary(options, keys, defaultValues)
    xas, yas, zas = xs - mainSet['xc'], ys - mainSet['yc'], zs - mainSet['zc']
    dns = xas * mainSet['ef'][0] + yas * mainSet['ef'][1] + zas * mainSet['ef'][2]
    dns = ClipWithSign(dns, 1.e-8, 1.e+8)
    uUas = (xas * mainSet['eu'][0] + yas * mainSet['eu'][1] + zas * mainSet['eu'][2]) / dns
    vUas = (xas * mainSet['ev'][0] + yas * mainSet['ev'][1] + zas * mainSet['ev'][2]) / dns
    if options['returnPositionsRightSideOfCamera']:
        possRightSideOfCamera = XYZ2PositionsRightSideOfCamera(mainSet, xs, ys, zs)
    else:
        possRightSideOfCamera = None
    return uUas, vUas, possRightSideOfCamera
def Xi2XForParabolicDistortion(xis): # 202207060912 (last read 2022-07-06, checked graphically with auxiliar code)
    ''' comments:
    .- input xis is a float-ndarray
    .- output xs is a float-ndarray
        .- it is solved: xis = xs ** 3 + 2 * xs ** 2 + xs
    '''
    p, qs, Deltas = -1. /3., -(xis + 2. / 27.), (xis + 4. / 27.) * xis
    if np.max(Deltas) < 0: # for xis in (-4/27, 0)
        n3s = (qs + 1j * np.sqrt(np.abs(Deltas))) / 2.
        ns = np.abs(n3s) ** (1. / 3.) * np.exp(1j * (np.abs(np.angle(n3s)) + 2. * np.pi * 1.) / 3.) # we ensure theta > 0; + 2 pi j for j = 0, 1, 2
    elif np.min(Deltas) >= 0: # for xis not in (-4/27, 0)
        auxs = (qs + np.sqrt(Deltas)) / 2.
        ns = np.sign(auxs) * (np.abs(auxs) ** (1. / 3.))
    else:
        possN, possP, ns = np.where(Deltas < 0)[0], np.where(Deltas >= 0)[0], np.zeros(xis.shape) + 1j * np.zeros(xis.shape)
        n3sN = (qs[possN] + 1j * np.sqrt(np.abs(Deltas[possN]))) / 2.
        ns[possN] = np.abs(n3sN) ** (1. / 3.) * np.exp(1j * (np.abs(np.angle(n3sN)) + 2. * np.pi * 1.) / 3.) # we ensure theta > 0; + 2 pi j for j = 0, 1, 2
        auxs = (qs[possP] + np.sqrt(Deltas[possP])) / 2.
        ns[possP] = np.sign(auxs) * (np.abs(auxs) ** (1. / 3.))
    xs = np.real(p / (3. * ns) - ns - 2. / 3.)
    return xs
