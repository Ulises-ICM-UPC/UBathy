# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ by Gonzalo Simarro and Daniel Calvete
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
import copy
import cv2  # type: ignore
import json
import numpy as np  # type: ignore
import os
import scipy as sc  # type: ignore
from scipy import optimize  # type: ignore
from scipy import signal  # type: ignore
from scipy.interpolate import griddata  # type: ignore
import sys
#
import ulises_ubathy as uli  # type: ignore
#
# ~~~~~~ data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
rangeC, grav = 'close', sc.constants.g
extsVids, extsImgs = ['mp4', 'avi', 'mov'], ['jpeg', 'jpg', 'png']
dGit = uli.GHLoadDGit()
#
# ~~~~~~ main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
def Inform_UBathy(pathFldMain, pos):  # lm:2025-06-08; lr:2025-07-08
    #
    # obtain par and inform
    par = uli.GHLoadPar(pathFldMain)
    uli.GHInform_2506('UBathy', pathFldMain, par, pos, margin=dGit['ind'], sB='', nFill=10)
    #
    return None
#
def Video2Frames(pathFldMain):  # lm:2025-07-08; lr:2025-07-08
    #
    # obtain par and videos
    par, videos = uli.GHBathyParAndVideos(pathFldMain)
    #
    # extract frames for videos
    for video in videos:
        print("{}{} Extracting frames from video {}".format(' '*dGit['ind'], dGit['sB1'], video))
        pathFldFrames = uli.GHBathyExtractVideoToPathFldFrames(pathFldMain, video, extsVids=extsVids, fps=par['frame_rate'], extImg='png', overwrite=par['overwrite_outputs'])  # IMP*: png
        if uli.IsFldModified_2506(pathFldFrames):
            print("\033[F\033[K{}{} Video {} successfully processed: {} frames extracted {}".format(' '*dGit['ind'], dGit['sB1'], video, len(os.listdir(pathFldFrames)), dGit['sOK']))
        else:
            print("\033[F\033[K{}{} Video {} was already processed: {} frames found {}".format(' '*dGit['ind'], dGit['sB1'], video, len(os.listdir(pathFldFrames)), dGit['sOK']))
    #
    return None
#
def CreateMeshes(pathFldMain):  # lm:2025-06-26; lr:2025-06-30
    #
    # obtain par and videos
    par, videos = uli.GHBathyParAndVideos(pathFldMain)
    #
    # obtain xsBoun, ysBoun and plBoun; polygon; boundary of the domain where to obtain the bathymetry
    pathBounTxt = os.path.join(pathFldMain, 'data', 'boundary_xy.txt')  # IMP*: nomenclature
    try:
        dataTMP = np.loadtxt(pathBounTxt, usecols=range(2), dtype=float, ndmin=2)
        xsBoun, ysBoun = [dataTMP[:, item] for item in range(2)]
        plBoun = {'xs': xsBoun, 'ys': ysBoun}
    except Exception:
        print("! Unable to read {} {}".format(pathBounTxt, dGit['sKO']))
        sys.exit()
    #
    # create and write mesh_B; depends only on plBoun
    print("{}{} Creating mesh_B (bathymetry mesh)".format(' '*dGit['ind'], dGit['sB1']))
    pathTMPTxt = os.path.join(pathFldMain, 'output', 'numerics', 'mesh_B.txt')  # IMP*: nomenclature
    if not os.path.exists(pathTMPTxt) or par['overwrite_outputs']:
        # obtain xsB and ysB
        xsB, ysB = uli.Polyline2InnerHexagonalMesh_2506(plBoun, par['delta_B'])
        # write pathTMPTxt
        os.makedirs(os.path.dirname(pathTMPTxt), exist_ok=True)
        with open(pathTMPTxt, 'w') as fileout:
            for posB in range(len(xsB)):
                fileout.write('{:15.3f} {:15.3f}\n'.format(xsB[posB], ysB[posB]))  # IMP*: formatting
        # write pathScrJpg
        pathScrJpg = os.path.join(pathFldMain, 'output', 'plots', 'mesh_B.png')  # IMP*: nomenclature
        uli.GHBathyPlotMesh(pathScrJpg, xsB, ysB, dGit['fs'], dGit['fs'], dGit['fontsize'], dpi=dGit['dpiHQ'], xsBoun=xsBoun, ysBoun=ysBoun)
        # inform
        print("\033[F\033[K{}{} Mesh_B (bathymetry mesh) successfully created: {} points {}".format(' '*dGit['ind'], dGit['sB1'], len(xsB), dGit['sOK']))
    else:
        # inform
        nOfPoints = len(np.loadtxt(pathTMPTxt, usecols=range(2), dtype=float, ndmin=2)[:, 0])
        print("\033[F\033[K{}{} Mesh_B (bathymetry mesh) was already available: {} points {}".format(' '*dGit['ind'], dGit['sB1'], nOfPoints, dGit['sOK']))
    #
    # create and write mesh_M and mesh_K for each video; IMP*: ztsK and ztsM correspond to the free surface
    for video in videos:
        #
        # obtain isPlw and load basics, including zSea
        pathFldDVideo = os.path.join(pathFldMain, 'data', 'videos', video)
        pathsPlwTxts = [item.path for item in os.scandir(pathFldDVideo) if 'crxyz' in item.name and item.name.endswith('.txt')]  # IMP*: nomenclature
        pathsCalTxts = [item.path for item in os.scandir(pathFldDVideo) if item.name.endswith('cal.txt')]  # IMP*: nomenclature
        pathsZsTxts = [item.path for item in os.scandir(pathFldDVideo) if item.name.endswith('zs.txt')]  # IMP*: nomenclature
        if (len(pathsPlwTxts) == 1) == (len(pathsCalTxts) == len(pathsZsTxts) == 1):  # one, and only one, must be true
            print("! Invalid .txt files for video {} {}".format(video, dGit['sKO']))
            sys.exit()
        if len(pathsPlwTxts) == 1:
            # load pathPlwTxt
            isPlw, pathPlwTxt = True, pathsPlwTxts[0]
            dataTMP = np.loadtxt(pathPlwTxt, usecols=range(5), dtype=float, ndmin=2)
            csPlw, rsPlw, xsPlw, ysPlw, zsPlw = [dataTMP[:, item] for item in range(5)]
            if np.std(zsPlw) > 1.e-3:  # WATCH OUT: epsilon
                print("! Invalid z values found in crxyz-txt file for video {}: z must be constant {}".format(video, dGit['sKO']))
                sys.exit()
            zSea = np.mean(zsPlw)  # IMP*
            if not np.allclose(csPlw, np.round(csPlw)) or not np.allclose(rsPlw, np.round(rsPlw)):
                print("! Invalid c/r values found in crxyz-txt file for video {}: c and r must be integers {}".format(video, dGit['sKO']))
                sys.exit()
            csPlw, rsPlw = [np.round(item).astype(int) for item in [csPlw, rsPlw]]
            # obtain affine
            ACR2XYPlw = uli.FindAffineA01_2504(csPlw, rsPlw, xsPlw, ysPlw)
            xsPlwR, ysPlwR = uli.ApplyAffineA01_2504(ACR2XYPlw, csPlw, rsPlw)
            if np.max(np.hypot(xsPlwR - xsPlw, ysPlwR - ysPlw)) > 1.e-3:  # WATCH OUT: epsilon
                print("! Invalid c/r/x/y values found in crxyz-txt file for video {}: c, r, x, and y must correspond to an affine transformation {}".format(video, dGit['sKO']))
                sys.exit()
            AXY2CRPlw = uli.FindAffineA01_2504(xsPlw, ysPlw, csPlw, rsPlw)
        else:
            isPlw, pathCalTxt, pathZsTxt = False, pathsCalTxts[0], pathsZsTxts[0]
            dMCS = uli.LoadDMCSFromCalTxt_2502(pathCalTxt, rangeC, incHor=False)
            zSea = np.loadtxt(pathZsTxt, dtype=float)  # IMP*
        #
        # load imgFrame
        pathFldFrames = uli.GHBathyExtractVideoToPathFldFrames(pathFldMain, video, active=False)
        if par['generate_scratch_plots']:
            pathsImgs = [item.path for item in os.scandir(pathFldFrames) if os.path.splitext(item.name)[1][1:] in extsImgs]
            if len(pathsImgs) == 0:
                print("! Unable to read frames for video {} {}".format(video, dGit['sKO']))
                sys.exit()
            imgFrame = cv2.imread(pathsImgs[0])  # WATCH OUT: the first one for instance
        #
        # obtain and write mesh_M; within the bathymetry boundary and within the planview/image ?BUFFER ZONE?
        print("{}{} Creating mesh_M (mode-decomposition mesh) for video {}".format(' '*dGit['ind'], dGit['sB1'], video))
        pathTMPNpz = os.path.join(pathFldMain, 'scratch', 'numerics', video, 'mesh_M.npz')  # IMP*: nomenclature
        if not os.path.exists(pathTMPNpz) or par['overwrite_outputs']:
            #
            # obtain initial mesh_M
            if isPlw:  # pixels, integers, correspond to the planview
                # obtain csM, rsM, xsM, ysM and ztsM
                mcsM, mrsM = np.meshgrid(np.arange(min(csPlw), max(csPlw)+1), np.arange(min(rsPlw), max(rsPlw)+1))  # IMP*: max included
                csM, rsM = [np.reshape(item, -1) for item in [mcsM, mrsM]]
                (xsM, ysM), ztsM = uli.ApplyAffineA01_2504(ACR2XYPlw, csM, rsM), zSea * np.ones(csM.shape)
            else:  # pixels, integers, correspond to the oblique image
                # obtain initial mesh in xyz
                xsM, ysM = uli.Polyline2InnerHexagonalMesh_2506(plBoun, par['delta_M'])
                ztsM = zSea * np.ones(xsM.shape)
                # obtain initial mesh in cr
                csM, rsM, possG = uli.XYZ2CDRD_2410(xsM, ysM, ztsM, dMCS, rtrnPossG=True, margin=0)
                csM, rsM = [np.round(item[possG]).astype(int) for item in [csM, rsM]]  # IMP*: integers
                # obtain updated mesh in cr
                csM, rsM = map(np.asarray, zip(*dict.fromkeys(zip(csM, rsM))))  # IMP*: eliminates duplicates
                # obtain updated mesh in xyz and cr
                ztsM = zSea * np.ones(csM.shape)
                xsM, ysM, possG = uli.CDRDZ2XY_2410(csM, rsM, ztsM, dMCS, rtrnPossG=True, margin=0)
                csM, rsM, xsM, ysM, ztsM = [item[possG] for item in [csM, rsM, xsM, ysM, ztsM]]
                # sort mesh using the distance to (x0, y0)
                xC, yC = np.mean(xsM), np.mean(ysM)
                dC = np.hypot(xC - dMCS['xc'], yC - dMCS['yc'])
                x0, y0 = xC + 5000 * (xC - dMCS['xc']) / dC, yC + 5000 * (yC - dMCS['yc']) / dC  # IMP*
                possS = np.argsort(np.hypot(xsM - x0, ysM - y0))
                csM, rsM, xsM, ysM, ztsM = [item[possS] for item in [csM, rsM, xsM, ysM, ztsM]]
                # obtain updated mesh in xyz and cr; uniformized in xy
                csMu, rsMu, xsMu, ysMu, ztsMu = [[item[0]] for item in [csM, rsM, xsM, ysM, ztsM]]
                for pos in range(1, len(csM)):  # potential point to the final mesh
                    if np.min(np.hypot(xsM[pos] - xsMu, ysM[pos] - ysMu)) >= 0.95 * par['delta_M']:
                        for L, item in zip([csMu, rsMu, xsMu, ysMu, ztsMu], [csM, rsM, xsM, ysM, ztsM]): L.append(item[pos])
                csM, rsM, xsM, ysM, ztsM = map(np.asarray, [csMu, rsMu, xsMu, ysMu, ztsMu])
            #
            # update mesh_M; within the bathymetry boundary
            possI = uli.CloudOfPoints2PossInsidePolyline_2508(xsM, ysM, plBoun)
            csM, rsM, xsM, ysM, ztsM = [item[possI] for item in [csM, rsM, xsM, ysM, ztsM]]
            #
            # write pathTMPNpz
            os.makedirs(os.path.dirname(pathTMPNpz), exist_ok=True)
            np.savez(pathTMPNpz, cs=csM, rs=rsM, xs=xsM, ys=ysM, zts=ztsM)  # IMP*: nomenclature
            #
            # write pathsJpgTMPs
            if par['generate_scratch_plots']:
                # mesh in xy
                pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', video, 'mesh_M.jpg')  # IMP*: nomenclature
                uli.GHBathyPlotMesh(pathScrJpg, xsM, ysM, dGit['fs'], dGit['fs'], dGit['fontsize'], dpi=dGit['dpiLQ'], xsBoun=xsBoun, ysBoun=ysBoun)
                # mesh in imgFrame
                pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', video, 'in_image_mesh_M.jpg')  # IMP*: nomenclature
                uli.DisplayCRInImage_2504(imgFrame, csM, rsM, colors=[[0, 255, 255]], margin=0, factor=0.1, pathOut=pathScrJpg)
            #
            # inform
            print("\033[F\033[K{}{} Mesh_M (mode-decomposition mesh) for video {} successfully generated: {} points {}".format(' '*dGit['ind'], dGit['sB1'], video, len(csM), dGit['sOK']))
        else:
            # inform
            print("\033[F\033[K{}{} Mesh_M (mode-decomposition mesh) for video {} was already available: {} points {}".format(' '*dGit['ind'], dGit['sB1'], video, len(np.load(pathTMPNpz)['cs']), dGit['sOK']))
        #
        # create and write mesh_K; within the bathymetry boundary and within the planview/image ?BUFFER ZONE?
        print("{}{} Creating mesh_K (wavenumber mesh) for video {}".format(' '*dGit['ind'], dGit['sB1'], video))
        pathTMPNpz = os.path.join(pathFldMain, 'scratch', 'numerics', video, 'mesh_K.npz')  # IMP*: nomenclature
        if not os.path.exists(pathTMPNpz) or par['overwrite_outputs']:
            #
            # obtain initial mesh_K
            xsK, ysK = uli.Polyline2InnerHexagonalMesh_2506(plBoun, par['delta_K'])
            ztsK = zSea * np.ones(xsK.shape)
            if isPlw:  # IMP*: pixels, floats, correspond to the planview; pixels not saved nor used except for scratch_plots
                possC = [np.where((csPlw == item[0]) & (rsPlw == item[1]))[0][0] for item in [[min(csPlw), min(rsPlw)], [max(csPlw), min(rsPlw)], [max(csPlw), max(rsPlw)], [min(csPlw), max(rsPlw)]]]
                possI = uli.CloudOfPoints2PossInsidePolyline_2508(xsK, ysK, {'xs': xsPlw[possC], 'ys': ysPlw[possC]})  # points of mesh_K in planview
                xsK, ysK, ztsK = [item[possI] for item in [xsK, ysK, ztsK]]
                csK, rsK = uli.ApplyAffineA01_2504(AXY2CRPlw, xsK, ysK)  # real valued; only for scratch_plots
            else:  # IMP*: pixels, floats, correspond to the oblique image; pixels not saved nor used except for scratch_plots
                csK, rsK, possG = uli.XYZ2CDRD_2410(xsK, ysK, ztsK, dMCS, rtrnPossG=True, margin=0)
                csK, rsK, xsK, ysK, ztsK = [item[possG] for item in [csK, rsK, xsK, ysK, ztsK]]
            #
            # write pathTMPNpz
            os.makedirs(os.path.dirname(pathTMPNpz), exist_ok=True)
            np.savez(pathTMPNpz, xs=xsK, ys=ysK, zts=ztsK)  # IMP*: only xsK, ysK and ztsK ; nomenclature
            #
            # write pathsJpgTMPs
            if par['generate_scratch_plots']:
                # mesh in xy
                pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', video, 'mesh_K.jpg')  # IMP*: nomenclature
                uli.GHBathyPlotMesh(pathScrJpg, xsK, ysK, dGit['fs'], dGit['fs'], dGit['fontsize'], dpi=dGit['dpiLQ'], xsBoun=xsBoun, ysBoun=ysBoun)
                # mesh in imgFrame
                pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', video, 'in_image_mesh_K.jpg')  # IMP*: nomenclature
                uli.DisplayCRInImage_2504(imgFrame, csK, rsK, colors=[[0, 255, 255]], margin=0, factor=0.1, pathOut=pathScrJpg)
            #
            # inform
            print("\033[F\033[K{}{} Mesh_K (wavenumber mesh) for video {} successfully created: {} points {}".format(' '*dGit['ind'], dGit['sB1'], video, len(xsK), dGit['sOK']))
        else:
            # inform
            print("\033[F\033[K{}{} Mesh_K (wavenumber mesh) for video {} was already available: {} points {}".format(' '*dGit['ind'], dGit['sB1'], video, len(np.load(pathTMPNpz)['xs']), dGit['sOK']))
    #
    return None
#
def ObtainWAndModes(pathFldMain):  # lm:2025-06-26; lr:2025-07-03
    #
    # obtain par and videos
    par, videos = uli.GHBathyParAndVideos(pathFldMain)
    #
    # obtain w and modes for each video
    for video in videos:
        #
        # obtain inform
        print("{}{} Computing mode decomposition for video {}".format(' '*dGit['ind'], dGit['sB1'], video))
        #
        # manage overwrite
        pathFldM = os.path.join(pathFldMain, 'scratch', 'numerics', video, 'M_modes')
        if os.path.exists(pathFldM) and not par['overwrite_outputs']:
            print("\033[F\033[K{}{} Mode decomposition for video {} was already available: {} modes found {}".format(' '*dGit['ind'], dGit['sB1'], video, len(os.listdir(pathFldM)), dGit['sOK']))
            continue
        if par['overwrite_outputs']:  # clean
            for pathFldTMP in [pathFldM, os.path.join(pathFldMain, 'scratch', 'plots', video, 'M_modes')]:
                uli.CleanAFld_2504(pathFldTMP)  # works also if pathFldTMP does not exist
        #
        # load pathFldFrames and fnsFrames
        pathFldFrames = uli.GHBathyExtractVideoToPathFldFrames(pathFldMain, video, active=False)
        fnsFrames = sorted([item for item in os.listdir(pathFldFrames) if os.path.splitext(item)[1][1:] in extsImgs])  # IMP*: sorted
        #
        # load ts in seconds; IMP*: frames can start at any time; IMP*: frames end 123456789012plw.ext or 123456789012.ext
        if os.path.splitext(fnsFrames[0])[0].endswith('plw'):  # 123456789012plw.ext all
            ts = np.asarray([int(os.path.splitext(fn)[0][-15:-3])/1000 for fn in fnsFrames])  # WATCH OUT: IMP*: ts in seconds, fn in milliseconds
        else:  # 123456789012.ext
            ts = np.asarray([int(os.path.splitext(fn)[0][-12:])/1000 for fn in fnsFrames])  # WATCH OUT: IMP*: ts in seconds, fn in milliseconds
        if len(ts) < 2:  # WATCH OUT: epsilon
            print("\033[F\033[K{}{} Video {} cannot be processed: duration too short {}".format(' '*dGit['ind'], dGit['sB1'], video, dGit['sWO']))
            continue
        ts, dtsMean = ts - ts[0], np.mean(np.diff(ts))  # IMP*: ts starts at 0
        ts = dtsMean * np.arange(len(ts))  # IMP*: we assume ~uniformity
        if par['decomposition_method'] == 'EOF' and par['min_period'] < 6 * dtsMean:
            print("\033[F\033[K{}{} Video {} cannot be processed: for EOF decomposition, 'min_period' must be > {:.2f} {}".format(' '*dGit['ind'], dGit['sB1'], video, 6*dtsMean, dGit['sWO']))
            continue
        #
        # load cs and rs for mesh_M; coordinates in the image, whether planview or oblique
        dataTMP = np.load(os.path.join(pathFldMain, 'scratch', 'numerics', video, 'mesh_M.npz'))
        csM, rsM, xsM, ysM = [dataTMP[item] for item in ['cs', 'rs', 'xs', 'ys']]
        if True:  # avoidable check
            assert np.allclose(csM, np.round(csM)) and np.allclose(rsM, np.round(rsM))
        csM, rsM = [np.round(item).astype(int) for item in [csM, rsM]]
        #
        # obtain nWtMax and nDtHilbert
        wtMax, dtHilbert = np.max(par['time_windows']), par['max_period']  # IMP*
        nWtMax, nDtHilbert = [int(item/dtsMean)+1 for item in [wtMax, dtHilbert]]
        if nWtMax + 2 * nDtHilbert > len(ts):
            if max(ts)-2*dtHilbert > 0:
                print("\033[F\033[K{}{} Video {} cannot be processed: max('time_windows') must be comfortably < {:.2f} s {}".format(' '*dGit['ind'], dGit['sB1'], video, max(ts)-2*dtHilbert, dGit['sWO']))
            else:
                print("\033[F\033[K{}{} Video {} cannot be processed: 'max_period' must be comfortably < {:.2f} s {}".format(' '*dGit['ind'], dGit['sB1'], video, max(ts)/2, dGit['sWO']))
            continue
        #
        # obtain nStep
        nStep = max(1, min(int(par['time_step'] / dtsMean), nWtMax - 1))  # IMP*: nWtMax-1 since there must be overlap for the updating scheme
        #
        # run through the video
        for posT in range(0, len(ts), nStep):
            #
            # disregard and inform
            if not 0 <= posT <= len(ts)-1-nWtMax-2*nDtHilbert-1:  # small margin
                continue
            print("{}{} Decomposition of subvideos starting at t = {:.1f} s".format(' '*2*dGit['ind'], dGit['sB2'], ts[posT+nDtHilbert]))  # IMP*: +nDtHilbert
            #
            # load or update XForTBallEx
            if posT == 0:  # initialize
                XForTBallEx = np.zeros((len(csM), nWtMax+2*nDtHilbert))  # Ex = "Hilbert Extended"
                possTLH = []
                for posH in range(nWtMax+2*nDtHilbert):
                    posTLH = posT + posH  # posH since posT = 0
                    possTLH.append(posTLH)
                    img = cv2.imread(os.path.join(pathFldFrames, fnsFrames[posTLH]))
                    XForTBallEx[:, posH] = img[rsM, csM, 1] / 255.  # load green channel
            else:
                nTMP = posT - posTOld
                XForTBallEx[:, :XForTBallEx.shape[1]-nTMP] = XForTBallEx[:, nTMP:XForTBallEx.shape[1]]  # IMP*: displaced nTMP
                possTLH = []
                for posH in range(nTMP):
                    posTLH = possTLHOld[-1] + 1 + posH  # IMP*: easy to read
                    possTLH.append(posTLH)
                    img = cv2.imread(os.path.join(pathFldFrames, fnsFrames[posTLH]))
                    XForTBallEx[:, XForTBallEx.shape[1]-nTMP+posH] = img[rsM, csM, 1] / 255.  # IMP*: easy to read, load green channel
            posTOld, possTLHOld = posT, possTLH
            #
            # perform RPCA and Hilbert to XForTBallExHilbert
            if par['candes_iterations'] > 0:
                XForTBallEx = uli.X2LSByCandes_2506(XForTBallEx, max_iter=par['candes_iterations'])[0]
            XForTBallExHilbert = signal.hilbert(XForTBallEx)
            #
            # crop the Hilbert extensions
            XForTBallHilbert = XForTBallExHilbert[:, nDtHilbert:-nDtHilbert]  # IMP*: crop the extension
            #
            # perform DMD/EOF for all 'time_windows'
            for wt in par['time_windows']:
                #
                # inform and obtain fnOut0
                print("{}{} Decomposition of subvideo from t = {:.1f} s to t = {:.1f} s".format(' '*3*dGit['ind'], dGit['sB3'], ts[posT+nDtHilbert], ts[posT+nDtHilbert] + wt))  # IMP*: +nDtHilbert
                fnOut0 = 't{}_wt{}'.format(str('{:.2f}'.format(ts[posT+nDtHilbert])).zfill(8), str('{:.2f}'.format(wt)).zfill(8))  # IMP*: +nDtHilbert
                #
                # obtain possWt and XForTBallHilbertWt
                possWt = range(np.min([int(wt/dtsMean), XForTBallHilbert.shape[1]]))
                XForTBallHilbertWt = XForTBallHilbert[:, possWt]
                #
                # obtain modes
                modes = {}
                if par['decomposition_method'] == 'DMD':
                    try:
                        Ts, phases, amplitudes = uli.X2DMD_2506(XForTBallHilbertWt, par['DMD_rank'], dt=dtsMean)  # IMP*: phases and amplitudes of the spatial component
                    except Exception:
                        continue
                    for posDMD in range(len(Ts)):
                        T, w = Ts[posDMD], 2 * np.pi / Ts[posDMD]
                        if not par['min_period'] <= T <= par['max_period']:
                            continue
                        modes[posDMD] = {'w': w, 'T': T, 'phases': phases[posDMD], 'amplitudes': amplitudes[posDMD]}
                elif par['decomposition_method'] == 'EOF':
                    try:
                        spatialEOFs, temporalEOFs, expVariances = uli.X2EOF_2502(XForTBallHilbertWt)[6:9]  # IMP* [6:9]
                    except Exception:
                        continue
                    for posEOF in range(len(expVariances)):
                        var = np.real(expVariances[posEOF])
                        if var < par['EOF_variance']:
                            continue
                        w, wStd = uli.GetWPhaseFitting_2506(dtsMean*np.arange(len(possWt)), temporalEOFs[posEOF], 3*dtsMean, verbose_plot=False)  # IMP*: 3*dtsMean as radius
                        if wStd / w > 0.15: # WATCH OUT: epsilon
                            continue
                        T = 2 * np.pi / w
                        if not par['min_period'] <= T <= par['max_period']:
                            continue
                        modes[posEOF] = {'var': var, 'wStdOverW': wStd/w, 'w': w, 'T': T, 'phases': np.angle(spatialEOFs[posEOF]), 'amplitudes': np.abs(spatialEOFs[posEOF])}
                else:
                    print("! Invalid 'decomposition_method' in parameters.json: must be 'DMD' or 'EOF'")
                    sys.exit()
                #
                # inform
                print("\033[F\033[K{}{} Subvideo from t = {:.1f} s to t = {:.1f} s processed: {} modes saved {}".format(' '*3*dGit['ind'], dGit['sB3'], ts[posT+nDtHilbert], ts[posT+nDtHilbert] + wt, len(modes), dGit['sOK']))  # IMP*: +nDtHilbert
                #
                # write modes
                for key in modes.keys():
                    # obtain fnOutWE
                    fnOutWE = '{}_T{}'.format(fnOut0, str('{:.2f}'.format(modes[key]['T'])).zfill(8))
                    # write pathTMPNpz
                    pathTMPNpz = os.path.join(pathFldMain, 'scratch', 'numerics', video, 'M_modes', '{}.npz'.format(fnOutWE))  # IMP*: nomenclature
                    os.makedirs(os.path.dirname(pathTMPNpz), exist_ok=True)
                    if par['decomposition_method'] == 'DMD':
                        np.savez(pathTMPNpz, w=modes[key]['w'], T=modes[key]['T'], phases=modes[key]['phases'], amplitudes=modes[key]['amplitudes'])  # IMP*: nomenclature
                    elif par['decomposition_method'] == 'EOF':
                        np.savez(pathTMPNpz, w=modes[key]['w'], T=modes[key]['T'], phases=modes[key]['phases'], amplitudes=modes[key]['amplitudes'], var=modes[key]['var'], wStdOverW=modes[key]['wStdOverW'])  # IMP*: nomenclature
                    # write pathScrJpg
                    if par['generate_scratch_plots']:
                        pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', video, 'M_modes', '{}.jpg'.format(fnOutWE))  # IMP*: nomenclature
                        uli.GHBathyPlotModes(pathScrJpg, xsM, ysM, modes[key]['phases'], modes[key]['amplitudes'], modes[key]['T'], dGit['fs'], dGit['fs'], dGit['fontsize'], dpi=dGit['dpiLQ'])
    #
    return None
#
def ObtainK(pathFldMain): # lm:2022-06-26; lr:2022-07-03
    #
    # obtain par and videos
    par, videos = uli.GHBathyParAndVideos(pathFldMain)
    #
    # obtain wavenumbers for each video
    for video in videos:
        #
        # inform
        print("{}{} Computing wavenumber fields for video {}".format(' '*dGit['ind'], dGit['sB1'], video))
        #
        # obtain pathFldM and disregard
        pathFldM = os.path.join(pathFldMain, 'scratch', 'numerics', video, 'M_modes')
        if not os.path.exists(pathFldM) or len(os.listdir(pathFldM)) == 0:
            print("\033[F\033[K{}{} No modes available for video {} {}".format(' '*dGit['ind'], dGit['sB1'], video, dGit['sWO']))
            continue
        #
        # load mesh_M
        dataTMP = np.load(os.path.join(pathFldMain, 'scratch', 'numerics', video, 'mesh_M.npz'))
        csM, rsM, xsM, ysM, ztsM = [dataTMP[item] for item in ['cs', 'rs', 'xs', 'ys', 'zts']]
        assert np.std(ztsM) < 1.e-3  # WATCH OUT: epsilon
        assert np.allclose(csM, np.round(csM)) and np.allclose(rsM, np.round(rsM))
        csM, rsM = [np.round(item).astype(int) for item in [csM, rsM]]
        #
        # load mesh_K
        dataTMP = np.load(os.path.join(pathFldMain, 'scratch', 'numerics', video, 'mesh_K.npz'))
        xsK, ysK, ztsK = [dataTMP[item] for item in ['xs', 'ys', 'zts']]
        assert np.std(ztsK) < 1.e-3  # WATCH OUT: epsilon
        #
        # manage overwrite
        pathFldK = os.path.join(pathFldMain, 'scratch', 'numerics', video, 'K_wavenumbers')
        if os.path.exists(pathFldK) and len(os.listdir(pathFldK)) == len(os.listdir(pathFldM)) and not par['overwrite_outputs']:
            print("\033[F\033[K{}{} Wavenumber fields for video {} were already available: {} fields found {}".format(' '*dGit['ind'], dGit['sB1'], video, len(os.listdir(pathFldK)), dGit['sOK']))
            continue
        if par['overwrite_outputs']:  # clean
            for pathFldTMP in [pathFldK, os.path.join(pathFldMain, 'scratch', 'plots', video, 'K_wavenumbers')]:
                uli.CleanAFld_2504(pathFldTMP)  # works also if pathFldTMP does not exist
        #
        # obtain ks for each subvideo mode
        for fnNpzM in sorted(os.listdir(pathFldM)):
            #
            # obtain fnNpzMWE and inform
            fnNpzMWE = os.path.splitext(fnNpzM)[0]
            print("{}{} Computing wavenumber field for mode {}".format(' '*2*dGit['ind'], dGit['sB2'], fnNpzMWE))
            #
            # obtain pathOut; here, to manage overwrite
            pathOutNpz = os.path.join(pathFldMain, 'scratch', 'numerics', video, 'K_wavenumbers', fnNpzM)  # IMP*: nomenclature; no trace of decomposition method
            if os.path.exists(pathOutNpz) and not par['overwrite_outputs']:
                print("\033[F\033[K{}{} Wavenumber field for mode {} was already available {}".format(' '*2*dGit['ind'], dGit['sB2'], fnNpzMWE, dGit['sOK']))
                continue
            #
            # read the mode file
            dataTMP = np.load(os.path.join(pathFldMain, 'scratch', 'numerics', video, 'M_modes', fnNpzM))
            w, T, phasesM = [dataTMP[item] for item in ['w', 'T', 'phases']]  # given in mesh_M
            #
            # obtain RKs
            hsAux = [par['min_depth']+(item+1)/par['n_neighborhoods_K']*(par['max_depth']-par['min_depth']) for item in range(par['n_neighborhoods_K'])][::-1]  # from large to small
            RKs = np.asarray([par['coeff_radius_K'] * uli.TH2LOneStep(T, hAux, grav) for hAux in hsAux])  # from large to small
            #
            # obtain wavenumbers kxs and kys for mesh_K
            kxs, kys = [np.full((len(xsK), len(RKs)), np.nan) for _ in range(2)]  # IMP*: np.nan
            for posK in range(len(xsK)):
                #
                # obtain distances of the points in mesh_M to posK
                dsToM = np.hypot(xsM - xsK[posK], ysM - ysK[posK])
                pos0M = np.argmin(dsToM)  # point in mesh_M closest to posK
                x0M, y0M, phase0M = xsM[pos0M], ysM[pos0M], phasesM[pos0M]
                #
                # analysis for RKs
                xsMTMP, ysMTMP, phasesMTMP, dsToMTMP = map(copy.deepcopy, [xsM, ysM, phasesM, dsToM])  # initialize auxiliar
                for posRK, RK in enumerate(RKs):  # IMP*, decreasing
                    # update xsMTMP, ysMTMP, phasesMTMP and dsToMTMP (RK decreases) and disregard
                    poss = np.where(dsToMTMP <= RK)[0]
                    xsMTMP, ysMTMP, phasesMTMP, dsToMTMP = [item[poss] for item in [xsMTMP, ysMTMP, phasesMTMP, dsToMTMP]]
                    if len(poss) < 6 or np.hypot(np.mean(xsMTMP) - xsK[posK], np.mean(ysMTMP) - ysK[posK]) > RK / 4:  # IMP*: OJO: do something similar in bathymetry?; be more demanding?
                        continue
                    # obtain xsDel, ysDel and phasesDel; centered
                    xsDel, ysDel, phasesDel = xsMTMP - x0M, ysMTMP - y0M, np.angle(np.exp(1j * (phasesMTMP - phase0M)))  # IMP*
                    # update kxs and kys
                    kx, ky, _, possG = uli.RANSACPlane(xsDel, ysDel, phasesDel, 0.25, pDesired=1-1.e-9, margin=0.2, max_nForRANSAC=100)  # WATCH OUT: epsilon
                    if len(possG) < 6:
                        continue
                    kxs[posK, posRK], kys[posK, posRK] = kx, ky
            #
            # inform
            print("\033[F\033[K{}{} Wavenumber field for mode {} computed {}".format(' '*2*dGit['ind'], dGit['sB2'], fnNpzMWE, dGit['sOK']))
            #
            # update kxs, kys and obtain ks and gammas; nans where gamma > 1.2; setting 1.0 or 2.0 instead of 1.2 is not so important
            ks = np.hypot(kxs, kys)
            for posRK in range(len(RKs)):
                possB = np.where((ks[:, posRK] < 1.e-6) | (1.2 * ks[:, posRK] < w ** 2 / grav))[0]  # bad positions, ks < is useful
                kxs[possB, posRK], kys[possB, posRK], ks[possB, posRK] = np.nan, np.nan, np.nan # IMP*
            gammas = w ** 2 / (grav * ks)  # = tanh(ks * h)
            #
            # obtain meanGs and stdGs; IMP*: nans where ks is nan and in more positions
            meanGs, stdGs = [np.full((len(xsK), len(RKs)), np.nan) for _ in range(2)]  # IMP*: np.nan
            for posK in range(len(xsK)):
                dsToKTMP = np.hypot(xsK - xsK[posK], ysK - ysK[posK])
                for posRK, RK in enumerate(RKs):
                    if np.isnan(ks[posK, posRK]):
                        continue  # initialized as np.nan
                    RG = max(0.5 * 2 * np.pi / ks[posK, posRK], 2.1 * par['delta_K'])  # IMP*
                    possG = np.where((~np.isnan(ks[:, posRK])) & (dsToKTMP <= RG))[0]  # good positions
                    if len(possG) < 3:
                        continue  # initialized as np.nan
                    meanGs[posK, posRK], stdGs[posK, posRK] = np.mean(gammas[possG, posRK]), np.std(gammas[possG, posRK])
            #
            # write pathOutNpz
            os.makedirs(os.path.dirname(pathOutNpz), exist_ok=True)
            np.savez(pathOutNpz, w=w, T=T, RKs=RKs, ks=ks, meanGs=meanGs, stdGs=stdGs)  # IMP*: WATCH OUT: meanGs and stdGs may have more np.nan than ks
            #
            # write pathScrJpg
            if par['generate_scratch_plots']:
                pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', video, 'K_wavenumbers', '{}.jpg'.format(fnNpzMWE))  # IMP*: nomenclature
                uli.GHBathyPlotWavenumbers(pathScrJpg, xsM, ysM, T, phasesM, xsK, ysK, RKs, ztsK, ks, gammas, stdGs, par['max_depth'], dGit['fs'], dGit['fs'], dGit['fontsize'], dpi=dGit['dpiLQ'])
    #
    return None
#
def ObtainB(pathFldMain):  # lm:2025-07-07; lr:2025-07-07
    #
    def GoalFunction220622(x, theArgs):
        gammas, ws, zts, g = [theArgs[key] for key in ['gammas', 'ws', 'zts', 'g']]
        gammasR = ws ** 2 / (g * uli.WGH2KOneStep(ws, g, np.clip(zts-x[0], 1.e-14, np.inf)))  # WATCH OUT: epsilon
        gf = uli.RMSE1D_2506(gammasR, gammas)
        return gf
    #
    # obtain par
    par = uli.GHBathyParAndVideos(pathFldMain)[0]  # IMP*: also checks videos
    #
    # load xsB and ysB
    pathTMPTxt = os.path.join(pathFldMain, 'output', 'numerics', 'mesh_B.txt')  # IMP*: nomenclature
    dataTMP = np.loadtxt(pathTMPTxt, usecols=range(2), dtype=float, ndmin=2)
    xsB, ysB = [dataTMP[:, item] for item in [0, 1]]
    #
    # run for dates
    for date12 in sorted(par['videos4dates']):
        #
        # obtain pathOutTxt and disregard
        pathOutTxt = os.path.join(pathFldMain, 'output', 'numerics', 'bathymetries', '{}.txt'.format(date12))  # IMP*: nomenclature        
        if os.path.exists(pathOutTxt) and not par['overwrite_outputs']:
            zbsB = np.loadtxt(pathOutTxt, usecols=range(2), dtype=float, ndmin=2)[:, 0]
            print("{}{} Bathymetry for date {} was already available: {} points {}".format(' '*dGit['ind'], dGit['sB1'], date12, np.sum(~np.isnan(zbsB)), dGit['sOK']))
            continue
        #
        # inform
        print("{}{} Computing bathymetry for date {}".format(' '*dGit['ind'], dGit['sB1'], date12))
        #
        # run through videos for the date to load sets of x, y, gamma, mean_gamma, w and zt
        xsKW, ysKW, gammasKW, meanGsKW, wsKW, ztsKW = [[] for _ in range(6)]  # IMP*: not to contain nan
        for video in par['videos4dates'][date12]:
            # load xsK, ysK and ztsK
            dataTMP = np.load(os.path.join(pathFldMain, 'scratch', 'numerics', video, 'mesh_K.npz'))
            xsK, ysK, ztsK = [dataTMP[item] for item in ['xs', 'ys', 'zts']]
            assert np.std(ztsK) < 1.e-3
            # obtain pathFldK and fnsNpzs
            pathFldK = os.path.join(pathFldMain, 'scratch', 'numerics', video, 'K_wavenumbers')
            if not os.path.exists(pathFldK) or len(os.listdir(pathFldK)) == 0:
                continue
            fnsNpzs = sorted(os.listdir(pathFldK))
            # run through fnsNpzs; modes
            for fnNpz in fnsNpzs:
                # load w, ks, meanGs, stdGs, RKs; ks, meanGs and stdGs are len(xsK) x len(RKs)
                dataTMP = np.load(os.path.join(pathFldK, fnNpz))
                w, ks, meanGs, stdGs, RKs = [dataTMP[item] for item in ['w', 'ks', 'meanGs', 'stdGs', 'RKs']]
                # run through RKs
                for posRK in range(len(RKs)):
                    # obtain possG
                    gs0 = stdGs[:, posRK]  # can contain nans
                    gs1 = np.abs(w ** 2 / (grav * ks[:, posRK]) - meanGs[:, posRK])  # can contain nans
                    possG = np.where(~np.isnan(gs0) & ~np.isnan(gs1) & (gs0 <= par['gamma_std_cutoff']) & (gs1 <= par['gamma_std_cutoff']))[0]  # does not contain nan
                    # update xsKW, ysKW, gammasKW, meanGsKW, wsKW and ztsKW
                    xsKW.extend(xsK[possG]); ysKW.extend(ysK[possG]); gammasKW.extend(w**2/(grav*ks[possG, posRK])); meanGsKW.extend(meanGs[possG, posRK]); wsKW.extend([w]*len(possG)); ztsKW.extend([np.mean(ztsK)]*len(possG))
        xsKW, ysKW, gammasKW, meanGsKW, wsKW, ztsKW = map(np.asarray, [xsKW, ysKW, gammasKW, meanGsKW, wsKW, ztsKW])
        #
        # inform and disregard
        if len(xsKW) == 0:
            print("\033[F\033[K{}{} No bathymetry was obtained for date {} {}".format(' '*dGit['ind'], dGit['sB1'], date12, dGit['sWO']))
            continue
        #
        # obtain zbsB and auxiliar RBs
        zbsB, RBs = [np.full(len(xsB), np.nan) for _ in range(2)]  # IMP*: np.nan
        for posB in range(len(xsB)):
            # obtain dsToKWTMP
            dsToKWTMP = np.hypot(xsKW - xsB[posB], ysKW - ysB[posB])
            # obtain L0; a characteristic wavelength
            poss0 = np.where(dsToKWTMP <= 1.1 * np.min(dsToKWTMP) + 1.e-3)[0]
            L0 = np.mean(2 * np.pi * grav * meanGsKW[poss0] / wsKW[poss0] ** 2)
            # obtain RB; a characteristic radius for search
            RB = max([par['coeff_radius_B'] * L0, 2.1 * par['delta_K']])  # IMP*
            # obtain possInKW, to compute the bathymetry, and disregard
            possInKW = np.where(dsToKWTMP <= RB)[0]
            if len(possInKW) < 10:  # WATCH OUT: epsilon
                continue  # initialized as np.nan
            # obtain gammasA, wsA and ztsA; A = Around
            gammasA, wsA, ztsA = [item[possInKW] for item in [gammasKW, wsKW, ztsKW]]
            # obtain possG and zb and disregard; first approximation through a RANSAC-like approach
            possG, zb = [], np.nan
            for zbH in np.arange(np.min(ztsA)-par['max_depth'], np.max(ztsA)-par['min_depth'], 0.05): # IMP*: 0.05
                gammasH = wsA ** 2 / (grav * uli.WGH2KOneStep(wsA, grav, np.clip(ztsA - zbH, par['min_depth'], np.inf)))  # IMP*: np.clip
                gammasH[ztsA - zbH <= par['min_depth']] = np.inf  # IMP*
                possGH = np.where(np.abs(gammasH - gammasA) < par['gamma_std_cutoff'])[0]  # IMP*: gamma_std_cutoff
                if len(possGH) > len(possG):
                    possG, zb = possGH, zbH
            if len(possG) < 10:  # WATCH OUT: epsilon
                continue  # initialized as np.nan
            # obtain zb through the minimization for the good positions
            theArgs = {'gammas': gammasA[possG], 'ws': wsA[possG], 'zts': ztsA[possG], 'g': grav}
            zb = optimize.minimize(GoalFunction220622, np.asarray([zb]), args=(theArgs)).x[0]  # IMP*
            # update zbsB and RBs
            zbsB[posB], RBs[posB] = zb, RB
        #
        # obtain ezbsB
        ezbsB = np.full(len(xsB), np.nan)  # IMP*
        for posB in range(len(xsB)):
            if np.isnan(zbsB[posB]) or np.isnan(RBs[posB]):
                continue  # initialized as np.nan
            dsToBTMP = np.hypot(xsB - xsB[posB], ysB - ysB[posB])
            possTMP = np.where(dsToBTMP <= RBs[posB])[0]
            if len(possTMP) < 3:
                continue  # initialized as np.nan
            ezbsB[posB] = np.std(zbsB[possTMP])  # IMP*: WATCH OUT: not np.nanstd; if any zbsB is nan, ezbsB is nan
        #
        # update zbsB and inform
        zbsB[np.isnan(ezbsB)] = np.nan  # IMP*
        print("\033[F\033[K{}{} Bathymetry for date {} computed: {} points {}".format(' '*dGit['ind'], dGit['sB1'], date12, np.sum(~np.isnan(zbsB)), dGit['sOK']))
        #
        # write zbsB and ezbsB in txt
        os.makedirs(os.path.dirname(pathOutTxt), exist_ok=True)
        with open(pathOutTxt, 'w') as fileout:
            for posB in range(len(xsB)):
                fileout.write('{:9.4f} {:9.4f}\n'.format(zbsB[posB], ezbsB[posB]))  # IMP*: formatting
        #
        # obtain thereIsGTB
        pathGTBTxt = os.path.join(pathFldMain, 'data', 'ground_truth', '{}_GT_xyz.txt'.format(date12))  # IMP*: nomenclature
        thereIsGTB = os.path.exists(pathGTBTxt)
        #
        # obtain zbsGTB; ground truth bathymetry interpolated to mesh_B
        if thereIsGTB:
            # obtain zbsGTB
            dataTMP = np.loadtxt(pathGTBTxt, usecols=range(3), dtype=float, ndmin=2)
            xsGT, ysGT, zbsGT = [dataTMP[:, item] for item in range(3)]  # IMP*: nomenclature
            zbsGTB = griddata((xsGT, ysGT), zbsGT, (xsB, ysB), method='linear')  # IMP*: np.nan outside the convex hull
            # write zbsGTB in txt
            pathTMPTxt = os.path.join(pathFldMain, 'output', 'numerics', 'bathymetries', '{}_GT.txt'.format(date12))  # IMP*: nomenclature
            os.makedirs(os.path.dirname(pathTMPTxt), exist_ok=True)
            with open(pathTMPTxt, 'w') as fileout:
                for posB in range(len(xsB)):
                    fileout.write('{:9.4f}\n'.format(zbsGTB[posB]))  # IMP*: formatting
        else:
            zbsGTB = None
        #
        # write pathTMPPngpathImg
        pathTMPPng = os.path.join(pathFldMain, 'output', 'plots', 'bathymetries', '{}.png'.format(date12))  # IMP*: nomenclature
        uli.GHBathyPlotBath(pathTMPPng, date12, xsB, ysB, zbsB, ezbsB, par['min_depth'], par['max_depth'], dGit['fw'], dGit['fh'], dGit['fontsize'], dpi=dGit['dpiLQ'], thereIsGT=thereIsGTB, zbsGT=zbsGTB)
    # 
    return None
#
def PerformKalman(pathFldMain): # lm:2025-07-08; lr:2025-07-09
    #
    # obtain par
    par = uli.GHBathyParAndVideos(pathFldMain)[0]  # IMP*: also checks videos; par = uli.GHLoadPar(pathFldMain)
    #
    # load xsB and ysB
    pathTMPTxt = os.path.join(pathFldMain, 'output', 'numerics', 'mesh_B.txt')  # IMP*: nomenclature
    dataTMP = np.loadtxt(pathTMPTxt, usecols=range(2), dtype=float, ndmin=2)
    xsB, ysB = [dataTMP[:, item] for item in range(2)]
    #
    # obtain pathFldOBat and fnsTxts
    pathFldOBat = os.path.join(pathFldMain, 'output', 'numerics', 'bathymetries')
    if not os.path.exists(pathFldOBat) or len(os.listdir(pathFldOBat)) == 0:
        print("{}{} No bathymetries available to filter {}".format(' '*dGit['ind'], dGit['sB1'], dGit['sWO']))
        return None
    fnsTxts = sorted([item for item in os.listdir(pathFldOBat) if len(item) == 16 and item.endswith('.txt') and int(par['Kalman_start']) <= int(item[:12]) <= int(par['Kalman_end'])])
    #
    # disregard
    if all(os.path.exists(os.path.join(pathFldMain, 'output', 'numerics', 'bathymetries_Kalman', item)) for item in fnsTxts):
        print("{}{} All bathymetries were already filtered: {} Kalman bathymetries found {}".format(' '*dGit['ind'], dGit['sB1'], len(fnsTxts), dGit['sOK']))
        return None
    #
    # obtain Kalman-filtered bathymetries
    PttsB, zbsB, tsB = [np.full(len(xsB), np.nan) for _ in range(3)]
    for fnTxt in fnsTxts:
        #
        # obtain fnTxtWE, date12 and datenumH and inform
        fnTxtWE, date12 = os.path.splitext(fnTxt)[0], os.path.splitext(fnTxt)[0]
        datenumH = uli.Date2Datenum_2504(date12+'0'*5)
        print("{}{} Computing Kalman filtered bathymetry for date {}".format(' '*dGit['ind'], dGit['sB1'], fnTxtWE))
        #
        # load bathymetry for fnTxt
        dataTMP = np.loadtxt(os.path.join(pathFldOBat, fnTxt))
        zbsBH, ezbsBH = [dataTMP[:, item] for item in range(2)]
        #
        # update Kalman bathymetry
        for posB in range(len(xsB)):
            if np.isnan(zbsB[posB]):
                PttsB[posB] = 0.
                zbsB[posB] = zbsBH[posB]
            else:
                if np.isnan(zbsBH[posB]) or np.isnan(ezbsBH[posB]): 
                    continue
                Ptt1 = PttsB[posB] + (par['variance_per_day'] * (datenumH - tsB[posB])) ** 2  # p dimension [L^2]
                Kt = Ptt1 / (Ptt1 + ezbsBH[posB] ** 2)  # dimension [-]
                PttsB[posB] = (1 - Kt) * Ptt1  # dimension [L^2]
                zbsB[posB] = zbsB[posB] + Kt * (zbsBH[posB] - zbsB[posB])
            tsB[posB] = datenumH   # works if at least one point is updated
        #
        # write zbsB and np.sqrt(PttsB) in txt
        pathTMPTxt = os.path.join(pathFldMain, 'output', 'numerics', 'bathymetries_Kalman', '{}.txt'.format(fnTxtWE))  # IMP*: nomenclature
        os.makedirs(os.path.dirname(pathTMPTxt), exist_ok=True)
        with open(pathTMPTxt, 'w') as fileout:
            for posB in range(len(xsB)):
                fileout.write('{:9.4f} {:9.4f}\n'.format(zbsB[posB], np.sqrt(PttsB[posB])))  # IMP*: formatting
        #
        # obtain thereIsGTB
        pathGTBTxt = os.path.join(pathFldMain, 'data', 'ground_truth', '{}_GT_xyz.txt'.format(fnTxtWE))  # IMP*: nomenclature
        thereIsGTB = os.path.exists(pathGTBTxt)
        #
        # obtain and write zbsGTB; ground truth bathymetry interpolated to mesh_B
        if thereIsGTB:
            # obtain zbsGTB
            dataTMP = np.loadtxt(pathGTBTxt, usecols=range(3), dtype=float, ndmin=2)
            xsGT, ysGT, zbsGT = [dataTMP[:, item] for item in range(3)]  # IMP*: nomenclature
            zbsGTB = griddata((xsGT, ysGT), zbsGT, (xsB, ysB), method='linear')  # IMP*: np.nan outside the convex hull; WATCH OUT: linear
            # write zbsGTB in txt
            pathTMPTxt = os.path.join(pathFldMain, 'output', 'numerics', 'bathymetries_Kalman', '{}_GT.txt'.format(fnTxtWE))  # IMP*: nomenclature
            os.makedirs(os.path.dirname(pathTMPTxt), exist_ok=True)
            with open(pathTMPTxt, 'w') as fileout:
                for posB in range(len(xsB)):
                    fileout.write('{:9.4f}\n'.format(zbsGTB[posB]))  # IMP*: formatting
        else:
            zbsGTB = None
        #
        # write pathTMPPng
        pathTMPPng = os.path.join(pathFldMain, 'output', 'plots', 'bathymetries_Kalman', '{}.png'.format(fnTxtWE))  # IMP*: nomenclature
        uli.GHBathyPlotBath(pathTMPPng, date12, xsB, ysB, zbsB, np.sqrt(PttsB), par['min_depth'], par['max_depth'], dGit['fw'], dGit['fh'], dGit['fontsize'], dpi=dGit['dpiLQ'], thereIsGT=thereIsGTB, zbsGT=zbsGTB, title2=r'$\sigma$-estimated [m]')
        #
        # inform
        nOfPoints = np.sum(~np.isnan(np.loadtxt(os.path.join(pathFldMain, 'output', 'numerics', 'bathymetries_Kalman', '{}.txt'.format(fnTxtWE)), usecols=range(2), dtype=float, ndmin=2)[:, 0]))
        print("\033[F\033[K{}{} Kalman filtered bathymetry for date {} computed: {} points {}".format(' '*dGit['ind'], dGit['sB1'], fnTxtWE, nOfPoints, dGit['sOK']))
        #
    #
    return None
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
