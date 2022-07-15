#'''
# Created on 2021 by Gonzalo Simarro and Daniel Calvete
#'''
#
import os
import ulises_ubathy as ulises
#
import copy
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy as sc
from scipy import optimize
from scipy import signal
import sys
#
def Video2Frames(pathFolderVideos, listOfVideos, fps, overwrite): # last read 2022-07-10
    #
    # obtain video filenames videoFns
    exts = ['mp4', 'MP4', 'avi', 'AVI', 'mov', 'MOV']
    videoFns = sorted([item for item in os.listdir(pathFolderVideos) if os.path.splitext(item)[1][1:] in exts])
    if len(listOfVideos) > 0: # (video in listOfVideos does not have extension; videoFn in videoFns does have)
        videoFns = [item for item in videoFns if os.path.splitext(item)[0] in listOfVideos] # intersection
    if len(videoFns) == 0:
        print('... no video-files to extract frames from')
    #
    # obtain frames from videos
    for videoFn in videoFns:
        #
        pathVideo = os.path.join(pathFolderVideos, videoFn)
        #
        # obtain and check pathFolderSnaps
        video = os.path.splitext(videoFn)[0]
        pathFolderSnaps = os.path.join(pathFolderVideos, video)
        if os.path.isdir(pathFolderSnaps) and not overwrite:
           print('... frame extraction for video {:} was already performed (the folder exists)'.format(video))
           continue
        #
        # load video and obtain fps
        fpsOfVideo = cv2.VideoCapture(pathVideo).get(cv2.CAP_PROP_FPS)
        if fps > fpsOfVideo:
            print('*** actual frames per second of the video ({:3.2f}) smaller than pretended ({:3.2f})'.format(fpsOfVideo, fps))
            print('*** select a smaller fps ***'); sys.exit()
        elif fps == 0:
            fps = fpsOfVideo
        else: # 0 < fps <= fpsOfVideo so that fpsOfVideo / fps >= 1 and also int(fpsOfVideo / fps) >= 1
            fps = fpsOfVideo / int(fpsOfVideo / fps) # to ensure that fpsOfVideo is multiple of fps (Video2Snaps checks anyway)
        #
        # write frames
        print('... frame extraction of video {:} from {:} at {:3.2f} fps'.format(video, videoFn, fps))
        ulises.Video2Snaps(pathVideo, pathFolderSnaps, fps, options={})
        #
    return None
#
def CreateMeshes(pathFolderData, pathFolderVideos, pathFolderScratch, listOfVideos, overwrite, verbosePlot): # last read 2022-07-10
    #
    # load parameters
    pathTMP = os.path.join(pathFolderData, 'parameters.json')
    with open(pathTMP, 'r') as f:
        par = json.load(f)
    #
    # load boundary of the domain where to obtain the bathymetry
    pathTMP = os.path.join(pathFolderData, 'xy_boundary.txt')
    rawData = np.asarray(ulises.ReadRectangleFromTxt(pathTMP, options={'c1':2, 'valueType':'float'}))
    xsB, ysB = rawData[:, 0], rawData[:, 1]
    #
    # create, write and load mesh_Zb (depends exclusively on xy_boundary)
    pathTMP = os.path.join(pathFolderScratch, 'mesh_Zb.npz')
    if not os.path.exists(pathTMP) or overwrite:
        #
        print('... creating mesh_Zb')
        #
        # create mesh_Zb
        xsZb, ysZb = ulises.Polyline2InnerHexagonalMesh({'xs':xsB, 'ys':ysB}, par['delta_Zb'], options={})
        #
        # write mesh_Zb
        ulises.MakeFolder(os.path.split(pathTMP)[0])
        np.savez(pathTMP, xs=xsZb, ys=ysZb)
        #
        # plot mesh_Zb
        if verbosePlot:
            pathTMPPlot = os.path.join(pathFolderScratch, 'plots', 'mesh_Zb.png')
            ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
            plt.plot(list(xsB) + [xsB[0]], list(ysB) + [ysB[0]], 'r-', lw=5)
            plt.plot(xsZb, ysZb, 'k.', markersize=0.4)
            plt.xlabel('x [m]'); plt.ylabel('y [m]'); plt.axis('equal')
            plt.savefig(pathTMPPlot, dpi=100); plt.close()
    else:
        print('... mesh_Zb was already created')
    #
    # obtain listOfVideos
    listOfVideosH = sorted([item for item in os.listdir(pathFolderData) if os.path.isdir(os.path.join(pathFolderData, item)) and item != 'groundTruth'])
    if listOfVideos == []:
        listOfVideos = copy.deepcopy(listOfVideosH)
    else:
        listOfVideos = [item for item in listOfVideosH if item in listOfVideos]
    #
    # create and write mesh_M and mesh_K for each video (depends on xy_boundary but also on the video)
    for video in listOfVideos:
        #
        # obtain paths and identify if it is a video of planviews or made up of oblique images
        pathCalTxts = [os.path.join(pathFolderData, video, item) for item in os.listdir(os.path.join(pathFolderData, video)) if item.endswith('cal.txt')]
        pathZsTxts = [os.path.join(pathFolderData, video, item) for item in os.listdir(os.path.join(pathFolderData, video)) if item.endswith('zs.txt')]
        pathPlwTxts = [os.path.join(pathFolderData, video, item) for item in os.listdir(os.path.join(pathFolderData, video)) if item.endswith('crxyz.txt')]
        if len(pathCalTxts) > 0 and len(pathZsTxts) > 0:
            isPlw, pathCalTxt, pathZsTxt = False, pathCalTxts[0], pathZsTxts[0]
        elif len(pathPlwTxts) > 0:
            isPlw, pathPlwTxt = True, pathPlwTxts[0]
        else:
            print('*** missing calibration, free surface or planview definition files in folder {:}'.format(os.path.join(pathFolderData, video))); sys.exit()
        #
        # create and write mesh_M
        pathTMP = os.path.join(pathFolderScratch, video, 'mesh_M.npz')
        if not os.path.exists(pathTMP) or overwrite:
            print('... creating mesh_M for video {:}'.format(video))
            # create initial mesh_M
            if isPlw: # pixels correspond to the planview and are integers
                rawData = np.asarray(ulises.ReadRectangleFromTxt(pathPlwTxt, {'c1':5, 'valueType':'float'}))
                csM, rsM, xsM, ysM, zsM = [rawData[:, item] for item in range(5)]
                if not (np.allclose(csM, csM.astype(int)) and np.allclose(rsM, rsM.astype(int))):
                    print('*** pixel coordinates must be integers >= 0 at {:}'.format(pathPlwTxt)); sys.exit()
            else: # pixels correspond to the oblique image nad are integers
                # load zt
                zt = ulises.ReadRectangleFromTxt(pathZsTxt, {'c1':1, 'valueType':'float'})[0]
                # obtain mesh in xy
                xsM, ysM = ulises.Polyline2InnerHexagonalMesh({'xs':xsB, 'ys':ysB}, par['delta_M'], options={})
                # obtain mesh in cr
                mainSet = ulises.ReadMainSetFromCalTxt(pathCalTxt, options={})
                csM, rsM, possGood = ulises.XYZ2CDRD(mainSet, xsM, ysM, zt*np.ones(len(xsM)), options={'returnGoodPositions':True})
                csM, rsM = [np.round(item[possGood]).astype(int) for item in [csM, rsM]] # IMP*
                # retain only unique points in cr
                auxs = list(set(zip(csM, rsM))) # unique, interesting
                csM, rsM = np.asarray([item[0] for item in auxs]), np.asarray([item[1] for item in auxs])
                # obtain mesh in xy from cr
                xsM, ysM, possGood = ulises.CDRDZ2XY(mainSet, csM, rsM, zt*np.ones(len(csM)), options={'returnGoodPositions':True})
                csM, rsM, xsM, ysM = [item[possGood] for item in [csM, rsM, xsM, ysM]]
                # sort mesh in xy using the distance to (x0, y0)
                xC, yC = np.mean(xsM), np.mean(ysM)
                dC = np.sqrt((xC - mainSet['xc']) ** 2 + (yC - mainSet['yc']) ** 2)
                x0, y0 = xC + 5000 * (xC - mainSet['xc']) / dC, yC + 5000 * (yC - mainSet['yc']) / dC # IMP*
                possSorted = np.argsort((xsM - x0) ** 2 + (ysM - y0) ** 2)
                csM, rsM, xsM, ysM = [item[possSorted] for item in [csM, rsM, xsM, ysM]]
                # obtain uniformized mesh in crxyz
                csMu, rsMu, xsMu, ysMu = [np.asarray([item[0]]) for item in [csM, rsM, xsM, ysM]] # initialize with the first point
                for pos in range(1, len(csM)): # potential point to the final mesh
                    cM, rM, xM, yM = [item[pos] for item in [csM, rsM, xsM, ysM]]
                    if np.min((xM - xsMu) ** 2 + (yM - ysMu) ** 2) >= (0.95 * par['delta_M']) ** 2:
                        csMu, rsMu, xsMu, ysMu = np.append(csMu, cM), np.append(rsMu, rM), np.append(xsMu, xM), np.append(ysMu, yM)
                csM, rsM, xsM, ysM, zsM = csMu, rsMu, xsMu, ysMu, zt*np.ones(len(csMu))
            # retain only the points within the bathymetry boundary (useful, indeed, for isPlw)
            possIn = ulises.CloudOfPoints2InnerPositionsForPolyline(xsM, ysM, {'xs':xsB, 'ys':ysB}, options={})
            csM, rsM, xsM, ysM, zsM = [item[possIn] for item in [csM, rsM, xsM, ysM, zsM]]
            # write mesh_M
            ulises.MakeFolder(os.path.split(pathTMP)[0])
            np.savez(pathTMP, cs=csM, rs=rsM, xs=xsM, ys=ysM, zs=zsM)
            # plot mesh_M
            if verbosePlot:
                pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video, 'mesh_M.png')
                ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
                plt.plot(list(xsB) + [xsB[0]], list(ysB) + [ysB[0]], 'r-', lw=5)
                plt.plot(xsM, ysM, 'k.', markersize=0.4)
                plt.xlabel('x [m]'); plt.ylabel('y [m]'); plt.axis('equal')
                plt.savefig(pathTMPPlot, dpi=100); plt.close()
                #
                pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video, 'mesh_M_inImage.png')
                pathImgs = [os.path.join(pathFolderVideos, video, item) for item in os.listdir(os.path.join(pathFolderVideos, video)) if item[-3:] in ['png', 'jpg']]
                if len(pathImgs) > 0:
                    img = cv2.imread(pathImgs[0])
                    img = ulises.DisplayCRInImage(img, csM, rsM, options={'colors':[[0, 0, 255]], 'size':1.5})
                    ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
                    cv2.imwrite(pathTMPPlot, img)
                else:
                    print('... no images in folder {:} to plot {:}'.format(os.path.join(pathFolderVideos, video), pathTMPPlot))
        else:
            print('... mesh_M for video {:} was already created'.format(video))
        #
        # create and write mesh_K
        pathTMP = os.path.join(pathFolderScratch, video, 'mesh_K.npz')
        if not os.path.exists(pathTMP) or overwrite:
            print('... creating mesh_K for video {:}'.format(video))
            # create initial mesh_K
            xsK, ysK = ulises.Polyline2InnerHexagonalMesh({'xs':xsB, 'ys':ysB}, par['delta_K'], options={})
            if isPlw: # pixels correspond to the planview
                rawData = np.asarray(ulises.ReadRectangleFromTxt(pathPlwTxt, {'c1':5, 'valueType':'float'}))
                csP, rsP, xsP, ysP, zsP = [rawData[:, item] for item in range(5)]; assert np.std(zsP) < 1.e-3
                nc, nr = [int(np.round(item.max())) + 1 for item in [csP, rsP]]
                possC = [np.where((csP == item[0]) & (rsP == item[1]))[0][0] for item in [[0, 0], [nc-1, 0], [nc-1, nr-1], [0, nr-1]]]
                possIn = ulises.CloudOfPoints2InnerPositionsForPolyline(xsK, ysK, {'xs':xsP[possC], 'ys':ysP[possC]}, options={}) # points in K in plw
                xsK, ysK = [item[possIn] for item in [xsK, ysK]]
                csK, rsK = ulises.ApplyAffineA01(ulises.FindAffineA01(xsP, ysP, csP, rsP), xsK, ysK) # real valued (only for TMPPlot)
                zsK = np.mean(zsP) * np.ones(len(xsK)) # IMP*
            else: # pixels correspond to the oblique image
                zt = ulises.ReadRectangleFromTxt(pathZsTxt, {'c1':1, 'valueType':'float'})[0]
                mainSet = ulises.ReadMainSetFromCalTxt(pathCalTxt, options={})
                csK, rsK, possGood = ulises.XYZ2CDRD(mainSet, xsK, ysK, zt*np.ones(len(xsK)), options={'returnGoodPositions':True})
                csK, rsK, xsK, ysK = [item[possGood] for item in [csK, rsK, xsK, ysK]]
                zsK = zt * np.ones(len(xsK))
            # write mesh_K
            ulises.MakeFolder(os.path.split(pathTMP)[0])
            np.savez(pathTMP, xs=xsK, ys=ysK, zs=zsK)
            # plot mesh_K
            if verbosePlot:
                pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video, 'mesh_K.png')
                ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
                plt.plot(list(xsB) + [xsB[0]], list(ysB) + [ysB[0]], 'r-', lw=5)
                plt.plot(xsK, ysK, 'k.', markersize=0.4)
                plt.xlabel('x [m]'); plt.ylabel('y [m]'); plt.axis('equal')
                plt.savefig(pathTMPPlot, dpi=100); plt.close()
                #
                pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video,'mesh_K_inImage.png')
                pathImgs = [os.path.join(pathFolderVideos, video, item) for item in os.listdir(os.path.join(pathFolderVideos, video)) if item[-3:] in ['png', 'jpg']]
                if len(pathImgs) > 0:
                    img = cv2.imread(pathImgs[0])
                    img = ulises.DisplayCRInImage(img, csK, rsK, options={'colors':[[0, 0, 255]], 'size':1.5})
                    ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
                    cv2.imwrite(pathTMPPlot, img)
                else:
                    print('... no images in folder {:} to plot {:}'.format(os.path.join(pathFolderVideos, video), pathTMPPlot))
        else:
            print('... mesh_K for video {:} was already created'.format(video))
    #
    return None
#
def ObtainWAndModes(pathFolderData, pathFolderVideos, pathFolderScratch, listOfVideos, overwrite, verbosePlot): # last read 2022-07-10
    #
    # load parameters
    pathTMP = os.path.join(pathFolderData, 'parameters.json')
    with open(pathTMP, 'r') as f:
        par = json.load(f)
    #
    # obtain list of videos
    listOfVideosH = sorted([item for item in os.listdir(pathFolderData) if os.path.isdir(os.path.join(pathFolderData, item)) and item != 'groundTruth'])
    if listOfVideos == []:
        listOfVideos = copy.deepcopy(listOfVideosH)
    else:
        listOfVideos = [item for item in listOfVideosH if item in listOfVideos]
    #
    # run for videos
    for video in listOfVideos:
        #
        # manage overwrite
        toContinue = False
        if os.path.exists(os.path.join(pathFolderScratch, video, 'M_modes')):
            fns = [item for item in os.listdir(os.path.join(pathFolderScratch, video, 'M_modes')) if par['DMD_or_EOF']+'.' in item]
            if len(fns) > 0 and overwrite:
                for fn in fns:
                    os.remove(os.path.join(pathFolderScratch, video, 'M_modes', fn))
                for fn in [item for item in os.listdir(os.path.join(pathFolderScratch, 'plots', video, 'M_modes')) if par['DMD_or_EOF']+'.' in item]:
                    os.remove(os.path.join(pathFolderScratch, 'plots', video, 'M_modes', fn))
            elif len(fns) > 0 and not overwrite:
                toContinue = True
        if toContinue:
            print('... modes {:} for video {:} were already computed'.format(par['DMD_or_EOF'], video))
            continue
        #
        # load fnsVideo (sorted)
        fnsVideo = sorted([item for item in os.listdir(os.path.join(pathFolderVideos, video)) if item[-3:] in ['png', 'jpg']])
        #
        # load ts in seconds
        ts = np.asarray([int(fn.split('_')[-1].split('.')[0])/1000. for fn in fnsVideo]) # IMP* written in miliseconds
        if len(ts) < 2:
            continue
        ts, dtsMean = ts - ts[0], np.mean(np.diff(ts))
        ts = dtsMean * np.arange(len(ts)) # IMP*
        if par['DMD_or_EOF'] == 'EOF' and par['min_period'] < 6 * dtsMean:
            print('*** working with EOF, "min_period" in json file must be >= 6 / fps (>= {:5.2f})'.format(6*dtsMean)); sys.exit()
        #
        # load cs and rs for mesh_M (coordinates in the image, whether planview or oblique)
        data = np.load(os.path.join(pathFolderScratch, video, 'mesh_M.npz'))
        csM, rsM, xsM, ysM = data['cs'].astype(int), data['rs'].astype(int), data['xs'], data['ys']
        #
        # load the video corresponding to mesh_M
        print('... loading video {:}'.format(video))
        XAll = np.zeros((len(csM), len(ts)))
        for posT in range(len(ts)):
            img = cv2.imread(os.path.join(pathFolderVideos, video, fnsVideo[posT]))
            XAll[:, posT] = img[rsM, csM, 1] / 255. # load green channel for the pixels corresponding to mesh_M
        #
        # run through the video
        wtMax, dtHilbert, nHilbert = np.max(par['time_windows']), par['max_period'], int(par['max_period']/dtsMean)+1
        for posT in range(nHilbert, len(ts), max([1, int(par['time_step']/dtsMean)])): # ts are already sorted
            t = ts[posT]
            if not (t-dtHilbert>=0 and t+wtMax+dtHilbert<=np.max(ts)):
                continue
            #
            # read lines for Hilbert-extended t-ball
            possTBallEx = sorted(np.where((ts>=t-dtHilbert) & (ts<=t+wtMax+dtHilbert))[0])
            XForTBallEx = XAll[:, possTBallEx]
            #
            # perform Candes (if robust) and Hilbert
            if par['candes_iter'] > 0:
                XForTBallEx = ulises.X2LSByCandes(XForTBallEx, options={'max_iter':par['candes_iter'], 'mu':1.})[0] # IMP*
            XForTBallExHilbert = signal.hilbert(XForTBallEx)
            #
            # crop the Hilbert extensions
            XForTBallHilbert = XForTBallExHilbert[:, nHilbert:-nHilbert]
            #
            # perform DMD/EOF for all parameters['time_windows']
            for wt in par['time_windows']:
                print('... {:} decomposition for t = {:5.1f} and wt = {:5.1f}'.format(par['DMD_or_EOF'], t, wt))
                fnOut0 = 't{:}_wt{:}'.format(str('{:3.2f}'.format(t)).zfill(8), str('{:3.2f}'.format(wt)).zfill(8))
                #
                # obtain positions for wt
                possWt = range(0, np.min([int(wt/dtsMean), XForTBallHilbert.shape[1]]))
                XForTBallHilbertWt = XForTBallHilbert[:, possWt]
                #
                # perform DMD/EOF for time windows
                modes = {}
                if par['DMD_or_EOF'] == 'DMD':
                    try:
                        Phi, Lambda = ulises.X2DMD(XForTBallHilbertWt, par['DMD_rank'])
                    except:
                        continue
                    for posDMD in range(len(Lambda)):
                        w = np.imag(np.log(Lambda[posDMD]) / dtsMean)
                        T = 2. * np.pi / w
                        if not (par['min_period'] <= T <= par['max_period']):
                            continue
                        modes[posDMD] = {'w':w, 'T':T, 'phases':np.angle(Phi[:, posDMD]), 'amplitudes':np.abs(Phi[:, posDMD])}
                elif par['DMD_or_EOF'] == 'EOF':
                    try:
                        EOF = ulises.X2EOF(XForTBallHilbertWt)
                    except:
                        continue
                    for posEOF in range(len(EOF['explainedVariances'])):
                        var = np.real(EOF['explainedVariances'][posEOF])
                        if var < par['EOF_variance']:
                            continue
                        w, wStd = ulises.GetWPhaseFitting(dtsMean*np.arange(len(possWt)), EOF['amplitudesForEOFs'][posEOF], 3*dtsMean, options={}) # IMP* x3
                        if wStd / w > 0.15: # IMP* 0.15
                            continue
                        T = 2. * np.pi / w
                        if not (par['min_period'] <= T <= par['max_period']):
                            continue
                        modes[posEOF] = {'var':var, 'wStdOverW':wStd/w, 'w':w, 'T':T, 'phases':np.angle(EOF['EOFs'][posEOF]), 'amplitudes':np.abs(EOF['EOFs'][posEOF])}
                else:
                    print('*** "DMD_or_EOF" in json file must be "DMD" or "EOF"'); sys.exit()
                #
                # write and plot the results
                for key in modes.keys():
                    #
                    # write the results
                    fnOut = fnOut0 + '_T{:}_{:}.npz'.format(str('{:3.2f}'.format(modes[key]['T'])).zfill(8), par['DMD_or_EOF'])
                    pathOut = os.path.join(pathFolderScratch, video, 'M_modes', fnOut)
                    ulises.MakeFolder(os.path.split(pathOut)[0])
                    if par['DMD_or_EOF'] == 'DMD':
                        np.savez(pathOut, w=modes[key]['w'], T=modes[key]['T'], phases=modes[key]['phases'], amplitudes=modes[key]['amplitudes'])
                    elif par['DMD_or_EOF'] == 'EOF':
                        np.savez(pathOut, w=modes[key]['w'], T=modes[key]['T'], phases=modes[key]['phases'], amplitudes=modes[key]['amplitudes'], var=modes[key]['var'], wStdOverW=modes[key]['wStdOverW'])
                    #
                    # plot the results
                    if verbosePlot:
                        pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video, 'M_modes', os.path.splitext(fnOut)[0]+'.png')
                        ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
                        #
                        fig = plt.figure(figsize=[6, 4]) # width x height
                        plt.suptitle('T = {:3.2f} s'.format(modes[key]['T']))
                        #
                        plt.subplot(1, 2, 1)
                        plt.title('phase [rad]')
                        sct = plt.scatter(xsM, ysM, marker='o', c=modes[key]['phases'], edgecolor='none', s=5, cmap='jet')
                        plt.colorbar(sct); plt.axis('equal'); plt.axis('off')
                        #
                        plt.subplot(1, 2, 2)
                        plt.title('amplitude [-]')
                        sct = plt.scatter(xsM, ysM, marker='o', c=modes[key]['amplitudes'], edgecolor='none', s=5, cmap='jet')
                        plt.colorbar(sct); plt.axis('equal'); plt.axis('off')
                        #
                        plt.tight_layout()
                        plt.savefig(pathTMPPlot, dpi=100)
                        plt.close('all')
    #
    return None
#
def ObtainK(pathFolderData, pathFolderScratch, listOfVideos, overwrite, verbosePlot): # last read 2022-07-10
    #
    # load parameters
    pathTMP = os.path.join(pathFolderData, 'parameters.json')
    with open(pathTMP, 'r') as f:
        par = json.load(f)
    #
    # obtain list of videos
    listOfVideosH = sorted([item for item in os.listdir(pathFolderData) if os.path.isdir(os.path.join(pathFolderData, item)) and item != 'groundTruth'])
    if listOfVideos == []:
        listOfVideos = copy.deepcopy(listOfVideosH)
    else:
        listOfVideos = [item for item in listOfVideosH if item in listOfVideos]
    #
    # run for videos
    for video in listOfVideos: # random.sample(listOfVideos, len(listOfVideos)): # WATCH OUT!!!
        #
        # load mesh_M
        data = np.load(os.path.join(pathFolderScratch, video, 'mesh_M.npz'))
        csM, rsM, xsM, ysM, zsM = data['cs'].astype(int), data['rs'].astype(int), data['xs'], data['ys'], data['zs']
        #
        # load mesh_K
        data = np.load(os.path.join(pathFolderScratch, video, 'mesh_K.npz'))
        xsK, ysK, zsK = data['xs'], data['ys'], data['zs']; assert np.std(zsK) < 1.e-3
        #
        # manage overwrite
        if os.path.exists(os.path.join(pathFolderScratch, video, 'K_wavenumbers')):
            fns0 = sorted([item[0:-4] for item in os.listdir(os.path.join(pathFolderScratch, video, 'M_modes')) if item.endswith(par['DMD_or_EOF']+'.npz')])
            fns1 = sorted([item[0:-6] for item in os.listdir(os.path.join(pathFolderScratch, video, 'K_wavenumbers')) if item.endswith(par['DMD_or_EOF']+'_K.npz')])
            if set(fns0) == set(fns1):
                print('... wavenumbers for video {:} were already computed'.format(video))
                continue
        #
        # obtain ks for each subvideo mode
        fns = sorted([item for item in os.listdir(os.path.join(pathFolderScratch, video, 'M_modes')) if item.endswith(par['DMD_or_EOF']+'.npz')])
        for fn in fns: # random.sample(fns, len(fns)): # WATCH OUT!!!
            #
            # obtain pathOut
            pathOut = os.path.join(pathFolderScratch, video, 'K_wavenumbers', os.path.splitext(fn)[0]+'_K.npz')
            if os.path.exists(pathOut) and not overwrite:
                print('... wavenumbers for {:} of video {:} were already computed'.format(fn, video))
                continue
            print('... computing wavenumbers for {:} of video {:}'.format(fn, video))
            #
            # read the mode file
            data = np.load(os.path.join(pathFolderScratch, video, 'M_modes', fn))
            w, T, phases, amplitudes = [data[item] for item in ['w', 'T', 'phases', 'amplitudes']] # in mesh_M
            #
            # obtain RKs
            hsAux = [par['min_depth']+(item+1)/par['nRadius_K']*(par['max_depth']-par['min_depth']) for item in range(par['nRadius_K'])]
            RKs = [0.6*ulises.TH2LOneStep(T, hAux, par['g']) for hAux in reversed(hsAux)] # reversed: from large to smaller
            #
            # obtain wavenumbers kxs and kys for mesh_K
            kxs, kys = [np.ones((len(xsK), len(RKs))) for item in range(2)]; kxs[:], kys[:] = np.NaN, np.NaN # initialized with NaN
            for posK in range(len(xsK)):
                # obtain distances of the points in mesh_M to posK
                dsToM = np.sqrt((xsM - xsK[posK]) ** 2 + (ysM - ysK[posK]) ** 2)
                pos0InM = np.argmin(dsToM) # point in mesh_M closest to posK
                # analysis for RKs
                xsMTMP, ysMTMP, phasesMTMP, dsToMTMP = [copy.deepcopy(item) for item in [xsM, ysM, phases, dsToM]]
                for posRK, RK in enumerate(RKs):
                    poss = np.where(dsToMTMP <= RK)[0]  # closest positions in mesh_M (posRK=0) or the previous neighborhood
                    xsMTMP, ysMTMP, phasesMTMP, dsToMTMP = [item[poss] for item in [xsMTMP, ysMTMP, phasesMTMP, dsToMTMP]] # update
                    if len(poss) < 6 or np.sqrt((np.mean(xsMTMP) - xsK[posK]) ** 2 + (np.mean(ysMTMP) - ysK[posK]) ** 2) > RK / 4: # IMP*
                        continue
                    # obtain points of the surface to fit
                    xsDel, ysDel, phasesDel = xsMTMP - xsM[pos0InM], ysMTMP - ysM[pos0InM], np.angle(np.exp(1j * (phasesMTMP - phases[pos0InM])))
                    AsDel = np.transpose(np.asarray([xsDel, ysDel, np.ones(len(xsDel))]))
                    # obtain the wavenumber through RANSAC and discard
                    possGood = []
                    for posRANSAC in range(par['nRANSAC_K']):
                        poss3 = random.sample(range(len(phasesDel)), 3)
                        try: # solving xsDel * sol[0] + ysDel * sol[1] + sol[2] = phasesDel
                            sol = np.linalg.solve(AsDel[poss3, :], phasesDel[poss3])
                        except:
                            continue
                        phasesDelR = np.dot(AsDel, sol)
                        possGoodH = np.where(np.abs(phasesDelR - phasesDel) < 0.25)[0] # IMP* error 0.25 = 15 sexagesimal degrees
                        if len(possGoodH) > len(possGood):
                            possGood = copy.deepcopy(possGoodH)
                    if len(possGood) < 6: # WATCH OUT
                        continue
                    #sol = np.dot(np.linalg.pinv(AsDel[possGood, :]), phasesDel[possGood]) # kx, ky = sol[0], sol[1]
                    sol = np.linalg.solve(np.dot(AsDel[possGood, :].transpose(), AsDel[possGood, :]), np.dot(AsDel[possGood, :].transpose(), phasesDel[possGood])) # kx, ky = sol[0], sol[1]
                    kxs[posK, posRK], kys[posK, posRK] = sol[0], sol[1]
            #
            # obtain ks and gammas (sets NaN to kxs, kys, ks and gammas where gamma > 1.2!!: setting 1.0 or 2.0 instead of 1.2 is not so important)
            ks = np.sqrt(kxs ** 2 + kys ** 2)
            for posRK in range(len(RKs)):
                possTMP = np.where((np.abs(ks[:, posRK]) < 1.e-6) | (1.2 * ks[:, posRK] < w ** 2 / par['g']))[0] # ks ** 2 < is useful
                kxs[possTMP, posRK], kys[possTMP, posRK], ks[possTMP, posRK] = np.NaN, np.NaN, np.NaN
            gammas = w ** 2 / (par['g'] * ks)
            #
            # obtain meanGs and stdGs (sets NaN where ks is NaN and in more positions): VERY IMP* to compute bathymetry
            meanGs, stdGs = [np.ones((len(xsK), len(RKs))) for item in range(2)]; meanGs[:], stdGs[:] = np.NaN, np.NaN # initialized with NaN
            for posK in range(len(xsK)):
                dsToKTMP = np.sqrt((xsK - xsK[posK]) ** 2 + (ysK - ysK[posK]) ** 2)
                for posRK, RK in enumerate(RKs):
                    if np.isnan(ks[posK, posRK]):
                        continue
                    RG = max([0.5 * 2. * np.pi / ks[posK, posRK], 2.1 * par['delta_K']]) # IMP*
                    possTMP = np.where((np.invert(np.isnan(ks[:, posRK]))) & (dsToKTMP <= RG))[0] # not-NaN points in the neighborhood
                    if len(possTMP) < 3: # IMP*
                        continue
                    meanGs[posK, posRK], stdGs[posK, posRK] = np.mean(gammas[possTMP, posRK]), np.std(gammas[possTMP, posRK])
            #
            # write the results
            ulises.MakeFolder(os.path.split(pathOut)[0])
            np.savez(pathOut, w=w, T=T, RKs=RKs, ks=ks, meanGs=meanGs, stdGs=stdGs)
            #
            # plot the results
            if verbosePlot:
                pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video, 'K_wavenumbers', os.path.splitext(fn)[0]+'_K.png')
                fig = plt.figure(figsize=[9, 9]) # width x height
                plt.suptitle('T = {:3.2f} s'.format(T))
                plt.tight_layout()
                for posRK in range(len(RKs)):
                    #
                    plt.subplot(len(RKs), 3, 3*posRK+1)
                    plt.title('phase [rad]')
                    sct = plt.scatter(xsM, ysM, marker='o', c=phases, vmin=-np.pi, vmax=np.pi, edgecolor='none', s=5, cmap='jet')
                    possTMP = np.where((xsK - np.mean(xsK)) ** 2 + (ysK - np.mean(ysK)) ** 2 <= RKs[posRK] ** 2)[0]
                    plt.plot(xsK[possTMP], ysK[possTMP], 'w.')
                    plt.colorbar(sct); plt.axis('equal'); plt.xlim([xsK.min(), xsK.max()]); plt.axis('off')
                    #
                    plt.subplot(len(RKs), 3, 3*posRK+2)
                    plt.title('zb [m] ({:})'.format(np.sum(np.invert(np.isnan(ks[:, posRK])))))
                    plt.plot(np.min(xsK), np.min(ysK), 'w.'); plt.plot(np.max(xsK), np.max(ysK), 'w.')
                    zbs = np.mean(zsK) - np.arctanh(np.clip(gammas[:, posRK], 0, 1-1.e-6)) / ks[:, posRK] # gammas and ks have NaN
                    sct = plt.scatter(xsK, ysK, marker='o', c=zbs, vmin=-par['max_depth'], vmax=0, edgecolor='none', s=5, cmap='gist_earth')
                    plt.colorbar(sct); plt.axis('equal'); plt.xlim([xsK.min(), xsK.max()]); plt.axis('off')
                    #
                    plt.subplot(len(RKs), 3, 3*posRK+3)
                    plt.title(r'$\sigma_\gamma$ [-] ({:})'.format(np.sum(np.invert(np.isnan(stdGs[:, posRK])))))
                    plt.plot(np.min(xsK), np.min(ysK), 'w.'); plt.plot(np.max(xsK), np.max(ysK), 'w.')
                    sct = plt.scatter(xsK, ysK, marker='o', c=stdGs[:, posRK], vmin=0, vmax=0.2, edgecolor='none', s=5, cmap='CMRmap_r')
                    plt.colorbar(sct); plt.axis('equal'); plt.xlim([xsK.min(), xsK.max()]); plt.axis('off')
                    #
                ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
                plt.savefig(pathTMPPlot, dpi=100)
                plt.close('all')
    #
    return None
#
def ObtainZb(pathFolderData, pathFolderScratch, pathFolderBathymetries, overwrite, verbosePlot): # last read 2022-07-10
    #
    def GoalFunction220622(x, theArgs): # IMP*
        ws, gammas, zts, g = [theArgs[key] for key in ['ws', 'gammas', 'zts', 'g']]
        gammasR = ws ** 2 / (par['g'] * ulises.WGH2KOneStep(ws, par['g'], zts-x[0]))
        gf = np.sqrt(np.mean((gammasR - gammas) ** 2))
        return gf
    def interp_weights220622(xy, uv, d=2):
        tri = sc.spatial.Delaunay(xy)
        simplex = tri.find_simplex(uv)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], uv - temp[:, d])
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    def interpolate220622(values, vtx, wts):
        return np.einsum('nj,nj->n', np.take(values, vtx), wts)
    #
    # load parameters
    pathTMP = os.path.join(pathFolderData, 'parameters.json')
    with open(pathTMP, 'r') as f:
        par = json.load(f)
    #
    # load videos4dates
    pathTMP = os.path.join(pathFolderData, 'videos4dates.json')
    with open(pathTMP, 'r') as f:
        videos4dates = json.load(f)
    dates = videos4dates.keys()
    #
    # load mesh_Zb
    data = np.load(os.path.join(pathFolderScratch, 'mesh_Zb.npz'))
    xsZb, ysZb = data['xs'], data['ys']
    #
    for date in dates:
        #
        # obtain pathOut
        pathOut = os.path.join(pathFolderScratch, 'Zb_bathymetries', '{:}_{:}_Zb.npz'.format(date, par['DMD_or_EOF']))
        if os.path.exists(pathOut) and not overwrite:
            print('... bathymetry for date {:} was already computed'.format(date))
            continue
        print('... computing bathymetry for date {:}'.format(date), end='', flush=True)
        pathOutTXT = os.path.join(pathFolderBathymetries, '{:}_Zb.txt'.format(date))
        pathmZbTXT = os.path.join(pathFolderBathymetries, 'mesh_Zb.txt')
        #
        # run through videos for the date
        xsKW, ysKW, ksKW, meanGsKW, TsKW, ztsKW = [np.asarray([]) for item in range(6)]
        for video in videos4dates[date]:
            #
            # load mesh_K
            data = np.load(os.path.join(pathFolderScratch, video, 'mesh_K.npz'))
            xsK, ysK, zsK = data['xs'], data['ys'], data['zs']; assert np.std(zsK) < 1.e-3
            #
            # load wavenumbers at mesh_K
            fns = sorted([item for item in os.listdir(os.path.join(pathFolderScratch, video, 'K_wavenumbers')) if item.endswith(par['DMD_or_EOF']+'_K.npz')])
            for fn in fns:
                data = np.load(os.path.join(pathFolderScratch, video, 'K_wavenumbers', fn))
                w, T, ks, meanGs, stdGs, RKs = data['w'], data['T'], data['ks'], data['meanGs'], data['stdGs'], data['RKs']
                gammas = w ** 2 / (par['g'] * ks) # contains NaNs
                for posRK in range(len(RKs)):
                    # obtain control variables
                    gs0 = stdGs[:, posRK] # contains NaNs in potentially different positions than gammas
                    gs1 = np.abs(gammas[:, posRK] - meanGs[:, posRK]) # contains NaNs where gammas or meanGs have NaNs
                    # obtain valid positions (not NaN and further conditions)
                    poss = np.where((np.invert(np.isnan(gs0))) & (np.invert(np.isnan(gs1))) & (gs0 <= par['stdGammaC']) & (gs1 <= par['stdGammaC']))[0] # IMP*
                    # add valid positions
                    xsKW, ysKW = np.append(xsKW, xsK[poss]), np.append(ysKW, ysK[poss])
                    ksKW, meanGsKW = np.append(ksKW, ks[poss, posRK]), np.append(meanGsKW, meanGs[poss, posRK])
                    TsKW, ztsKW = np.append(TsKW, T*np.ones(len(poss))), np.append(ztsKW, np.mean(zsK)*np.ones(len(poss)))
        gammasKW, wsKW = (2. * np.pi / TsKW) ** 2 / (par['g'] * ksKW), 2. * np.pi / TsKW
        #assert np.sum(np.isnan(ksKW)) == np.sum(np.isnan(meanGsKW)) == np.sum(np.isnan(gammasKW)) == np.sum(np.isnan(wsKW)) == 0
        #
        if len(xsKW) == 0:
            print(' failed')
            continue
        #
        # obtain zsZb and zsZbe
        zsZb, zsZbe = [np.ones(len(xsZb)) for item in range(2)]; zsZb[:], zsZbe[:] = np.NaN, np.NaN
        for posZb in range(len(xsZb)):
            #
            dsToKWTMP = np.sqrt((xsKW - xsZb[posZb]) ** 2 + (ysKW - ysZb[posZb]) ** 2)
            
            poss0 = np.where(dsToKWTMP <= 1.1 * np.min(dsToKWTMP) + 1.e-3)[0]
            R_Zb = max([par['cRadius_Zb'] * np.mean(2. * np.pi * par['g'] * meanGsKW[poss0] / wsKW[poss0] ** 2), 2.1 * par['delta_K']])
            possInKW = np.where(dsToKWTMP <= R_Zb)[0]
            #
            if len(possInKW) < 10: # WATCH
                continue
            #
            # obtain gammas, ws and zts around
            gammasTMP, wsTMP, ztsTMP = [item[possInKW] for item in [gammasKW, wsKW, ztsKW]]
            #
            # obtain zb through RANSAC
            possGood, zb = [], np.NaN
            for zbH in np.arange(-par['max_depth'], -0.1, 0.05):
                gammasH = wsTMP ** 2 / (par['g'] * ulises.WGH2KOneStep(wsTMP, par['g'], ztsTMP - zbH))
                possGoodH = np.where(np.abs(gammasH - gammasTMP) < par['stdGammaC'])[0]
                if len(possGoodH) > len(possGood):
                    possGood, zb = [copy.deepcopy(item) for item in [possGoodH, zbH]]
            #
            if len(possGood) < 10: # WATCH
                continue
            #
            # obtain zb through the minimization for RANSAC good positions
            theArgs = {'gammas':gammasTMP[possGood], 'ws':wsTMP[possGood], 'zts':ztsTMP[possGood], 'g':par['g']}
            zb = optimize.minimize(GoalFunction220622, np.asarray([zb]), args=(theArgs)).x[0] # IMP* CHECK
            #
            # obtain zbe: the self error DANI
            #zbe = np.abs(np.mean(ztsKW[possInKW] - np.arctanh(np.clip(gammasKW[possInKW], 0, 0.999999)) / ksKW[possInKW]) - zb)
            #zbe = np.std(ztsKW[possInKW] - np.arctanh(np.clip(gammasKW[possInKW], 0, 0.999999)) / ksKW[possInKW])
            #if len(poss0) < 10:
                #zbe = 1.
            zbe = np.std(ztsKW[poss0] - np.arctanh(np.clip(gammasKW[poss0], 0, 0.999999)) / ksKW[poss0])
            if len(poss0) < 5:
                zbe = 1.
            #
            zsZb[posZb], zsZbe[posZb] = zb, zbe # nsPairs[posZb], nsPairsG[posZb] = len(possInKW), len(possGood)
        #
        # write the results
        ulises.MakeFolder(os.path.split(pathOut)[0])
        np.savez(pathOut, zsZb=zsZb, zsZbe=zsZbe)
        #
        ulises.MakeFolder(os.path.split(pathOutTXT)[0])
        fileOutTXT = open(pathOutTXT, 'w')
        filemZbTXT = open(pathmZbTXT, 'w')
        for posZb in range(len(xsZb)):
            fileOutTXT.write('{:9.4f} {:9.4f}\n'.format(zsZb[posZb], zsZbe[posZb]))
            filemZbTXT.write('{:15.3f} {:15.3f}\n'.format(xsZb[posZb], ysZb[posZb]))
        fileOutTXT.close()
        filemZbTXT.close()
        #
        print(' success')
        #
        # interpolate and write groundTruth bathymetry
        pathBath = os.path.join(pathFolderData, 'groundTruth', '{:}_GT_xyz.txt'.format(date))
        if os.path.exists(pathBath):
            rawData = np.asarray(ulises.ReadRectangleFromTxt(pathBath, {'c1':3, 'valueType':'float'}))
            xsB, ysB, zbsB = rawData[:, 0], rawData[:, 1], rawData[:, 2]
            xsysB, xsysZb = np.transpose(np.asarray([xsB, ysB])), np.transpose(np.asarray([xsZb, ysZb]))
            vtx, wts = interp_weights220622(xsysB, xsysZb)
            zbsBInZb = interpolate220622(zbsB, vtx, wts)
            #
            pathOutGT = os.path.join(pathFolderScratch, 'Zb_bathymetries', '{:}_GT_Zb.npz'.format(date))
            np.savez(pathOutGT, zbsBInZb=zbsBInZb)
            #
            pathOutGTTXT = os.path.join(pathFolderBathymetries, '{:}_GT_Zb.txt'.format(date))
            fileOutGTTXT = open(pathOutGTTXT, 'w')
            for posZb in range(len(xsZb)):
                fileOutGTTXT.write('{:9.4f}\n'.format(zbsBInZb[posZb]))
            fileOutGTTXT.close()
        else:
            zbsBInZb = np.NaN
        # plot the result
        if verbosePlot:
            #
            pathTMPPlot = os.path.join(pathFolderScratch, 'plots', 'Zb_bathymetries', '{:}_{:}_Zb.png'.format(date, par['DMD_or_EOF']))
            ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
            pathTMPPlotTXT = os.path.join(pathFolderBathymetries, 'plots', '{:}_Zb.png'.format(date))
            ulises.MakeFolder(os.path.split(pathTMPPlotTXT)[0])
            #
            fig = plt.figure(figsize=[9, 4]) # width x height
            plt.suptitle('date = {:}'.format(date))
            #
            plt.subplot(1, 3, 1)
            plt.title('z_b [m]')
            plt.plot(np.min(xsZb), np.min(ysZb), 'w.'); plt.plot(np.max(xsZb), np.max(ysZb), 'w.')
            sct = plt.scatter(xsZb, ysZb, marker='o', c=zsZb, edgecolor='none', vmin=-par['max_depth'], vmax=-par['min_depth'], s=5, cmap='gist_earth')
            plt.colorbar(sct); plt.axis('equal'); plt.axis('off')
            #
            plt.subplot(1, 3, 2)
            plt.title('self er [m]')
            plt.plot(np.min(xsZb), np.min(ysZb), 'w.'); plt.plot(np.max(xsZb), np.max(ysZb), 'w.')
            sct = plt.scatter(xsZb, ysZb, marker='o', c=zsZbe, vmin=0, vmax=1, edgecolor='none', s=5, cmap='jet')
            plt.colorbar(sct); plt.axis('equal'); plt.axis('off')
            #
            plt.subplot(1, 3, 3)
            plt.title('er [-]')
            plt.plot(np.min(xsZb), np.min(ysZb), 'w.'); plt.plot(np.max(xsZb), np.max(ysZb), 'w.')
            sct = plt.scatter(xsZb, ysZb, marker='o', c=(zsZb-zbsBInZb)/np.abs(zbsBInZb), vmin=-1, vmax=1, edgecolor='none', s=5, cmap='seismic_r')
            plt.colorbar(sct); plt.axis('equal'); plt.axis('off')
            #
            plt.tight_layout()
            plt.savefig(pathTMPPlot, dpi=100)
            plt.savefig(pathTMPPlotTXT, dpi=100)
            plt.close('all')
    #
    return None
#
