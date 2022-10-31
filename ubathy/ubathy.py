#'''
# Created on 2022 by Gonzalo Simarro and Daniel Calvete
#'''
#
import ulises_ubathy as ulises
#
import copy
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy as sc
from scipy import optimize
from scipy import signal
import sys
#
grav = sc.constants.g
#
def Video2Frames(pathFolderVideos, listOfVideos, fps, overwrite): # last read 2022-10-28
    #
    # obtain video filenames videoFns
    extsVid = ['mp4', 'MP4', 'avi', 'AVI', 'mov', 'MOV']
    videoFns = sorted([item for item in os.listdir(pathFolderVideos) if os.path.splitext(item)[1][1:] in extsVid])
    if len(listOfVideos) > 0: # (the videos in listOfVideos DO NOT have extension; a videoFn in videoFns DOES have)
        videoFns = [item for item in videoFns if os.path.splitext(item)[0] in listOfVideos]
    if len(videoFns) == 0:
        print('... no video-files to extract frames from')
    #
    # obtain frames from videos
    for videoFn in videoFns:
        pathVideo = os.path.join(pathFolderVideos, videoFn)
        #
        # obtain pathFolderSnaps and manage overwrite
        video = os.path.splitext(videoFn)[0]
        pathFolderSnaps = os.path.join(pathFolderVideos, video)
        if os.path.isdir(pathFolderSnaps) and not overwrite:
           print('... frame extraction for video {:} was already performed (the video folder exists)'.format(video)); continue
        #
        # load video and obtain fps
        fpsOfVideo = cv2.VideoCapture(pathVideo).get(cv2.CAP_PROP_FPS)
        if fps > fpsOfVideo:
            print('*** required fps ({:3.2f}) larger than actual fps of the video ({:3.2f})'.format(fps, fpsOfVideo)); sys.exit()
        elif fps == 0:
            fps = fpsOfVideo
        else: # 0 < fps <= fpsOfVideo so that int(fpsOfVideo / fps) >= 1
            fps = fpsOfVideo / int(fpsOfVideo / fps) # to ensure that fpsOfVideo is multiple of fps
        #
        # write frames
        print('... frame extraction of video {:} from {:} at {:3.2f} fps'.format(video, videoFn, fps))
        ulises.Video2Snaps(pathVideo, pathFolderSnaps, fps, options={})
        #
    return None
#
def CreateMeshes(pathFolderData, pathFolderVideos, pathFolderScratch, listOfVideos, overwrite, verbosePlot): # last read 2022-10-28
    #
    extsImg = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
    #
    # load parameters
    pathTMP = os.path.join(pathFolderData, 'parameters.json')
    with open(pathTMP, 'r') as f:
        par = json.load(f)
    #
    # load boundary of the domain where to obtain the bathymetry
    pathTMP = os.path.join(pathFolderData, 'xy_boundary.txt')
    dataTMP = np.asarray(ulises.ReadRectangleFromTxt(pathTMP, options={'c1':2, 'valueType':'float'}))
    xsBoun, ysBoun = [dataTMP[:, item] for item in range(2)]
    #
    # create, write and load mesh_B (depends exclusively on xy_boundary)
    pathTMP = os.path.join(pathFolderScratch, 'mesh_B.npz')
    if not os.path.exists(pathTMP) or overwrite:
        print('... creating mesh_B')
        #
        # create mesh_B
        xsB, ysB = ulises.Polyline2InnerHexagonalMesh({'xs':xsBoun, 'ys':ysBoun}, par['delta_B'], options={})
        #
        # write mesh_B in npz
        ulises.MakeFolder(os.path.split(pathTMP)[0])
        np.savez(pathTMP, xs=xsB, ys=ysB)
        #
        # plot mesh_B
        if verbosePlot:
            pathTMPPlot = os.path.join(pathFolderScratch, 'plots', 'mesh_B.png')
            ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
            plt.plot(list(xsBoun) + [xsBoun[0]], list(ysBoun) + [ysBoun[0]], 'r-', lw=5)
            plt.plot(xsB, ysB, 'g.', markersize=0.4)
            plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal')
            plt.tight_layout()
            plt.savefig(pathTMPPlot, dpi=100); plt.close()
    else:
        print('... mesh_B was already created')
    #
    # obtain listOfVideos
    listOfVideosTMP = sorted([item for item in os.listdir(pathFolderData) if os.path.isdir(os.path.join(pathFolderData, item)) and item != 'groundTruth'])
    if len(listOfVideos) > 0:
        listOfVideos = [item for item in listOfVideosTMP if item in listOfVideos]
    else:
        listOfVideos = copy.deepcopy(listOfVideosTMP)
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
            #
            # create initial mesh_M
            if isPlw: # pixels correspond to the planview and are integers
                dataTMP = np.asarray(ulises.ReadRectangleFromTxt(pathPlwTxt, {'c1':5, 'valueType':'float'}))
                csC, rsC, xsC, ysC, zsC = [dataTMP[:, item] for item in range(5)]; assert np.std(zsC) < 1.e-3
                csC, rsC = [np.round(item).astype(int) for item in [csC, rsC]]
                mcsM, mrsM = np.meshgrid(np.arange(min(csC), max(csC)+1), np.arange(min(rsC), max(rsC)+1)) # IMP
                csM, rsM = [np.reshape(item, -1) for item in [mcsM, mrsM]]
                (xsM, ysM), zsM = ulises.ApplyAffineA01(ulises.FindAffineA01(csC, rsC, xsC, ysC), csM, rsM), np.mean(zsC) * np.ones(len(csM))
            else: # pixels correspond to the oblique image and are integers
                #
                # load zt and obtain mesh in xy
                zt = ulises.ReadRectangleFromTxt(pathZsTxt, {'c1':1, 'valueType':'float'})[0]
                xsM, ysM = ulises.Polyline2InnerHexagonalMesh({'xs':xsBoun, 'ys':ysBoun}, par['delta_M'], options={})
                #
                # obtain mesh in cr
                mainSet = ulises.ReadMainSetFromCalTxt(pathCalTxt, options={})
                csM, rsM, possGood = ulises.XYZ2CDRD(mainSet, xsM, ysM, zt*np.ones(len(xsM)), options={'returnGoodPositions':True})
                csM, rsM = [np.round(item[possGood]).astype(int) for item in [csM, rsM]] # IMP*
                #
                # retain only unique points in cr
                auxs = list(set(zip(csM, rsM))) # unique, interesting
                csM, rsM = np.asarray([item[0] for item in auxs]), np.asarray([item[1] for item in auxs])
                #
                # obtain mesh in xy from cr
                xsM, ysM, possGood = ulises.CDRDZ2XY(mainSet, csM, rsM, zt*np.ones(len(csM)), options={'returnGoodPositions':True})
                csM, rsM, xsM, ysM = [item[possGood] for item in [csM, rsM, xsM, ysM]]
                #
                # sort mesh in xy using the distance to (x0, y0)
                xC, yC = np.mean(xsM), np.mean(ysM)
                dC = np.sqrt((xC - mainSet['xc']) ** 2 + (yC - mainSet['yc']) ** 2)
                x0, y0 = xC + 5000 * (xC - mainSet['xc']) / dC, yC + 5000 * (yC - mainSet['yc']) / dC # IMP*
                possSorted = np.argsort((xsM - x0) ** 2 + (ysM - y0) ** 2)
                csM, rsM, xsM, ysM = [item[possSorted] for item in [csM, rsM, xsM, ysM]]
                #
                # obtain uniformized mesh in crxyz
                csMu, rsMu, xsMu, ysMu = [np.asarray([item[0]]) for item in [csM, rsM, xsM, ysM]] # initialize with the first point
                for pos in range(1, len(csM)): # potential point to the final mesh
                    cM, rM, xM, yM = [item[pos] for item in [csM, rsM, xsM, ysM]]
                    if np.min((xM - xsMu) ** 2 + (yM - ysMu) ** 2) >= (0.95 * par['delta_M']) ** 2:
                        csMu, rsMu, xsMu, ysMu = np.append(csMu, cM), np.append(rsMu, rM), np.append(xsMu, xM), np.append(ysMu, yM)
                csM, rsM, xsM, ysM, zsM = csMu, rsMu, xsMu, ysMu, zt*np.ones(len(csMu))
            #
            # retain only the points within the bathymetry boundary (useful, indeed, for isPlw)
            possIn = ulises.CloudOfPoints2InnerPositionsForPolyline(xsM, ysM, {'xs':xsBoun, 'ys':ysBoun}, options={})
            csM, rsM, xsM, ysM, zsM = [item[possIn] for item in [csM, rsM, xsM, ysM, zsM]]
            #
            # write mesh_M in npz
            ulises.MakeFolder(os.path.split(pathTMP)[0])
            np.savez(pathTMP, cs=csM, rs=rsM, xs=xsM, ys=ysM, zs=zsM)
            #
            # plot mesh_M
            if verbosePlot:
                pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video, 'mesh_M.png')
                ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
                plt.plot(list(xsBoun) + [xsBoun[0]], list(ysBoun) + [ysBoun[0]], 'r-', lw=5)
                plt.plot(xsM, ysM, 'g.', markersize=0.4)
                plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal')
                plt.tight_layout()
                plt.savefig(pathTMPPlot, dpi=100); plt.close()
                #
                pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video, 'mesh_M_inImage.png')
                pathsImg = [os.path.join(pathFolderVideos, video, item) for item in os.listdir(os.path.join(pathFolderVideos, video)) if os.path.splitext(item)[1][1:] in extsImg]
                if len(pathsImg) > 0:
                    img = cv2.imread(pathsImg[0])
                    img = ulises.DisplayCRInImage(img, csM, rsM, options={'colors':[[0, 255, 0]], 'size':1.5})
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
            #
            # create initial mesh_K
            xsK, ysK = ulises.Polyline2InnerHexagonalMesh({'xs':xsBoun, 'ys':ysBoun}, par['delta_K'], options={})
            if isPlw: # the pixels correspond to the planview
                dataTMP = np.asarray(ulises.ReadRectangleFromTxt(pathPlwTxt, {'c1':5, 'valueType':'float'}))
                csP, rsP, xsP, ysP, zsP = [dataTMP[:, item] for item in range(5)]; assert np.std(zsP) < 1.e-3
                nc, nr = [int(np.round(max(item)-min(item))) for item in [csP, rsP]]
                possC = [np.where((csP == item[0]) & (rsP == item[1]))[0][0] for item in [[min(csP), min(rsP)], [max(csP), min(rsP)], [max(csP), max(rsP)], [min(csP), max(rsP)]]]
                possIn = ulises.CloudOfPoints2InnerPositionsForPolyline(xsK, ysK, {'xs':xsP[possC], 'ys':ysP[possC]}, options={}) # points of mesh_K in plw
                (xsK, ysK), zsK = [item[possIn] for item in [xsK, ysK]], np.mean(zsP) * np.ones(len(possIn))
                csK, rsK = ulises.ApplyAffineA01(ulises.FindAffineA01(xsP, ysP, csP, rsP), xsK, ysK) # real valued (only for TMPPlot)
            else: # the pixels correspond to the oblique image
                zt = ulises.ReadRectangleFromTxt(pathZsTxt, {'c1':1, 'valueType':'float'})[0]
                mainSet = ulises.ReadMainSetFromCalTxt(pathCalTxt, options={})
                csK, rsK, possGood = ulises.XYZ2CDRD(mainSet, xsK, ysK, zt*np.ones(len(xsK)), options={'returnGoodPositions':True})
                (csK, rsK, xsK, ysK), zsK = [item[possGood] for item in [csK, rsK, xsK, ysK]], zt * np.ones(len(xsK))
            #
            # write mesh_K in npz
            ulises.MakeFolder(os.path.split(pathTMP)[0])
            np.savez(pathTMP, xs=xsK, ys=ysK, zs=zsK)
            #
            # plot mesh_K
            if verbosePlot:
                pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video, 'mesh_K.png')
                ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
                plt.plot(list(xsBoun) + [xsBoun[0]], list(ysBoun) + [ysBoun[0]], 'r-', lw=5)
                plt.plot(xsK, ysK, 'g.', markersize=0.4)
                plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal')
                plt.tight_layout()
                plt.savefig(pathTMPPlot, dpi=100); plt.close()
                #
                pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video, 'mesh_K_inImage.png')
                pathsImg = [os.path.join(pathFolderVideos, video, item) for item in os.listdir(os.path.join(pathFolderVideos, video)) if os.path.splitext(item)[1][1:] in extsImg]
                if len(pathsImg) > 0:
                    img = cv2.imread(pathsImg[0])
                    img = ulises.DisplayCRInImage(img, csK, rsK, options={'colors':[[0, 255, 0]], 'size':1.5})
                    ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
                    cv2.imwrite(pathTMPPlot, img)
                else:
                    print('... no images in folder {:} to plot {:}'.format(os.path.join(pathFolderVideos, video), pathTMPPlot))
        else:
            print('... mesh_K for video {:} was already created'.format(video))
    #
    return None
#
def ObtainWAndModes(pathFolderData, pathFolderVideos, pathFolderScratch, listOfVideos, overwrite, verbosePlot): # last read 2022-10-28
    #
    extsImg = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
    #
    # load parameters
    pathTMP = os.path.join(pathFolderData, 'parameters.json')
    with open(pathTMP, 'r') as f:
        par = json.load(f)
    #
    # obtain list of videos
    listOfVideosTMP = sorted([item for item in os.listdir(pathFolderData) if os.path.isdir(os.path.join(pathFolderData, item)) and item != 'groundTruth'])
    if len(listOfVideos) > 0:
        listOfVideos = [item for item in listOfVideosTMP if item in listOfVideos]
    else:
        listOfVideos = copy.deepcopy(listOfVideosTMP)
    #
    # run for videos
    for video in listOfVideos:
        #
        # manage overwrite
        if os.path.exists(os.path.join(pathFolderScratch, video, 'M_modes')) and not overwrite:
            if len([item for item in os.listdir(os.path.join(pathFolderScratch, video, 'M_modes')) if '{:}.'.format(par['DMD_or_EOF']) in item]) > 0:
                print('... modes {:} for video {:} were already computed (the scratch folder exists and contains {:} files)'.format(par['DMD_or_EOF'], video, par['DMD_or_EOF'])); continue
        if overwrite:
            pathsTMP = [os.path.join(pathFolderScratch, video, 'M_modes'), os.path.join(pathFolderScratch, 'plots', video, 'M_modes')]
            for pathTMP in pathsTMP:
                if os.path.exists(pathTMP):
                    for fn in [item for item in os.listdir(pathTMP) if '{:}.'.format(par['DMD_or_EOF']) in item]:
                        os.remove(os.path.join(pathTMP, fn))
        #
        # load fnsVideo (sorted)
        fnsVideo = sorted([item for item in os.listdir(os.path.join(pathFolderVideos, video)) if os.path.splitext(item)[1][1:] in extsImg])
        #
        # load ts in seconds
        if os.path.splitext(fnsVideo[0])[0].endswith('plw'): # ************plw.ext
            ts = np.asarray([int(os.path.splitext(fn)[0][-15:-3])/1000. for fn in fnsVideo]) # IMP* ts in seconds, the fn in miliseconds
        else: # ************.ext
            ts = np.asarray([int(os.path.splitext(fn)[0][-12:])/1000. for fn in fnsVideo]) # IMP* ts in seconds, the fn in miliseconds
        if len(ts) < 2: continue
        ts, dtsMean = ts - ts[0], np.mean(np.diff(ts))
        ts = dtsMean * np.arange(len(ts)) # IMP*
        if par['DMD_or_EOF'] == 'EOF' and par['min_period'] < 6 * dtsMean:
            print('*** working with EOF, "min_period" in json file must be >= 6 / fps (>= {:5.2f})'.format(6*dtsMean)); sys.exit()
        #
        # load cs and rs for mesh_M (coordinates in the image, whether planview or oblique)
        dataTMP = np.load(os.path.join(pathFolderScratch, video, 'mesh_M.npz'))
        csM, rsM, xsM, ysM = [dataTMP[item] for item in ['cs', 'rs', 'xs', 'ys']]
        csM, rsM = [np.round(item).astype(int) for item in [csM, rsM]]
        #
        # load the video corresponding to mesh_M
        print('... loading video {:}'.format(video))
        XAll = np.zeros((len(csM), len(ts)))
        for posT in range(len(ts)):
            img = cv2.imread(os.path.join(pathFolderVideos, video, fnsVideo[posT]))
            XAll[:, posT] = img[rsM, csM, 1] / 255. # load green channel
        #
        # run through the video
        wtMax, dtHilbert, nHilbert = np.max(par['time_windows']), par['max_period'], int(par['max_period']/dtsMean)+1
        for posT in range(nHilbert, len(ts), max([1, int(par['time_step']/dtsMean)])):
            if not (ts[posT]-dtHilbert>=0 and ts[posT]+wtMax+dtHilbert<=np.max(ts)): continue
            #
            # read lines for Hilbert-extended t-ball
            possTBallEx = sorted(np.where((ts>=ts[posT]-dtHilbert) & (ts<=ts[posT]+wtMax+dtHilbert))[0])
            XForTBallEx = XAll[:, possTBallEx]
            #
            # perform RPCA and Hilbert
            if par['candes_iter'] > 0:
                XForTBallEx = ulises.X2LSByCandes(XForTBallEx, options={'max_iter':par['candes_iter'], 'mu':1.})[0] # IMP*
            XForTBallExHilbert = signal.hilbert(XForTBallEx)
            #
            # crop the Hilbert extensions
            XForTBallHilbert = XForTBallExHilbert[:, nHilbert:-nHilbert]
            #
            # perform DMD/EOF for all 'time_windows'
            for wt in par['time_windows']:
                print('... {:} decomposition for t = {:5.1f} and wt = {:5.1f}'.format(par['DMD_or_EOF'], ts[posT], wt))
                fnOut0 = 't{:}_wt{:}'.format(str('{:3.2f}'.format(ts[posT])).zfill(8), str('{:3.2f}'.format(wt)).zfill(8))
                #
                # obtain positions for wt
                possWt = range(0, np.min([int(wt/dtsMean), XForTBallHilbert.shape[1]]))
                XForTBallHilbertWt = XForTBallHilbert[:, possWt]
                #
                # perform DMD/EOF for the time window
                modes = {}
                if par['DMD_or_EOF'] == 'DMD':
                    try:
                        Phi, Lambda = ulises.X2DMD(XForTBallHilbertWt, par['DMD_rank'])
                    except: continue
                    for posDMD in range(len(Lambda)):
                        w = np.imag(np.log(Lambda[posDMD]) / dtsMean)
                        T = 2. * np.pi / w
                        if not (par['min_period'] <= T <= par['max_period']): continue
                        modes[posDMD] = {'w':w, 'T':T, 'phases':np.angle(Phi[:, posDMD]), 'amplitudes':np.abs(Phi[:, posDMD])}
                elif par['DMD_or_EOF'] == 'EOF':
                    try:
                        EOF = ulises.X2EOF(XForTBallHilbertWt)
                    except: continue
                    for posEOF in range(len(EOF['explainedVariances'])):
                        var = np.real(EOF['explainedVariances'][posEOF])
                        if var < par['EOF_variance']: continue
                        w, wStd = ulises.GetWPhaseFitting(dtsMean*np.arange(len(possWt)), EOF['amplitudesForEOFs'][posEOF], 3*dtsMean, options={}) # IMP* 3*dtsMean as radius
                        if wStd / w > 0.15: continue # IMP* 0.15 as critical value
                        T = 2. * np.pi / w
                        if not (par['min_period'] <= T <= par['max_period']): continue
                        modes[posEOF] = {'var':var, 'wStdOverW':wStd/w, 'w':w, 'T':T, 'phases':np.angle(EOF['EOFs'][posEOF]), 'amplitudes':np.abs(EOF['EOFs'][posEOF])}
                else:
                    print('*** "DMD_or_EOF" in json file must be "DMD" or "EOF"'); sys.exit()
                #
                # write and plot the results
                for key in modes.keys():
                    #
                    # write the results in npz
                    fnOut = '{:}_T{:}_{:}.npz'.format(fnOut0, str('{:3.2f}'.format(modes[key]['T'])).zfill(8), par['DMD_or_EOF'])
                    pathTMP = os.path.join(pathFolderScratch, video, 'M_modes', fnOut)
                    ulises.MakeFolder(os.path.split(pathTMP)[0])
                    if par['DMD_or_EOF'] == 'DMD':
                        np.savez(pathTMP, w=modes[key]['w'], T=modes[key]['T'], phases=modes[key]['phases'], amplitudes=modes[key]['amplitudes'])
                    elif par['DMD_or_EOF'] == 'EOF':
                        np.savez(pathTMP, w=modes[key]['w'], T=modes[key]['T'], phases=modes[key]['phases'], amplitudes=modes[key]['amplitudes'], var=modes[key]['var'], wStdOverW=modes[key]['wStdOverW'])
                    #
                    # plot the results
                    if verbosePlot:
                        pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video, 'M_modes', '{:}.png'.format(os.path.splitext(fnOut)[0]))
                        ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
                        #
                        #fig = plt.figure(figsize=[6, 4]) # width x height
                        plt.suptitle('T = {:3.2f} s'.format(modes[key]['T']))
                        #
                        plt.subplot(1, 2, 1)
                        plt.title('phase [rad]')
                        plt.plot(np.min(xsM), np.min(ysM), 'w.'); plt.plot(np.max(xsM), np.max(ysM), 'w.')
                        sct = plt.scatter(xsM, ysM, marker='o', c=modes[key]['phases'], edgecolor='none', s=5, cmap='jet')
                        plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal'); plt.axis('off')
                        #
                        plt.subplot(1, 2, 2)
                        plt.title('amplitude [-]')
                        plt.plot(np.min(xsM), np.min(ysM), 'w.'); plt.plot(np.max(xsM), np.max(ysM), 'w.')
                        sct = plt.scatter(xsM, ysM, marker='o', c=modes[key]['amplitudes'], edgecolor='none', s=5, cmap='jet')
                        plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal'); plt.axis('off')
                        #
                        plt.tight_layout()
                        plt.savefig(pathTMPPlot, dpi=100)
                        plt.close('all')
    #
    return None
#
def ObtainK(pathFolderData, pathFolderScratch, listOfVideos, overwrite, verbosePlot): # last read 2022-10-28
    #
    # load parameters
    pathTMP = os.path.join(pathFolderData, 'parameters.json')
    with open(pathTMP, 'r') as f:
        par = json.load(f)
    #
    # obtain list of videos
    listOfVideosTMP = sorted([item for item in os.listdir(pathFolderData) if os.path.isdir(os.path.join(pathFolderData, item)) and item != 'groundTruth'])
    if len(listOfVideos) > 0:
        listOfVideos = [item for item in listOfVideosTMP if item in listOfVideos]
    else:
        listOfVideos = copy.deepcopy(listOfVideosTMP)
    #
    # run for videos
    for video in random.sample(listOfVideos, len(listOfVideos)): # randomized
        #
        # load mesh_M
        dataTMP = np.load(os.path.join(pathFolderScratch, video, 'mesh_M.npz'))
        csM, rsM, xsM, ysM, zsM = [dataTMP[item] for item in ['cs', 'rs', 'xs', 'ys', 'zs']]; assert np.std(zsM) < 1.e-3
        csM, rsM = [np.round(item).astype(int) for item in [csM, rsM]]
        #
        # load mesh_K
        dataTMP = np.load(os.path.join(pathFolderScratch, video, 'mesh_K.npz'))
        xsK, ysK, zsK = [dataTMP[item] for item in ['xs', 'ys', 'zs']]; assert np.std(zsK) < 1.e-3
        #
        # manage overwrite
        if os.path.exists(os.path.join(pathFolderScratch, video, 'K_wavenumbers')):
            fnsTMP0 = set([item[0:-4] for item in os.listdir(os.path.join(pathFolderScratch, video, 'M_modes')) if item.endswith('{:}.npz'.format(par['DMD_or_EOF']))])
            fnsTMP1 = set([item[0:-6] for item in os.listdir(os.path.join(pathFolderScratch, video, 'K_wavenumbers')) if item.endswith('{:}_K.npz'.format(par['DMD_or_EOF']))])
            if fnsTMP0 == fnsTMP1 and not overwrite:
                print('... wavenumbers for video {:} were already computed'.format(video)); continue
        if overwrite:
            pathsTMP = [os.path.join(pathFolderScratch, video, 'K_wavenumbers'), os.path.join(pathFolderScratch, 'plots', video, 'K_wavenumbers')]
            for pathTMP in pathsTMP:
                if os.path.exists(pathTMP):
                    for fn in [item for item in os.listdir(pathTMP) if '{:}_K.'.format(par['DMD_or_EOF']) in item]:
                        os.remove(os.path.join(pathTMP, fn))
        #
        # obtain ks for each subvideo mode
        if not os.path.exists(os.path.join(pathFolderScratch, video, 'M_modes')):
            print('... there are no modes to obtain wavenumbers from for video {:}'.format(video)); continue
        fns = sorted([item for item in os.listdir(os.path.join(pathFolderScratch, video, 'M_modes')) if item.endswith('{:}.npz'.format(par['DMD_or_EOF']))])
        for fn in random.sample(fns, len(fns)): # randomized
            #
            # obtain pathOut (here, to manage overwrite)
            fnOut = '{:}_K.npz'.format(os.path.splitext(fn)[0])
            pathOut = os.path.join(pathFolderScratch, video, 'K_wavenumbers', fnOut)
            if os.path.exists(pathOut) and not overwrite:
                print('... wavenumbers for {:} of video {:} were already computed'.format(fn, video)); continue
            print('... computing wavenumbers for {:} of video {:}'.format(fn, video))
            #
            # read the mode file
            dataTMP = np.load(os.path.join(pathFolderScratch, video, 'M_modes', fn))
            w, T, phasesM = [dataTMP[item] for item in ['w', 'T', 'phases']] # given in mesh_M
            #
            # obtain RKs
            hsAux = [par['min_depth']+(item+1)/par['nRadius_K']*(par['max_depth']-par['min_depth']) for item in range(par['nRadius_K'])] # from small to larger
            RKs = np.asarray([par['cRadius_K']*ulises.TH2LOneStep(T, hAux, grav) for hAux in reversed(hsAux)]) # from large to smaller
            #
            # obtain wavenumbers kxs and kys for mesh_K
            kxs, kys = [np.empty((len(xsK), len(RKs))) for item in range(2)]; kxs[:], kys[:] = np.NaN, np.NaN # initialized as NaNs
            for posK in range(len(xsK)):
                #
                # obtain distances of the points in mesh_M to posK
                dsToM = np.sqrt((xsM - xsK[posK]) ** 2 + (ysM - ysK[posK]) ** 2)
                pos0InM = np.argmin(dsToM) # point in mesh_M closest to posK
                #
                # analysis for RKs
                xsMTMP, ysMTMP, phasesMTMP, dsToMTMP = [copy.deepcopy(item) for item in [xsM, ysM, phasesM, dsToM]] # initialize auxiliar
                for posRK, RK in enumerate(RKs):
                    poss = np.where(dsToMTMP <= RK)[0]
                    xsMTMP, ysMTMP, phasesMTMP, dsToMTMP = [item[poss] for item in [xsMTMP, ysMTMP, phasesMTMP, dsToMTMP]] # update auxiliar
                    if len(poss) < 6 or np.sqrt((np.mean(xsMTMP) - xsK[posK]) ** 2 + (np.mean(ysMTMP) - ysK[posK]) ** 2) > RK / 4: continue # IMP*
                    #
                    # obtain points of the surface to fit
                    xsDel, ysDel, phasesDel = xsMTMP - xsM[pos0InM], ysMTMP - ysM[pos0InM], np.angle(np.exp(1j * (phasesMTMP - phasesM[pos0InM])))
                    AsDel = np.transpose(np.asarray([xsDel, ysDel, np.ones(len(xsDel))]))
                    #
                    # obtain the wavenumber through RANSAC
                    if par['nRANSAC_K'] > 0:
                        possGood = []
                        for posRANSAC in range(par['nRANSAC_K']):
                            poss3 = random.sample(range(len(phasesDel)), 3)
                            try: # solving xsDel * sol[0] + ysDel * sol[1] + sol[2] = phasesDel
                                sol = np.linalg.solve(AsDel[poss3, :], phasesDel[poss3])
                            except: continue
                            possGoodH = np.where(np.abs(np.dot(AsDel, sol) - phasesDel) < 0.25)[0] # IMP* error 0.25 = 15 sexagesimal degrees
                            if len(possGoodH) > len(possGood):
                                possGood = copy.deepcopy(possGoodH)
                    else:
                        possGood = list(range(len(xsDel)))
                    if len(possGood) < 6: continue # IMP*
                    sol = np.linalg.solve(np.dot(AsDel[possGood, :].transpose(), AsDel[possGood, :]), np.dot(AsDel[possGood, :].transpose(), phasesDel[possGood])) # kx, ky = sol[0], sol[1]
                    kxs[posK, posRK], kys[posK, posRK] = sol[0], sol[1]
            #
            # obtain ks and gammas setting NaNs to kxs, kys, ks and gammas where gamma > 1.2: setting 1.0 or 2.0 instead of 1.2 is not so important
            ks = np.sqrt(kxs ** 2 + kys ** 2)
            for posRK in range(len(RKs)):
                possTMP = np.where((ks[:, posRK] < 1.e-6) | (1.2 * ks[:, posRK] < w ** 2 / grav))[0] # bad positions, ks < is useful
                kxs[possTMP, posRK], kys[possTMP, posRK], ks[possTMP, posRK] = np.NaN, np.NaN, np.NaN # IMP*
            gammas = w ** 2 / (grav * ks)
            #
            # obtain meanGs and stdGs setting NaNs where ks is NaN (and in more positions), IMP*
            meanGs, stdGs = [np.empty((len(xsK), len(RKs))) for item in range(2)]; meanGs[:], stdGs[:] = np.NaN, np.NaN # initialized as NaNs
            for posK in range(len(xsK)):
                dsToKTMP = np.sqrt((xsK - xsK[posK]) ** 2 + (ysK - ysK[posK]) ** 2)
                for posRK, RK in enumerate(RKs):
                    if np.isnan(ks[posK, posRK]): continue # IMP* meanGs and stdGs are NaNs
                    RG = max([0.5 * 2. * np.pi / ks[posK, posRK], 2.1 * par['delta_K']]) # IMP*
                    possTMP = np.where((np.invert(np.isnan(ks[:, posRK]))) & (dsToKTMP <= RG))[0] # IMP* not-NaN points in the neighborhood
                    if len(possTMP) < 3: continue # IMP*
                    meanGs[posK, posRK], stdGs[posK, posRK] = np.mean(gammas[possTMP, posRK]), np.std(gammas[possTMP, posRK]) # these mean and std are actually nanmean and nanstd
            #
            # write the results in npz
            ulises.MakeFolder(os.path.split(pathOut)[0])
            np.savez(pathOut, w=w, T=T, RKs=RKs, ks=ks, meanGs=meanGs, stdGs=stdGs)
            #
            # plot the results
            if verbosePlot:
                pathTMPPlot = os.path.join(pathFolderScratch, 'plots', video, 'K_wavenumbers', '{:}.png'.format(os.path.splitext(fnOut)[0]))
                #
                #fig = plt.figure(figsize=[9, 9]) # width x height
                plt.suptitle('T = {:3.2f} s'.format(T))
                for posRK in range(len(RKs)):
                    plt.subplot(len(RKs), 3, 3*posRK+1)
                    plt.title('phase [rad]')
                    sct = plt.scatter(xsM, ysM, marker='o', c=phasesM, vmin=-np.pi, vmax=np.pi, edgecolor='none', s=5, cmap='jet')
                    possTMP = np.where((xsK - np.mean(xsK)) ** 2 + (ysK - np.mean(ysK)) ** 2 <= RKs[posRK] ** 2)[0]
                    plt.plot(xsK[possTMP], ysK[possTMP], 'w.')
                    plt.colorbar(sct); plt.axis('equal'); plt.xlim([np.min(xsK), np.max(xsK)]); plt.axis('off')
                    #
                    plt.subplot(len(RKs), 3, 3*posRK+2)
                    plt.title(r'$z_b$ [m]')
                    plt.plot(np.min(xsK), np.min(ysK), 'w.'); plt.plot(np.max(xsK), np.max(ysK), 'w.')
                    zbs = np.mean(zsK) - np.arctanh(np.clip(gammas[:, posRK], 0, 1-1.e-6)) / ks[:, posRK] # gammas and ks have NaN
                    sct = plt.scatter(xsK, ysK, marker='o', c=zbs, vmin=-par['max_depth'], vmax=0, edgecolor='none', s=5, cmap='gist_earth')
                    plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal'); plt.xlim([np.min(xsK), np.max(xsK)]); plt.axis('off')
                    #
                    plt.subplot(len(RKs), 3, 3*posRK+3)
                    plt.title(r'$\sigma_\gamma$ [-]')
                    plt.plot(np.min(xsK), np.min(ysK), 'w.'); plt.plot(np.max(xsK), np.max(ysK), 'w.')
                    sct = plt.scatter(xsK, ysK, marker='o', c=stdGs[:, posRK], vmin=0, vmax=0.2, edgecolor='none', s=5, cmap='CMRmap_r')
                    plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal'); plt.xlim([np.min(xsK), np.max(xsK)]); plt.axis('off')
                    #
                plt.tight_layout()
                ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
                plt.savefig(pathTMPPlot, dpi=100)
                plt.close('all')
    #
    return None
#
def ObtainB(pathFolderData, pathFolderScratch, pathFolderBathymetries, overwrite, verbosePlot): # last read 2022-10-28
    #
    def GoalFunction220622(x, theArgs): # IMP*
        gammas, ws, zts, g = [theArgs[key] for key in ['gammas', 'ws', 'zts', 'g']]
        gammasR = ws ** 2 / (grav * ulises.WGH2KOneStep(ws, grav, zts-x[0]))
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
    dates = sorted(videos4dates.keys())
    #
    # load mesh_B
    pathTMP = os.path.join(pathFolderScratch, 'mesh_B.npz')
    dataTMP = np.load(pathTMP)
    xsB, ysB = [dataTMP[item] for item in ['xs', 'ys']]
    #
    # write mesh_B in txt
    pathTMP = os.path.join(pathFolderBathymetries, 'mesh_B.txt')
    ulises.MakeFolder(os.path.split(pathTMP)[0])
    fileTMP = open(pathTMP, 'w')
    for posB in range(len(xsB)):
        fileTMP.write('{:15.3f} {:15.3f}\n'.format(xsB[posB], ysB[posB]))
    fileTMP.close()
    #
    # plot mesh_B
    if verbosePlot:
        pathTMPPlot = os.path.join(pathFolderBathymetries, 'plots', 'mesh_B.png')
        ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
        plt.plot(xsB, ysB, 'g.', markersize=0.4)
        plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal')
        plt.tight_layout()
        plt.savefig(pathTMPPlot, dpi=100); plt.close()
    #
    for date in dates:
        #
        # obtain pathOut (here, to manage overwrite)
        pathOut = os.path.join(pathFolderScratch, 'B_bathymetries', '{:}_{:}_B.npz'.format(date, par['DMD_or_EOF'])) # IMP* (in scratch we keep DMD or EOF)
        if os.path.exists(pathOut) and not overwrite: # we avoid 'continue' to ensure that we produce the cheap results in the folder 'bathymetries'
            print('... bathymetry for date {:} was already computed'.format(date)) 
            #
            # load zbsB and ezbsB
            dataTMP = np.load(pathOut)
            zbsB, ezbsB = [dataTMP[item] for item in ['zbsB', 'ezbsB']]
        else:
            print('... computing bathymetry for date {:}'.format(date), end='', flush=True)
            #
            # run through videos for the date to load (w, k)
            xsKW, ysKW, gammasKW, meanGsKW, wsKW, ztsKW = [np.asarray([]) for item in range(6)] # will not contain NaNs (IMP*)
            for video in videos4dates[date]:
                #
                # load mesh_K
                dataTMP = np.load(os.path.join(pathFolderScratch, video, 'mesh_K.npz'))
                xsK, ysK, zsK = [dataTMP[item] for item in ['xs', 'ys', 'zs']]; assert np.std(zsK) < 1.e-3
                #
                # load wavenumbers at mesh_K
                if not os.path.exists(os.path.join(pathFolderScratch, video, 'K_wavenumbers')): continue
                fns = sorted([item for item in os.listdir(os.path.join(pathFolderScratch, video, 'K_wavenumbers')) if item.endswith(par['DMD_or_EOF']+'_K.npz')])
                for fn in fns:
                    dataTMP = np.load(os.path.join(pathFolderScratch, video, 'K_wavenumbers', fn))
                    w, T, ks, meanGs, stdGs, RKs = [dataTMP[item] for item in ['w', 'T', 'ks', 'meanGs', 'stdGs', 'RKs']] # ks, meanGs and stdGs are len(xsK) x len(RKs)
                    for posRK in range(len(RKs)):
                        #
                        # obtain control variables
                        gs0 = stdGs[:, posRK] # can contain NaNs
                        gs1 = np.abs(w ** 2 / (grav * ks[:, posRK]) - meanGs[:, posRK]) # can contain NaNs
                        #
                        # obtain and add valid positions
                        poss = np.where((np.invert(np.isnan(gs0))) & (np.invert(np.isnan(gs1))) & (gs0 <= par['stdGammaC']) & (gs1 <= par['stdGammaC']))[0] # IMP*
                        xsKW, ysKW = np.append(xsKW, xsK[poss]), np.append(ysKW, ysK[poss])
                        gammasKW, meanGsKW = np.append(gammasKW, w ** 2 / (grav * ks[poss, posRK])), np.append(meanGsKW, meanGs[poss, posRK])
                        wsKW, ztsKW = np.append(wsKW, w*np.ones(len(poss))), np.append(ztsKW, np.mean(zsK)*np.ones(len(poss)))
            #
            # obtain zbsB and ezbsB
            zbsB, ezbsB = [np.empty(len(xsB)) for item in range(2)]; zbsB[:], ezbsB[:] = np.NaN, np.NaN # initialized as NaNs
            if len(xsKW) == 0:
                print(' fail') # will write NaNs
            else:
                RBs = np.empty(len(xsB)); RBs[:] = np.NaN
                for posB in range(len(xsB)):
                    #
                    # obtain gammas, ws and zts around
                    dsToKWTMP = np.sqrt((xsKW - xsB[posB]) ** 2 + (ysKW - ysB[posB]) ** 2)
                    poss0 = np.where(dsToKWTMP <= 1.1 * np.min(dsToKWTMP) + 1.e-3)[0] # one point in space, but several realizations
                    RB = max([par['cRadius_B'] * np.mean(2. * np.pi * grav * meanGsKW[poss0] / wsKW[poss0] ** 2), 2.1 * par['delta_K']])
                    possInKW = np.where(dsToKWTMP <= RB)[0]
                    if len(possInKW) < 10: continue # IMP*
                    gammasTMP, wsTMP, ztsTMP = [item[possInKW] for item in [gammasKW, wsKW, ztsKW]]
                    #
                    # obtain zb through RANSAC
                    possGood, zb = [], np.NaN
                    for zbH in np.arange(np.min(ztsTMP)-par['max_depth'], np.max(ztsTMP)-par['min_depth'], 0.05): # IMP*
                        gammasH = wsTMP ** 2 / (grav * ulises.WGH2KOneStep(wsTMP, grav, np.clip(ztsTMP - zbH, par['min_depth'], 1.e+3))) # IMP*
                        gammasH[np.where(ztsTMP - zbH <= par['min_depth'])[0]] = np.inf # IMP*
                        possGoodH = np.where(np.abs(gammasH - gammasTMP) < par['stdGammaC'])[0]
                        if len(possGoodH) > len(possGood):
                            possGood, zb = [copy.deepcopy(item) for item in [possGoodH, zbH]]
                    if len(possGood) < 10: continue # IMP*
                    #
                    # obtain zb through the minimization for RANSAC good positions
                    theArgs = {'gammas':gammasTMP[possGood], 'ws':wsTMP[possGood], 'zts':ztsTMP[possGood], 'g':grav}
                    zb = optimize.minimize(GoalFunction220622, np.asarray([zb]), args=(theArgs)).x[0] # IMP*
                    zbsB[posB], RBs[posB] = zb, RB
                #
                # obtain ezbsB and ensure zbsB is NaN if ezbsB is NaN
                for posB in range(len(xsB)):
                    if np.isnan(zbsB[posB]): continue # IMP* ezbsB is NaN
                    dsToBTMP = np.sqrt((xsB - xsB[posB]) ** 2 + (ysB - ysB[posB]) ** 2)
                    possTMP = np.where(dsToBTMP <= RBs[posB])[0]
                    if len(possTMP) < 3: continue # IMP* ezbsB is NaN
                    ezbsB[posB] = np.std(zbsB[possTMP]) # IMP* std (this is more demanding than using nanstd)
                zbsB[np.where(np.isnan(ezbsB))[0]] = np.NaN # IMP* (two lines above)
                print(' success')
            #
            # write the results in npz
            ulises.MakeFolder(os.path.split(pathOut)[0])
            np.savez(pathOut, zbsB=zbsB, ezbsB=ezbsB)
        #
        # write the results in txt
        pathTMP = os.path.join(pathFolderBathymetries, '{:}_B.txt'.format(date))
        ulises.MakeFolder(os.path.split(pathTMP)[0])
        fileTMP = open(pathTMP, 'w')
        for posB in range(len(xsB)):
            fileTMP.write('{:9.4f} {:9.4f}\n'.format(zbsB[posB], ezbsB[posB]))
        fileTMP.close()
        #
        # interpolate the groundTruth bathymetry
        pathGTB = os.path.join(pathFolderData, 'groundTruth', '{:}_GT_xyz.txt'.format(date))
        if os.path.exists(pathGTB):
            #
            # write and/or load the interpolated groundTruth bathymetry in npz
            pathTMP = os.path.join(pathFolderScratch, 'B_bathymetries',  '{:}_GT_B.npz'.format(date))
            if os.path.exists(pathTMP):
                dataTMP = np.load(pathTMP)
                zbsGTInB = dataTMP['zbsGTInB']
            else:
                dataTMP = np.asarray(ulises.ReadRectangleFromTxt(pathGTB, {'c1':3, 'valueType':'float'}))
                xsGT, ysGT, zsGT = [dataTMP[:, item] for item in range(3)]
                xsysGT, xsysB = np.transpose(np.asarray([xsGT, ysGT])), np.transpose(np.asarray([xsB, ysB]))
                vtx, wts = interp_weights220622(xsysGT, xsysB)
                zbsGTInB = interpolate220622(zsGT, vtx, wts)
                np.savez(pathTMP, zbsGTInB=zbsGTInB)
            #
            # write the interpolated groundTruth bathymetry in txt
            pathTMP = os.path.join(pathFolderBathymetries, '{:}_GT_B.txt'.format(date))
            fileTMP = open(pathTMP, 'w')
            for posB in range(len(xsB)):
                fileTMP.write('{:9.4f}\n'.format(zbsGTInB[posB]))
            fileTMP.close()
        else:
            zbsGTInB = np.NaN
        #
        # plot the results
        if verbosePlot:
            #
            pathTMPPlot0 = os.path.join(pathFolderScratch, 'plots', 'B_bathymetries', '{:}_{:}_B.png'.format(date, par['DMD_or_EOF'])) # IMP* (in scratch we keep DMD or EOF)
            ulises.MakeFolder(os.path.split(pathTMPPlot0)[0])
            pathTMPPlot1 = os.path.join(pathFolderBathymetries, 'plots', '{:}_B.png'.format(date))
            ulises.MakeFolder(os.path.split(pathTMPPlot1)[0])
            #
            plt.suptitle('date = {:}'.format(date))
            #
            plt.subplot(1, 3, 1)
            plt.title(r'$z_b$ [m]')
            plt.plot(np.min(xsB), np.min(ysB), 'w.'); plt.plot(np.max(xsB), np.max(ysB), 'w.')
            sct = plt.scatter(xsB, ysB, marker='o', c=zbsB, edgecolor='none', vmin=-par['max_depth'], vmax=-par['min_depth'], s=5, cmap='gist_earth')
            plt.colorbar(sct); plt.axis('equal'); plt.axis('off')
            #
            plt.subplot(1, 3, 2)
            plt.title('self error [m]')
            plt.plot(np.min(xsB), np.min(ysB), 'w.'); plt.plot(np.max(xsB), np.max(ysB), 'w.')
            sct = plt.scatter(xsB, ysB, marker='o', c=ezbsB, vmin=0, vmax=1, edgecolor='none', s=5, cmap='jet')
            plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal'); plt.axis('off')
            #
            plt.subplot(1, 3, 3)
            plt.title('relative error [-]')
            plt.plot(np.min(xsB), np.min(ysB), 'w.'); plt.plot(np.max(xsB), np.max(ysB), 'w.')
            sct = plt.scatter(xsB, ysB, marker='o', c=(zbsB-zbsGTInB)/np.abs(zbsGTInB), vmin=-1, vmax=1, edgecolor='none', s=5, cmap='seismic_r')
            plt.colorbar(sct); plt.xlabel(r'$x$ [m]'); plt.ylabel(r'$y$ [m]'); plt.axis('equal'); plt.axis('off')
            #
            plt.tight_layout()
            plt.savefig(pathTMPPlot0, dpi=100)
            plt.savefig(pathTMPPlot1, dpi=100)
            plt.close()
    #
    return None
#
def PerformKalman(pathFolderData, pathFolderBathymetries, verbosePlot): # last read 2022-10-28
    #
    # load parameters
    pathTMP = os.path.join(pathFolderData, 'parameters.json')
    with open(pathTMP, 'r') as f:
        par = json.load(f)
    #
    # read bathymetry mesh
    pathTMP = os.path.join(pathFolderBathymetries, 'mesh_B.txt')
    dataTMP = np.loadtxt(pathTMP)
    xsB, ysB = [dataTMP[:, item] for item in range(2)]
    #
    # read bathymetry filenames
    fns = sorted([fn for fn in os.listdir(pathFolderBathymetries) if fn.endswith('B.txt') and len(fn) == 18 and int(par['Kalman_ini']) <= int(fn[0:12]) <= int(par['Kalman_fin'])])
    #
    # perform Kalman filter
    Ptts, zbs, ts = [np.empty(len(xsB)) for item in range(3)]; Ptts[:], zbs[:], ts[:] = np.NaN, np.NaN, np.NaN # initialized as NaNs
    for fn in fns:
        print('... aggregating {:}'.format(fn))
        datenumH = ulises.Date2Datenum(fn[0:12]+'0000000')
        #
        # load bathymetry for fn
        dataTMP = np.loadtxt(os.path.join(pathFolderBathymetries, fn))
        zbsBH, ezbsBH = [dataTMP[:, item] for item in range(2)]
        #
        # update Kalman bathymetry
        for posB in range(len(xsB)):
            if np.isnan(zbs[posB]):
                Ptts[posB] = 0.
                zbs[posB] = zbsBH[posB]
            else:
                if np.isnan(zbsBH[posB]) or np.isnan(ezbsBH[posB]): continue
                Ptt1 = Ptts[posB] + (par['var_per_day'] * (datenumH - ts[posB])) ** 2 # p dimension [L^2]
                Kt = Ptt1 / (Ptt1 + ezbsBH[posB] ** 2) # dimension [-]
                Ptts[posB] = (1. - Kt) * Ptt1 # dimension [L^2]
                zbs[posB] = zbs[posB] + Kt * (zbsBH[posB] - zbs[posB])
            ts[posB] = datenumH # does not apply if the above 'continue' rules
        #
        # write the results in txt
        pathTMP = os.path.join(pathFolderBathymetries, fn.replace('B.txt', 'B_Kalman.txt'))
        ulises.MakeFolder(os.path.split(pathTMP)[0])
        fileTMP = open(pathTMP, 'w')
        for posB in range(len(xsB)):
            fileTMP.write('{:9.4f} {:9.4f}\n'.format(zbs[posB], Ptts[posB]))
        fileTMP.close()
        #
        # plot the results
        if verbosePlot:
            pathTMP = os.path.join(pathFolderBathymetries, fn.replace('B.txt', 'GT_B.txt'))
            if os.path.exists(pathTMP):
                zbsGTInB = np.loadtxt(pathTMP)
            else:
                zbsGTInB = np.empty(len(zbs)); zbsGTInB[:] = np.NaN
            #
            pathTMPPlot = os.path.join(pathFolderBathymetries, 'plots', fn.replace('B.txt', 'B_Kalman.png'))
            ulises.MakeFolder(os.path.split(pathTMPPlot)[0])
            #
            plt.suptitle('date = {:}'.format(fn[0:12]))
            #
            plt.subplot(1, 3, 1)
            plt.title(r'$z_b$ [m]')
            plt.plot(np.min(xsB), np.min(ysB), 'w.'); plt.plot(np.max(xsB), np.max(ysB), 'w.')
            sct = plt.scatter(xsB, ysB, marker='o', c=zbs, edgecolor='none', vmin=-par['max_depth'], vmax=-par['min_depth'], s=5, cmap='gist_earth')
            plt.colorbar(sct); plt.axis('equal'); plt.axis('off')
            #
            plt.subplot(1, 3, 2)
            plt.title(r'$\sqrt{P}$ [m]')
            plt.plot(np.min(xsB), np.min(ysB), 'w.'); plt.plot(np.max(xsB), np.max(ysB), 'w.')
            auxs = np.sqrt(Ptts)
            auxs[np.where(np.isnan(zbs))[0]] = np.NaN
            sct = plt.scatter(xsB, ysB, marker='o', c=auxs, vmin=0, vmax=0.5, edgecolor='none', s=5, cmap='jet')
            plt.colorbar(sct); plt.axis('equal'); plt.axis('off')
            #
            plt.subplot(1, 3, 3)
            plt.title('relative error [-]')
            plt.plot(np.min(xsB), np.min(ysB), 'w.'); plt.plot(np.max(xsB), np.max(ysB), 'w.')
            sct = plt.scatter(xsB, ysB, marker='o', c=(zbs-zbsGTInB)/np.abs(zbsGTInB), vmin=-1, vmax=1, edgecolor='none', s=5, cmap='seismic_r')
            plt.colorbar(sct); plt.axis('equal'); plt.axis('off')
            #
            plt.tight_layout()
            plt.savefig(pathTMPPlot, dpi=100)
            plt.close('all')
    #
    return None
