#'''
# Created on 2022 by Gonzalo Simarro and Daniel Calvete
#'''
#
import itertools
import numpy as np
import os
import sys
#
sys.path.insert(0, 'ubathy')
import ubathy as ubathy # WATCH OUT
#
pathFolderMain = 'example'
assert os.path.exists(pathFolderMain)
#
#''' --------------------------------------------------------------------------
# Extraction of the videos
#''' --------------------------------------------------------------------------
#
pathFolderVideos = os.path.join(pathFolderMain, 'videos')
listOfVideos = [] # if [], takes all the available
FPS = 0.0 # if 0.0, FPS is the video time resolution
overwrite = False
#
print('Extraction of the videos')
ubathy.Video2Frames(pathFolderVideos, listOfVideos, FPS, overwrite)
#
#''' --------------------------------------------------------------------------
# Creation of the meshes
#''' --------------------------------------------------------------------------
#
pathFolderData = os.path.join(pathFolderMain, 'data')
pathFolderVideos = os.path.join(pathFolderMain, 'videos')
pathFolderScratch = os.path.join(pathFolderMain, 'scratch')
listOfVideos = [] # if [], takes all the available
overwrite = False
verbosePlot = True
#
print('Creation of the meshes')
ubathy.CreateMeshes(pathFolderData, pathFolderVideos, pathFolderScratch, listOfVideos, overwrite, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Decomposition of the videos
#''' --------------------------------------------------------------------------
#
pathFolderData = os.path.join(pathFolderMain, 'data')
pathFolderVideos = os.path.join(pathFolderMain, 'videos')
pathFolderScratch = os.path.join(pathFolderMain, 'scratch')
listOfVideos = [] # if [], takes all the available
overwrite = False
verbosePlot = True
#
print('Decomposition of the videos')
ubathy.ObtainWAndModes(pathFolderData, pathFolderVideos, pathFolderScratch, listOfVideos, overwrite, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Obtaining the wavenumbers
#''' --------------------------------------------------------------------------
#
pathFolderData = os.path.join(pathFolderMain, 'data')
pathFolderScratch = os.path.join(pathFolderMain, 'scratch')
listOfVideos = []
overwrite = False
verbosePlot = True
#
print('Obtaining the wavenumbers')
ubathy.ObtainK(pathFolderData, pathFolderScratch, listOfVideos, overwrite, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Obtaining the bathymetry
#''' --------------------------------------------------------------------------
#
pathFolderData = os.path.join(pathFolderMain, 'data')
pathFolderScratch = os.path.join(pathFolderMain, 'scratch')
pathFolderBathymetries = os.path.join(pathFolderMain, 'bathymetries')
overwrite = False
verbosePlot = True
#
print('Obtaining the bathymetry')
ubathy.ObtainZb(pathFolderData, pathFolderScratch, pathFolderBathymetries, overwrite, verbosePlot)
#
