#'''
# Created on 2022 by Gonzalo Simarro and Daniel Calvete
#'''
#
import os
import sys
#
sys.path.insert(0, 'ubathy')
import ubathy as ubathy
#
pathFolderMain = 'example'
#
assert os.path.exists(pathFolderMain)
#
pathFolderData = os.path.join(pathFolderMain, 'data')
pathFolderVideos = os.path.join(pathFolderMain, 'videos')
pathFolderScratch = os.path.join(pathFolderMain, 'scratch')
pathFolderBathymetries = os.path.join(pathFolderMain, 'bathymetries')
#
#''' --------------------------------------------------------------------------
# Extraction of the videos
#''' --------------------------------------------------------------------------
#
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
overwrite = False
verbosePlot = True
#
print('Obtaining the bathymetry')
ubathy.ObtainB(pathFolderData, pathFolderScratch, pathFolderBathymetries, overwrite, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Filtering (Kalman) the bathymetries
#''' --------------------------------------------------------------------------
#
pathFolderBathymetries = os.path.join(pathFolderMain, 'bathymetries_Kalman')
#
verbosePlot = True
#
print('Filtering (Kalman) the bathymetry')
ubathy.PerformKalman(pathFolderData, pathFolderBathymetries, verbosePlot)
#
