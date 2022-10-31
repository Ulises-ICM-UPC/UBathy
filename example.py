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
pathFolderMain = 'example' # USER DEFINED
#
#''' --------------------------------------------------------------------------
# Extraction of the videos
#''' --------------------------------------------------------------------------
#
pathFolderVideos = os.path.join(pathFolderMain, 'videos') # USER DEFINED
listOfVideos = [] # if [], takes all the available
FPS = 0.0 # if 0.0, FPS is the video time resolution
overwrite = False # USER DEFINED
#
print('Extraction of the videos')
ubathy.Video2Frames(pathFolderVideos, listOfVideos, FPS, overwrite)
#
#''' --------------------------------------------------------------------------
# Creation of the meshes
#''' --------------------------------------------------------------------------
#
pathFolderData = os.path.join(pathFolderMain, 'data')
#pathFolderVideos = os.path.join(pathFolderMain, 'videos') # USER DEFINED
pathFolderScratch = os.path.join(pathFolderMain, 'scratch')
listOfVideos = [] # if [], takes all the available
overwrite = False # USER DEFINED
verbosePlot = True # USER DEFINED
#
print('Creation of the meshes')
ubathy.CreateMeshes(pathFolderData, pathFolderVideos, pathFolderScratch, listOfVideos, overwrite, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Decomposition of the videos
#''' --------------------------------------------------------------------------
#
#pathFolderData = os.path.join(pathFolderMain, 'data') # USER DEFINED
#pathFolderVideos = os.path.join(pathFolderMain, 'videos') # USER DEFINED
#pathFolderScratch = os.path.join(pathFolderMain, 'scratch') # USER DEFINED
#listOfVideos = [] # if [], takes all the available
overwrite = False # USER DEFINED
verbosePlot = True # USER DEFINED
#
print('Decomposition of the videos')
ubathy.ObtainWAndModes(pathFolderData, pathFolderVideos, pathFolderScratch, listOfVideos, overwrite, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Obtaining the wavenumbers
#''' --------------------------------------------------------------------------
#
#pathFolderData = os.path.join(pathFolderMain, 'data') # USER DEFINED
#pathFolderScratch = os.path.join(pathFolderMain, 'scratch') # USER DEFINED
#listOfVideos = [] # if [], takes all the available # USER DEFINED
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
#pathFolderData = os.path.join(pathFolderMain, 'data') # USER DEFINED
#pathFolderScratch = os.path.join(pathFolderMain, 'scratch') # USER DEFINED
#pathFolderBathymetries = os.path.join(pathFolderMain, 'bathymetries') # USER DEFINED
overwrite = False # USER DEFINED
verbosePlot = True # USER DEFINED
#
print('Obtaining the bathymetry')
ubathy.ObtainB(pathFolderData, pathFolderScratch, pathFolderBathymetries, overwrite, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Filtering (Kalman) the bathymetries
#''' --------------------------------------------------------------------------
#
#pathFolderData = os.path.join(pathFolderMain, 'data') # USER DEFINED
pathFolderBathymetries = os.path.join(pathFolderMain, 'bathymetries_Kalman') # USER DEFINED
verbosePlot = True
#
print('Filtering (Kalman) the bathymetry')
ubathy.PerformKalman(pathFolderData, pathFolderBathymetries, verbosePlot)
