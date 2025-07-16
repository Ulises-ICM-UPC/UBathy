# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ by Gonzalo Simarro and Daniel Calvete
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
import os
import sys
#
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # WATCH OUT
#
sys.path.insert(0, 'ubathy')
import ubathy as ubathy
#
# ~~~~~~ user data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
pathFldMain = 'example'
#
# ~~~~~~ main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
if True:  # inform
    #
    ubathy.Inform_UBathy(pathFldMain, 0)
#
if True:  # extract frames
    #
    print('\nStep 1 out of 6: Extract frames from videos')
    ubathy.Video2Frames(pathFldMain)
#
if True:  # create meshes
    #
    print('\nStep 2 out of 6: Create the meshes')
    ubathy.CreateMeshes(pathFldMain)
#
if True:  # decompose videos
    #
    print('\nStep 3 out of 6: Decompose the videos')
    ubathy.ObtainWAndModes(pathFldMain)
#
if True:  # obtain wavenumbers
    #
    print('\nStep 4 out of 6: Obtain the wavenumbers')
    ubathy.ObtainK(pathFldMain)
#
if True:  # obtain bathymetry
    print('\nStep 5 out of 6: Obtain the bathymetry')
    ubathy.ObtainB(pathFldMain)
#
if True:  # filter bathymetry
    print('\nStep 6 out of 6: Filter the bathymetry (Kalman)')
    ubathy.PerformKalman(pathFldMain)
#
if True:  # inform
    #
    ubathy.Inform_UBathy(pathFldMain, 1)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#