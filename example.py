# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ by Gonzalo Simarro and Daniel Calvete
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
import os
import sys
#
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # WATCH OUT
#
sys.path.insert(0, 'udrone')
import udrone as udrone  # type: ignore
#
# ~~~~~~ user data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
pathFldMain = 'example'
#
# ~~~~~~ main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
if True:  # inform
    #
    udrone.Inform_UDrone(pathFldMain, 0)
#
if True:  # extract frames
    #
    print('\nStep 1 out of 5: Extract frames from video')
    udrone.Video2Frames(pathFldMain)
#
if True:  # calibrate basis
    #
    print('\nStep 2 out of 5: Calibrate the basis')
    udrone.CalibrationOfBasisImages(pathFldMain)  # always smart-overwrite
    #
    print('\nStep 3 out of 5: Calibrate the basis forcing a unique set of intrinsic parameters')
    udrone.CalibrationOfBasisImagesConstantIntrinsic(pathFldMain)  # always smart-overwrite
#
if True:  # autocalibrate frames
    #
    print('\nStep 4 out of 5: (Auto)calibrate the video frames')
    udrone.AutoCalibrationOfFramesViaGCPs(pathFldMain)
#
if True:  # generate planviews and timestack
    #
    print('\nStep 5 out of 5: Generate of the planviews and the timestacks')
    udrone.PlanviewsFromImages(pathFldMain)
#
if True:  # inform
    #
    udrone.Inform_UDrone(pathFldMain, 1)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
