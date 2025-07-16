# UDrone

`UDrone` is an open source software written in Python for automatic image calibration of drone video images from a set of images that are manually calibrated.

### Description
The calibration algorithm assumes that the intrinsic parameters of the camera remain unchanged while the extrinsic parameters (position and orientation) vary. The result of the process is the common intrinsic camera parameters for all images and the extrinsic parameters for each individual images extracted from a video. In addition, for each image a planview can be generated and, for the video, planviews for mean (_timex_) and sigma (_variance_) of the video images and timestacks are generated as well. The development of this software is suitable for processing videos obtained from moving cameras such as those acquired from drones. Details on the algorithm and methodology are described in
> *Simarro, G.; Calvete, D.; Plomaritis, T.A.; Moreno-Noguer, F.; Giannoukakou-Leontsini, I.; Montes, J.; Durán, R. The Influence of Camera Calibration on Nearshore Bathymetry Estimation from UAV Videos. Remote Sens. 2021, 13, 150. https://doi.org/10.3390/rs13010150*

The automatic calibration process consists of the following steps:

 1. [Video setup](#video-setup)
 2. [Intrinsic camera calibration](#basis-calibration)
 3. [Automatic frame calibration](#automatic-calibration)
 
Further, `UDrone` generates planviews for the calibrated images, mean and sigma images for the planviews and timestacks of the video:

 4. [Planview and timestack generation](#planviews-and-timestacks)
 5. [UBathy compatibility](#ubathy)


### Requirements and project structure
To run the software it is necessary to have Python (3.9) and install the following dependencies:
- cv2 (4.2.0)
- numpy (1.19.5)
- scipy (1.3.3)
- matplotlib (3.3.4)

In parenthesis we indicate the version with which the software has been tested. It is possible that it works with older versions. 

The structure of the project is the following:
* `example.py`
* `example_notebook.py`
* **`udrone`**
  * `udrone.py`
  * `ulises_udrone.py`
* **`example`**
  * **`data`**
    * `parameters.json`
    * `planviews_xy.txt`
    * `timestacks_xyz.txt`
    * `videoFilename.mp4` (.avi or .mov)
    * **`basis`**
      * `<anyname01>.png`
      * `<anyname01>cdg.txt`
      * `<anyname01>cdh.txt`
      * . . .
    * **`frames`**
      * `videoFilename_000000000000.png`** (.jpg or .jpeg)
      * . . .
  * **`output`**
    * **`numerics`**
      * **`planviews`**
        * `planviews_crxyz.txt`
      * **`timestacks`**
        * `timestacks_rt.txt`
        * `timestack_t01_cxyz.txt`
        * . . .
    * **`plots`**
      * `mean.png`
      * `sigma.png`
      * **`planviews`**
        * `videoFilename_000000000000plw.png`
        * . . .
      * **`timestacks`**
        * `timestack_t01.png`
        * . . .
    * **`ubathy`**
      * `planviews_crxyz.txt`
      * **`planviews`**
        * `videoFilename_000000000000plw.png`
        * . . .
  * **`scratch`**
    * **`frames`**
      * `videoFilename_000000000000.png`** (.jpg or .jpeg)
      * . . .
    * **`numerics`**
      * **`autocalibrations`**
        * `videoFilename_000000000000cal.txt`
        * `videoFilename_000000000000cdg.txt`
        * `videoFilename_000000000000cdh.txt`
        * . . .
      * **`autocalibrations_filtered`**
        * `videoFilename_000000000000cal.txt`
        * . . .
      * **`calibration_basis`**
        * `<anyname01>cal.txt`
        * `<anyname01>cal0.txt`
        * . . .
    * **`plots`**
      * `extrinsic_parameters.jpg`
      * **`autocalibrations`**
        * `videoFile_000000000000cal.jpg`
        * . . .
      * **`autocalibrations_filtered`**
        * `videoFile_000000000000cal.jpg`
        * . . .
      * **`calibration_basis`**
        * `<anyname01>cal.jpg`
        * `<anyname01>cal0.jpg`
        * . . .
      * **`planviews`**
        * `videoFilename_000000000000.jpg`
        * . . .
      * **`timestacks`**
        * **`timestack_t01`**
          * `videoFilename_000000000000.jpg`
          * . . .

The local modules of `UDrone` are located in the **`udrone`** folder. Folder **`scratch`** contains auxiliary files. Once completed, the files can be deleted.  

To run a demo in folder **`example`** experienced users can run the `example.py` file in a terminal. Alternatively we provide the file `example_notebook.ipynb` to be used in a Jupyter Notebook. In that case, import modules and set the main path of the example:


```python
import sys
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, 'udrone')
import udrone as udrone
pathFldMain = 'example'
```

Adjust the parameter values in the `parameters.json` file located in the **`data`** folder. To check the current values, run:


```python
udrone.Inform_UDrone(pathFldMain, 0)
```

As general control parameter, the `overwrite_outputs` parameter controls whether results from previous runs are recomputed (`true`) or preserved (`false`). To facilitate the verification of the results, auxiliary images can be generated. Set parameter `generate_scratch_plots = true`, and to `false` otherwise.


## Video setup

If the individual frames of the video are available, they must be placed in the folder **`data/frames`**. `UDrone`  handles `PNG` (recommended) and `JPEG` image formats.

### Run video extraction
If the video has been stored in `MP4`, `AVI` or `MOV` image format, if should be placed in **`data`** (e.g. `<videoFilename>.mp4`). 

Set the extraction rate of the frames in file `parameters.json`:

| Object-name | Description | Suggested value | Units |
|:--|:--:|:--:|:--:|
| `frame_rate` | Extraction framerate | _2.0_ | _1/s_ |
Set `frame_rate=0` to extract all frames from the video.

Run the code to extract frames from de video:


```python
udrone.Video2Frames(pathFldMain)
```

As a result, images of each extracted frame `<frame>.png` are generated in the **`scratch/frames`** folder with the format `<videoFilename>_<milliseconds>.png`.


## Basis calibration
The intrinsic parameters of the camera are determined by a manual calibration of selected frames that will also be used in the automatic calibration of all the extracted frames. To manually calibrate the frames selected for the basis, placed in the folder **`data/basis`**, it is necessary that each image `<basisFrame>.png` is supplied with a file containing the Ground Control Points (GCP) and, optionally, the Horizon Points (HP). The structure of each of these files is the following:
* `<basisFrame>cdg.txt`: For each GCP one line with (minimum 6)
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate`
* `<basisFrame>cdh.txt`: For each HP one line with (minimum 3)
>`pixel-column`, `pixel-row`

Quantities must be separated by at least one blank space between them and the last record should not be continued with a newline (return).

To generate `<basisImage>cdg.txt` and `<basisImage>cdh.txt` files the [UClick](https://github.com/Ulises-ICM-UPC/UClick) software is available.

### Run basis calibration
Set the value of maximum error allowed for the basis calibration:

| Object-name | Description | Suggested value | Units |
|:--|:--:|:--:|:--:|
| `max_reprojection_error_px` | Critical reprojection pixel error | _5._ | _pixel_ |


Select an intrinsic camera calibration model setting `camera_lens_model`:

|  |  |  |  |
|:--|:--:|:--:|:--:|
| Object-name-value | "parabolic" | "quartic" | "full" |
| _Lens radial distortion_ | parabolic | parabolic + quartic | parabolic + quartic |
| _Lens tangential distortion_ | no | no | yes |
| _Square pixels_ | yes | yes | no |
| _Decentering_ | no | no | yes |

The `parabolic` model is recommended by default, unless the images are highly distorted.

To facilitate the verification that the GCPs have been correctly selected in each image of the basis, images showing the GCPs and HPs (black), the reprojection of GCPs (yellow) and the horizon line (yellow) on the images can be generated if `generate_scratch_plots = True`. Images (`<basisFrame>cal.jpg`) will be placed on **`scratch/plots/calibration_basis`** folder.

Run the initial calibration algorithm for each image of the basis:


```python
udrone.CalibrationOfBasisImages(pathFldMain)
```

In case that the reprojection error of a GCP is higher than the error `max_reprojection_error_px` for a certain image `<basisFrame>`, a message will appear suggesting to re-run the calibration of the basis or to modify the values or to delete points in the file `<basisFrame>cdg.txt`. If the calibration error of an image exceeds the error `max_reprojection_error_px` the calibration is given as _failed_. Consider re-run the calibration of the basis or verify the GPCs and HPs.

Then, run the algorithm to obtain the optimal intrinsic parameters of the camera.


```python
udrone.CalibrationOfBasisImagesConstantIntrinsic(pathFldMain)
```

As a result of the calibration, the calibration file `<basisFrame>cal.txt` is generated in the **`scratch/numerics/calibration_basis`** directory for each of the frames. This file contains the following parameters:

| Magnitudes | Variables | Units |
|:--|:--:|:--:|
| Camera position coordinates | `xc`, `yc`, `zc` | _m_ |
| Camera orientation angles | `ph`, `sg`, `ta` | _rad_ |
| Lens radial distortion (parabolic, quartic) | `k1a`, `k2a` | _-_ |
| Lens tangential distortion (parabolic, quartic) | `p1a`, `p2a` | _-_ |
| Pixel size | `sca`, `sra` | _-_ |
| Decentering | `oc`, `or` | _pixel_ |
| Image size | `nc`, `nr` | _pixel_ |
| Calibration error | `errorT`| _pixel_ |

The different calibration files `<basisFrame>cal.txt` differ only in extrinsic paramaters (`xc`, `yc`, `zc`, `ph`, `sg`, `ta`) and the calibration error (`errorT`). A `<basisFrame>cal0.txt` file with the manual calibration parameters for each frame of the basis will also have been generated.

## Automatic calibration

In this step, each video frame (`<frame>.png`) will be automatically calibrated. In a first phase, pairs of matching features between the images of the basis and the frames are identified using either the ORB or SIFT algorithm. Optionally, a horizon detection algorithm can also be applied to the frames. The configuration parameters for this process are defined in the `parameters.json` file:

| Object-name | Description | Suggested value | 
|:--|:--:|:--:|
| `feature_detection_method` | Feature Matching method | `"sift"` |
| `max_features` | Maximum number of features | _2000_ |
| `enable_horizon_detection` | Allow automatic detection of the horizon | `true` |

Then, to refine the calibration and eliminate outliers, a temporal filtering step is applied. This process is controlled by the size of the temporal window used for the filtering: 

| Object-name | Description| Suggested value | Units |
|:--|:--:|:--:|:--:|
| `outlier_filtering_window_sec` | Temporal filtering window | _2.0_ | _s_ |
Set `outlier_filtering_window_sec = 0` to disable outlier filtering.

Run the algorithm to calibrate frames automatically:


```python
udrone.AutoCalibrationOfFramesViaGCPs(pathFldMain)
```

In case autocalibration process fails, it is reported that the calibration of the frame `<frame>.png` is _not calibratable_. Increasing the basis frames can improve the calibration process. 

For each frame (`<frame>.png`), a corresponding calibration file (`<frame>cal.txt`) will be generated, following the same format as previously described. The unfiltered and filtered calibration files are stored in the folders **`scratch/numerics/autocalibrations`** and **`scratch/numerics/autocalibrations_filtered`**, respectively. A plot `extrinsic_parameters.jpg` with the values of calibrated extrinsic parameters for each frame is located in **`scratch/plots`**

To facilitate the verification that the GCPs have been correctly identified in each frame, images showing the reprojection of the GCPs can be generated  if `generate_scratch_plots = True`. Images (`<frame>cal.jpg`) will be placed on a **`scratch/plots/autocalibrations`** and **`scratch/plots/autocalibrations_filtered`** folders. 


## Planviews and timestacks

Once the frames have been calibrated, planviews can be generated. The region of the planview is the one delimited by the minimum area rectangle containing the points of the plane specified in the file `planviews_xy.txt` in the folder **`data`**. The planview image will be oriented so that the nearest corner to the point of the first of the file `planviews_xy.txt` will be placed in the upper left corner of the image. The structure of this file is the following:
* `planviews_xy.txt`: For each points one line with 
> `x-coordinate`, `y-coordinate`

A minimum number of three not aligned points is required. These points are to be given in the same coordinate system as the GCPs. Set `z_sea_level` (in _meters_) in `parameters.json` to specify the sea level for projecting the frames. 

To obtain time series of the pixel values of the frames along several paths in the space, a file with the coordinates of points along the paths must be provided. For each path, the values are obtained along the straight segments bounded by consecutive points in the file. The structure of this file, located in the folder **`data`**, is the following:
* `timestacks_xyz.txt`: For each point one line with
>  `path-label`, `x-coordinate`, `y-coordinate`, `z-coordinate`

A minimum number of two points is required for each path. These points are to be given in the same coordinate system as the GCPs.

The resolution of planviews and timestacks are fixed by:

| Object-name | Description| Suggested value | Units |
|:--|:--:|:--:|:--:|
| `ppm_for_planviews` | Planview resolution | _1.0_ | _pixels-per-meter_ |
| `ppm_for_timestacks` | Timestack resolution | _2.0_ | _pixels-per-meter_ |

Run the algorithm to generate the planviews:


```python
udrone.PlanviewsFromImages(pathFldMain)
```

As a result, for each of the calibrated frames `<frame>.png`, a planview `<frame>plw.png` will be placed in the folder **`output/plots/planviews`**. Note that objects outside the plane at height `z0` will show apparent displacements due to real camera movement. In the folder **`output/numerics/planviews`**, the file `planviews_crxyz.txt` will be located, containing the coordinates of the corner of the planviews images:
* `planviews_crxyz.txt`: For each corner one line with 
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate`

For each timestack-path, a `timestack_<path-label>.png` will be placed in the folder  **`output/plots/timestacks`**. In the folder **`output/numerics/timestacks`**, a file `timestacks_rt.txt` containing the filename (time) of each row of the timestack and a file for each timestack-path `timestack_<path-label>_cxyz.txt` containing the spatial coordinates of each column and will be located. The structure of these files are the following:
* `timestacks_rt.txt`: 
>`<frame>.png`, `pixel-row`
* `timestack_<path-label>_cxyz.txt`: 
>`pixel-column`, `x-coordinate`, `y-coordinate`, `z-coordinate`

If `include_gaps_in_timestack` is set to `true`, uncalibrated frames are represented as black rows in the time stack. If set to `false`, only calibrated frames are included, and the gaps are omitted

From the planview of each calibrated frame, time exposure (_timex_) and sigma images can be generated by computing the mean value (`mean.png`) and the standard deviation (`sigma.png`) of all images in the folder **`output/plots`**, respectively.

To check the planview domain and the paths of the timestack, in the folders **`scratch/plots/planviews`** and **`scratch/plots/timestacks/timestack_<path-label>`** the calibrated frames `<frame>.jpg` with the region of the planview and each of the paths are placed.


## UBathy

To enable later use of `UDrone` outputs in [UBathy](https://github.com/Ulises-ICM-UPC/UBasic), set `generate_ubathy_data` to `True` in `parameters.json`. This will generate the **`output/ubathy`** folder, which contains the `planview_crxyz.txt` file and the **`planviews`** subfolder with the `<frame>plw.png` images. These files can be directly exported to `UBathy`.


## Contact us

Are you experiencing problems? Do you want to give us a comment? Do you need to get in touch with us? Please contact us!

To do so, we ask you to use the [Issues section](https://github.com/Ulises-ICM-UPC/UDrone/issues) instead of emailing us.

## Contributions

Contributions to this project are welcome. To do a clean pull request, please follow these [guidelines](https://github.com/MarcDiethelm/contributing/blob/master/README.md)

## License

UDrone is released under a [AGPL-3.0 license](https://github.com/Ulises-ICM-UPC/UDrone/blob/master/LICENSE). If you use UDrone in an academic work, please cite:

    @Article{rs13010150,
      AUTHOR = {Simarro, Gonzalo and Calvete, Daniel and Plomaritis, Theocharis A. and Moreno-Noguer, Francesc and Giannoukakou-Leontsini, Ifigeneia and Montes, Juan and Durán, Ruth},
      TITLE = {The Influence of Camera Calibration on Nearshore Bathymetry Estimation from UAV Videos},
      JOURNAL = {Remote Sensing},
      VOLUME = {13},
      YEAR = {2021},
      NUMBER = {1},
      ARTICLE-NUMBER = {150},
      URL = {https://www.mdpi.com/2072-4292/13/1/150},
      ISSN = {2072-4292},
      DOI = {10.3390/rs13010150}
      }

    @Online{ulisesdrone, 
      author = {Simarro, Gonzalo and Calvete, Daniel},
      title = {UDrone},
      year = 2021,
      url = {https://github.com/Ulises-ICM-UPC/UDrone}
      }

