# UBathy

`UBathy` is an open source software written in Python for nearshore bathymetry estimation from videos of calibrated images and/or of geo-referenced planviews. 

### Description
The algorithm for bathymetry estimation is based on extracting wave modes from videos of nearshore surface wave propagation. These videos can be formed either from camera images, which must have been previously calibrated, or from geo-referenced planviews. For each wave mode extracted from the videos, the frequency and the spatially dependent wavenumbers are obtained. The frequencies and wavenumbers from different videos are used to estimate the bathymetry by adjusting the dispersion relation for linear surface waves. Bathymetries estimated at different times are finally aggregated using a Kalman filter to obtain the final bathymetries. The development of this software is suitable for Argus-type video monitoring stations and from moving cameras such as those acquired from drones. The calibration of these videos and the generation of planviews, which are necessary in the case of drones, can be done using the software [UBasic](https://github.com/Ulises-ICM-UPC/UBasic), [UCalib](https://github.com/Ulises-ICM-UPC/UCalib) or [UDrone](https://github.com/Ulises-ICM-UPC/UDrone). Details on the algorithm and methodology are described in
> *Simarro, G.; Calvete, D.; Luque, P.; Orfila, A.; Ribas, F. UBathy: A New Approach for Bathymetric Inversion from Video Imagery. Remote Sens. 2019, 11, 2722. https://doi.org/10.3390/rs11232722*

The bathymetry estimation process consists of the following steps:

 1. [Video setup](#video-setup)
 2. [Generation of meshes](#generation-of-meshes)
 3. [Mode decomposition](#mode-decomposition)
 4. [Wavenumber computation](#wavenumber-computation)
 5. [Bathymetry estimation](#bathymetry-estimation)
 
Finally, `UBathy` allows to aggregate bathymetries obtained at different times using a Kalman filter:

 6. [Kalman estimation](#kalman-estimation)


### Requirements and project structure
To run the software it is necessary to have Python3 (3.8) and install the following dependencies:
- cv2 (4.2.0)
- numpy (1.19.5)
- scipy (1.3.3)

In parenthesis we indicate the version with which the software has been tested. It is possible that it works with older versions. 

The structure of the project is the following:
* `example.py`
* `example_notebook.py`
* **`ubathy`**
  * `ubathy.py`
  * `ulises_ubathy.py`
* **`example`**
  * **`videos`**
    * `videoFilename01.mp4` (.avi or .mov)
    * **`videoFilename01`**
      * `videoFilename01_000000000000.png` (or .jpg)
      * . . .
    * . . .
  * **`data`**
    * `xy_boundary.txt`
    * `parameters.json`
    * `videos4dates.json`
    * `parametersKalman.json`
    * **`videoFilename01`**
      * `<anyname>cal.txt`
      * `<anyname>zs.txt`
      * or
      * `<anyname>crxyz.txt`
    * **`groundTruth`**
      * `date01_GT_xyz.txt`
      * . . . 
    * . . .
  * **`scratch`**
    * `mesh_Zb.npz`
    * **`videoFilename01`**
      * `mesh_M.npz`
      * `mesh_K.npz`
      * **`M_modes`**
        * `t<timeInVideo>_w<windowTime>_T<period>_<modeType>.npz`
        * . . . 
      * **`K_wavenumber`**
        * `t<timeInVideo>_w<windowTime>_T<period>_<modeType>_K.npz`
        * . . . 
    * . . .
    * **`Zb_bathymetries`**
      * `date01_<modeType>_Zb.npz`
      * `date01_GT_Zb.npz`
      * . . .
    * **`plots`**
      * `mesh_Zb.png`
      * **`videoFilename01`**
        * `mesh_M.png`
        * `mesh_M_inImage.png`
        * `mesh_K.png`
        * `mesh_K_inImage.png`
        * **`M_modes`**
          * `t<timeInVideo>_w<windowTime>_T<period>_<modeType>.png`
          * . . . 
        * **`K_wavenumber`**
          * `t<timeInVideo>_w<windowTime>_T<period>_<modeType>_K.png`
          * . . .
      * . . .
      * **`Zb_bathymetries`**
        * `date01_<modeType>_Zb.png`
        * . . .
  * **`bathymetries`**
    * `mesh_Zb.txt`
    * `date01_Zb.txt`
    * `date01_Zb_kalman.txt`
    * `date01_GT_Zb.txt`
    * . . .
    * **`plots`**
      * `mesh_Zb.png`
      * `date01_Zb.png`
      * `date01_Zb_kalman.png`
      * . . .


The local modules of `UBathy` are located in the **`ubathy`** folder.

To run a demo in folder **`example`** experienced users can run the `example.py` file in a terminal. Alternatively we provide the file `example_notebook.ipynb` to be used in a Jupyter Notebook. In that case, import modules and set the main path of the example:


```python
import sys
import os
sys.path.insert(0, 'ubathy')
import ubathy as ubathy
pathFolderMain = 'example'
```

Set also the folders where the videos (**`videos`**) and data files (**`data`**) are located, and the folders where the temporary data (**`scratch`**) and the estimated bathymetries (**`bathymetries`**) will be stored:


```python
pathFolderData = os.path.join(pathFolderMain, 'data')
pathFolderVideos = os.path.join(pathFolderMain, 'videos')
pathFolderScratch = os.path.join(pathFolderMain, 'scratch')
pathFolderBathymetries = os.path.join(pathFolderMain, 'bathymetries')
```

## Video setup
The bathymetric extraction algorithm is applied to videos stored in the **`videos`** folder. For each video `<videoFilename>` a folder with the same name must contain the frames with the format `<videoFilename>_<milliseconds>.png` (or `.jpg`). Depending on whether the frames come from calibrated camera images or from geo-referenced planviews, different files have to be provided.

#### Calibrated camera images
A calibration file `<anyname>cal.txt` of the camera must exist in the folder **`data/<videoFilename>`** with the following parameters:

| Magnitudes | Variables | Units |
|:--|:--:|:--:|
| Camera position coordinates | `xc`, `yc`, `zc` | _m_ |
| Camera orientation angles | `ph`, `sg`, `ta` | _rad_ |
| Lens radial distortion (parabolic, quartic) | `k1a`, `k2a` | _-_ |
| Lens tangential distortion (parabolic, quartic) | `p1a`, `p2a` | _-_ |
| Pixel size | `sc`, `sr` | _-_ |
| Decentering | `oc`, `or` | _pixel_ |
| Image size | `nc`, `nr` | _pixel_ |
| Calibration error | `errorT`| _pixel_ |

This file can be obtained using the [UBasic](https://github.com/Ulises-ICM-UPC/UBasic) software. In the same directory should also exist the file `<anyname>zs.txt` with the position of the water surface during the recording of the video with the following structure:
* `<anyname>zs.txt`: One line with
>`z-coordinate-free-surface`

The coordinates of the camera position `(xc, yc, zc)` and the position of the water surface `(zs)` are referenced to the same coordinate system in which the bathymetries are to be obtained.

#### Geo-referenced planviews
If the frames correspond to planviews, in the folder **`data/<videoFilename>`** there must exists the file `<anyname>_crxyz.txt` with the correspondence between the pixels of the planview and the coordinate system in which the bathymetry is going to be obtained. The structure of this file is the following:
* `<anyname>_crxyz.txt`: One line for each pixel
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate-free-surface`

On previous files, quantities must be separated by at least one blank space between them and the last record should not be continued with a newline (return).

### Run video extraction
If the video has been stored in `MP4`, `AVI` or `MOV` image format, the frames must be extracted and placed in the above mentioned folder **`videos/<videoFilename>`** before processing. To do so, set a list of the videos to be processed with the name of each `<videoFilename>` (i.e. `<videoFilename>.mp4`). In case you want to extract all the videos that are in **`videos`** provide an empty list (i.e. `[]`)


```python
listOfVideos = [] 
```

Set the extraction rate of the frames:

|  | Parameter | Suggested value | Units |
|:--|:--:|:--:|:--:|
| Extraction framerate | `FPS` | _2.0_ | _1/s_ |
Set FPS=0 to extract all frames from the video.


```python
FPS = 0.00
```

In the case that a `<videoFilename>` has already been extracted, set `overwrite = True` to extract it again and to `False` otherwise. Run the code to extract frames from de videos:


```python
overwrite = False
#
ubathy.Video2Frames(pathFolderVideos, listOfVideos, FPS, overwrite)
```

As a result, for each `<videoFilename>` a folder **`videos/<videoFilename>`** containing the frames with the format `<videoFilename>_<milliseconds>.png` is generated.

## Generation of meshes
In general, the meshes for the extraction of wave modes, the calculation of wavenumbers and bathymetries are performed on the basis of a regular mesh of equilateral triangles in an x-y plane domain. The mesh for the extraction of the modes is adjusted according to the position of the pixels of the images of each video `<videoFilename>` if the frames come from calibrated camera images. For geo-referenced planviews, the mesh extraction of wave modes corresponds to spatial coordinates of the pixels specified in the file `<anyname>_crxyz.txt`.

To generate the meshes it is necessary to provide in the folder **`data`** a file `xy_boundary.txt` with the coordinates of the polygon vertices of the region in which the bathymetry is going to be extracted and to set in the file `parameters.json` the spacing between the nodes of the different meshes.

The structure of each of these files is the following:
* `xy_boundary.txt`: For each vertex point one line with (minimum 3)
>`x-coordinate`, `y-coordinate`

Quantities must be separated by at least one blank space between. These points are to be given in the same coordinate system in which the bathymetry is going to be obtained
* `parameters.json`: Set the values of node distance for ech mesh

| Object-name | Description | Suggested value | Units | 
|:--|:--:|:--:|:--:|
| `delta_M` | approx. mode-node distance | 2.5 | _m_ |
| `delta_K` | wavenumber-node distance | 5.0 | _m_ |
| `delta_Zb` | bathymetry-node distance | 10.0| _m_ |

Note that `delta_M` is not used for videos of geo-referenced planviews.

### Run meshes generation
To verify the arrangement of the mesh, images of the meshes and the position of the grid nodes relative to the video frames can be generated. Set parameter `verbosePlot = True`, and to `False` otherwise. In the case that the meshes have already been generated, set `overwrite = True` to generate them again and to `False` otherwise. Set the values of these control parameters and run the code to generate the meshes:


```python
overwrite = False
verbosePlot = True
#
ubathy.CreateMeshes(pathFolderData, pathFolderVideos, pathFolderScratch, listOfVideos, overwrite, verbosePlot)
pathFolderData = os.path.join(pathFolderMain, 'data')
```

As a result, in the scratch folder **`scratch`**, the file containing the mesh for the bathymetry `mesh_Zb.npz`, which is common for all the videos, and the meshes for obtaining the wave modes `mesh_M.npz` and the wavenumbers `mesh_K.npz` for each video `<videoFilename>` will be obtained. In case the plots have been generated, the corresponding figures will be found in the **`scratch/plots`** folder, following the same folder structure as the data files. 

## Mode decomposition
Once the frames of the videos are available and the meshes have been generated, the decomposition of the waves into modes can be performed. Prior to the decomposition into modes, an algorithm based on Principal Component Analysis (Robust PCA) can be applied to reduce the noise in the images. The code includes two algorithms for the decomposition of the waves into modes. One is based on Empirical Orthogonal Functions (EOF) and the other on Dynamic Mode Decomposition (DMD). These analyses are performed on sub-videos of the main video. The length of these sub-videos and their number will determine the wave periods that can be solved and the number of modes.
The following parameters need to be set in the `parameters.json` file: 

| Object-name | Description | Suggested value | Suggested range | Units | 
|:--|:--:|:--:|:--:|:--:|
| `candes_iter` | iterations robust algorithm | 50 | 0-100 | _-_ |
| `DMD_or_EOF` | type of mode decomposition  | DMD | DMD or EOF |  |
| `DMD_rank` | number of DMD modes | 6 | 4-10 | _-_ |
| `EOF_variance` | minimun variance of the EOF modes | 0.025 | 0.015-0.030 | _-_ |
| `time_step` | time steps for video analysis | 30.0 | 1.0-60 | _s_ |
| `time_windows` | temporal windows lengths | [60.0, 90.0, 120.0] | 30-150 | _s_ |
| `min_period` | minimum wave period  | 3 | site-dependent  | _s_ |
| `max_period` | maximum wave period | 15 | site-dependent | _s_ |

If `candes_iter=0` no Robust PCA is performed.

### Run mode decomposition
Set the values of the plot generation and overwrite parameters, and run the wave mode decomposition code:


```python
overwrite = False
verbosePlot = True
#
ubathy.ObtainWAndModes(pathFolderData, pathFolderVideos, pathFolderScratch, listOfVideos, overwrite, verbosePlot)
```

As a result, for all the videos included in the list `listOfVideos` and for every time window, the wave modes that verify the conditions set in the `parameters.json` file will be obtained. Each of these wave modes has an associated wave period. For each mode, a `t<timeInVideo>_w<windowTime>_T<period>_<modeType>.npz` file will be created in the **`scratch/<videoFilename>/M_modes`** folder containing, among other quantities, the phase of the waves at the nodes of the `mesh_M.npz`. In case the plots have been generated, the corresponding figures with the phase and amplitude of the modes will be found in the **`scratch/plots`** folder, following the same folder structure as the data files. 

## Wavenumber computation
The spatial structure of the modes is analysed to extract the wavenumber corresponding to each spatial point of the mode. The wavenumber is determined at each point from the values of the phase of the mode in its vicinity. The size of the spatial neighborhood for this analysis is determined by the range of depths to be measured and the number of neighborhoods to use. The following parameters need to be set in the `parameters.json` file:

| Object-name | Description | Suggested value | Suggested range | Units | 
|:--|:--:|:--:|:--:|:--:|
| `min_depth` | minimum depth of inversion  | 0.5 | site-dependent | _m_ |
| `max_depth` | maximum depth of inversion  | 6.0 | site-dependent | _m_ |
| `nRadius_K` | number of space neighborhoods for wavenumber calculation | 2 | 2-5 | _-_ |
| `nRANSAC_K` | number of iterations for RANSAC implementation | 50 | 20-100 | _-_ |

### Run wavenumber computation
Set the values of the plot generation and overwrite parameters, and run the wavenumber computation code:


```python
overwrite = False
verbosePlot = True
#
ubathy.ObtainK(pathFolderData, pathFolderScratch, listOfVideos, overwrite, verbosePlot)
```

As a result, for all the modes obtained after analising the videos included in the list `listOfVideos`, the wavenumber will be obtained. For each mode, a `t<timeInVideo>_w<windowTime>_T<period>_<modeType>_K.npz` file will be created in the **`scratch/<videoFilename>/K_modes`** folder containing, among other quantities, the wave period and wavenumber at the nodes of the `mesh_K.npz`. In case the plots have been generated, the corresponding figures with the phase and amplitude of the modes will be found in the **`scratch/plots`** folder, following the same folder structure as the data files. 

## Bathymetry estimation
Finally, once the wave modes have been extracted from the videos, and from these, the wavenumbers, the bathymetries can be estimated. These bathymetries will be made by composing periods and wavenumbers from different videos listed in the file `videos4dates.json` in the folder **`data`**. The fields in this file have the following format:
* `videos4dates.json`: For each bathymetry the name of all the `<videoFilename>`'s to compose a `<date>`
>  `"<date01>"`: [`"<videoFilename01>"`, `"<videoFilename02>"`, `"<videoFilename03>"`, . . .]

To compose the bathymetry at a point in space, frequency and wavenumber pairs are used in an environment determined by the wavelength at that point. In the composition of the bathymetry the standard deviation of the _gamma_ parameter of the different modes is used as a quality criterion. The following parameters need to be set in the `parameters.json` file:

| Object-name | Description | Default value | Suggested range | Units | 
|:--|:--:|:--:|:--:|:--:|
| `stdGammaC` | critical value for the standart deviation of the _gamma_ parameter | 0.075 | 0.060-0.090 | _-_ |
| `cRadius_Zb` | wavelength ratio for the space neighborhood for depth calculation  | 0.2 | 0.10-0.30 | _-_ |

In case a reference bathymetry is available for a date `<date>`, the relative difference between the estimated bathymetry and the reference bathymetry is computed. This bathymetry must be located in the folder **`data/groundTruth`** with the name `<date>_GT_xyz.txt`.The structure of tnis files is the following:
* `<date>_GT_xyz.txt`: For each points reference bathymetry one line with
>`x-coordinate`, `y-coordinate`, `z-coordinate-bathymetry`

These points are to be given in the same coordinate system in which the bathymetry is going to be obtained. The `z`-axis is assumed to be upward directed.

### Run bathymetry estimation
Set the values of the plot generation and overwrite parameters, and run the bathymetry estimation code:


```python
overwrite = False
verbosePlot = True
#
ubathy.ObtainZb(pathFolderData, pathFolderScratch, pathFolderBathymetries, overwrite, verbosePlot)
```

As a result, for each `<date>` in the file `videos4dates.json`, a bathymetry `<date>_<modeType>_Zb.npz` at the nodes of the `mesh_Zb.npz` will be created in the folder **`scratch/Zb_bathymetries`**. If the file `<date>_GT_xyz.txt` exists for a `<date>` for which the bathymetry has been estimated, an `<date>_GT_Zb.npz` file will also be created with the bathymetry interpolated at the nodes of the `mesh_Zb.npz`. In case the plots have been generated, the corresponding figures with the bathymetry will be found in the **`scratch/plots`** folder, following the same folder structure as the data files.

To facilitate the reading and processing of these bathymetries, files containing the grid points and bathymetries in _plain text_ are placed in the folder **`bathymetries`**. The structure of each of these files is the following:
* `mesh_Zb.txt`: For each grid points of the mesh one line with
>`x-coordinate`, `y-coordinate`

* `<date>_Zb.txt`: For each grid points of the mesh one line with
>`z-coordinate`, `self_error`

* `<date>_GT_Zb.txt`: For each grid points of the mesh one line with
>`z-coordinate`

The order of the grid points in the mesh and bathymetry files is the same. Therefore, to the same line number belongs the coordinates and the bathymetry of the same point in the corresponding files. The values of the bathymetry at the points where it has not been possible to evaluate are indicated by _`NaN`_ values. In case the plots have been generated, the corresponding figures with the bathymetry will be found in the **`bathymetries/plots`** folder.

## Kalman estimation
Coming soon!


## Contact us

Are you experiencing problems? Do you want to give us a comment? Do you need to get in touch with us? Please contact us!

To do so, we ask you to use the [Issues section](https://github.com/Ulises-ICM-UPC/UBathy/issues) instead of emailing us.

## Contributions

Contributions to this project are welcome. To do a clean pull request, please follow these [guidelines](https://github.com/MarcDiethelm/contributing/blob/master/README.md)

## License

UCalib is released under a [AGPL-3.0 license](https://github.com/Ulises-ICM-UPC/UBathy/blob/master/LICENSE). If you use UDrone in an academic work, please cite:

    @Article{rs11232722,
      AUTHOR = {Simarro, Gonzalo and Calvete, Daniel and Luque, Pau and Orfila, Alejandro and Ribas, Francesca},
      TITLE = {UBathy: A New Approach for Bathymetric Inversion from Video Imagery},
      JOURNAL = {Remote Sensing},
      VOLUME = {11},
      YEAR = {2019},
      NUMBER = {23},
      ARTICLE-NUMBER = {2722},
      URL = {https://www.mdpi.com/2072-4292/11/23/2722},
      ISSN = {2072-4292},
      DOI = {10.3390/rs11232722}
      }

    @Online{ulisesbathy, 
      author = {Simarro, Gonzalo and Calvete, Daniel and Nosequienmas},
      title = {UBathy},
      year = 2022,
      url = {https://github.com/Ulises-ICM-UPC/UBathy}
      }

