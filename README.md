# UBathy

`UBathy` is an open source software written in Python for nearshore bathymetry estimation from videos of calibrated raw images and/or of georeferenced planviews. 

### Description
The algorithm for bathymetry estimation is based on extracting wave modes from videos of nearshore surface wave propagation. These videos can be formed either from raw camera images, which must have been previously calibrated, or from georeferenced planviews. For each wave mode extracted from the videos, the frequency and the spatially dependent wavenumbers are obtained. The bathymetries are obtained by fitting the surface waves dispersion relationship with the wavenumbers and frequencies of the different modes. Bathymetries estimated at different times are finally aggregated using a Kalman filter to obtain the final bathymetries. The videos may be recorded on Argus-type video monitoring stations and from moving cameras, such as drones or satellites. The calibration of these videos and the generation of planviews, which are necessary in the case of drones, can be done using the software [UBasic](https://github.com/Ulises-ICM-UPC/UBasic), [UCalib](https://github.com/Ulises-ICM-UPC/UCalib) or [UDrone](https://github.com/Ulises-ICM-UPC/UDrone). Details on the algorithm and methodology are described in
> *Simarro, G.; Calvete, D. UBathy (v2.0): A Software to Obtain the Bathymetry from Video Imagery. Remote Sens. 2022, 14, 6139. https://doi.org/10.3390/rs14236139*

The bathymetry estimation process consists of the following steps:

 1. [Video setup](#video-setup)
 2. [Generation of meshes](#generation-of-meshes)
 3. [Mode decomposition](#mode-decomposition)
 4. [Wavenumber computation](#wavenumber-computation)
 5. [Bathymetry estimation](#bathymetry-estimation)
 
Finally, `UBathy` allows to aggregate bathymetries obtained at different times using a Kalman filter:

 6. [Kalman filtering](#kalman-filtering)


### Requirements and project structure
To run the software it is necessary to have Python3 (3.9) and install the following dependencies:
- OpenCV (4.5.1)
- NumPy (1.19.5)
- SciPy (1.6.0)
- matplotlib (3.3.4)

In parenthesis we indicate the version with which the software has been tested. It is possible that it works with older versions. 

The structure of the project is the following:
* `example.py`
* `example_notebook.py`
* **`ubathy`**
  * `ubathy.py`
  * `ulises_ubathy.py`
* **`example`**
  * **`data`**
    * `boundary_xy.txt`
    * `parameters.json`
    * `videos4dates.json`
    * **`videos`**
      * **`videoFilename01`**
        * `<anyname>cal.txt` and `<anyname>zs.txt`
        * or `<anyname>crxyz<anymorename>.txt`
        * `videoFilename01.mp4` (.avi or .mov)
      * **`videoFilename02`**
        * `<anyname>cal.txt` and `<anyname>zs.txt`
        * or `<anyname>crxyz<anymorename>.txt`
        * **`frames`** or **`planviews`**
          * `<anyname>000000000000.png` (or .jpg)
          * . . .
      * . . .
    * **`groundTruth`**
      * `date01_GT_xyz.txt`
      * . . . 
  * **`output`**
    * **`numerics`**
      * `mesh_B.txt`
      * **`bathymetries`**
        * `date01.txt`
        * `date01_GT.txt`
        * . . .
      * **`bathymetries_Kalman`**
        * `date01.txt`
        * `date01_GT.txt`
        * . . .
    * **`plots`**
      * `mesh_B.png`
      * **`bathymetries`**
        * `date01.png`
        * . . .
      * **`bathymetries_Kalman`**
        * `date01.png`
        * . . .
  * **`scratch`**
    * **`frames`**
      * **`videoFilename01`**
        * `<videoFilename01>000000000000.png`
        * . . .
      * . . .
    * **`numerics`**
      * **`videoFilename01`**
        * `mesh_M.npz`
        * `mesh_K.npz`
        * **`M_modes`**
          * `t<timeInVideo>_w<windowTime>_T<period>_<modeType>.npz`
          * . . . 
        * **`K_wavenumber`**
          * `t<timeInVideo>_w<windowTime>_T<period>_<modeType>.npz`
          * . . . 
      * . . .
    * **`plots`**
      * **`videoFilename01`**
        * `in_image_mesh_M.jpg`
        * `in_image_mesh_K.jpg`
        * `mesh_M.jpg`
        * `mesh_K.jpg`
        * **`M_modes`**
          * `t<timeInVideo>_w<windowTime>_T<period>_<modeType>.jpg`
          * . . . 
        * **`K_wavenumber`**
          * `t<timeInVideo>_w<windowTime>_T<period>_<modeType>.jpg`
          * . . . 
      * . . .

The local modules of `UBathy` are located in the **`ubathy`** folder. Folder **`scratch`** contains auxiliary files. Once completed, the files can be deleted.  

For a complete demo run of `UBathy` we have provided a set of pre-calculated bathymetries to be used for Kalman filtering. These are located in the **`output_example`** folder. Rename this folder to **`output`** for a complete demo.

To run a demo in folder **`example`** experienced users can run the `example.py` file in a terminal. Alternatively we provide the file `example_notebook.ipynb` to be used in a Jupyter Notebook. In that case, import modules and set the main path of the example:


```python
import sys
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, 'ubathy')
import ubathy as ubathy
pathFldMain = 'example'
```

Adjust the parameter values in the `parameters.json` file located in the **`data`** folder. To check the current values, run:


```python
ubathy.Inform_UBathy(pathFldMain, 0)
```

As general control parameter, the `overwrite_outputs` parameter controls whether results from previous runs are recomputed (`true`) or preserved (`false`). To facilitate the verification of the results, auxiliary images can be generated. Set parameter `generate_scratch_plots = true`, and to `false` otherwise.

## Video setup
The bathymetric extraction algorithm is applied to videos stored in the **`data/videos`** folder. For each video `<videoFilename>` a folder with the same name must contain the frames with the format `<anyname><milliseconds>.png` (or `.jpg`). For planview videos the filemane might end with `plw` (`<anyname><milliseconds>plw.png`). Depending on whether the frames come from calibrated camera images or from georeferenced planviews, different files have to be provided.

#### Calibrated camera images
A calibration file `<anyname>cal.txt` of the camera must exist in the folder **`data/videos/<videoFilename>`** with the following parameters:

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

#### Georeferenced planviews
If the frames correspond to planviews, in the folder **`data/videos/<videoFilename>`** there must exists the file `<anyname>crxyz<anymorename>.txt` with the correspondence between the planview corner pixels and the coordinate system in which the bathymetry is going to be obtained. The structure of this file is the following:
* `<anyname>crxyz<anymorename>.txt`: One line for each corner pixel
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate-free-surface`

This file can be obtained using the [UDrone](https://github.com/Ulises-ICM-UPC/UDrone) software. On previous files, quantities must be separated by at least one blank space between them and the last record should not be continued with a newline (return).

### Run video extraction
If the video has been stored in `MP4`, `AVI` or `MOV` image format, the frames must be extracted and will be placed in the folder **`scratch/frame/<videoFilename>`** before processing. 

Set the extraction rate of the frames in `parameters.json`:

|  | Parameter | Suggested value | Units |
|:--|:--:|:--:|:--:|
| Extraction framerate | `frame_rate` | _2.0_ | _1/s_ |
Set frame_rate=0 to extract all frames from the video.

Run the code to extract frames from de videos:


```python
ubathy.Video2Frames(pathFldMain)
```

As a result, for each `<videoFilename>` a folder **`scratch/frames/<videoFilename>`** containing the frames with the format `<videoFilename>_<milliseconds>.png` is generated.

## Generation of meshes
In general, the meshes for the extraction of wave modes and for the calculation of wavenumbers and bathymetries are performed on the basis of a regular mesh of equilateral triangles in an x-y plane domain. The mesh for the extraction of the modes is adjusted according to the position of the pixels of the images of each video `<videoFilename>` if the frames come from calibrated camera images. For georeferenced planviews, the mesh extraction of wave modes corresponds to spatial coordinates of the planview whose corner pixels are specified in the file `<anyname>crxyz<anymorename>.txt`.

To generate the meshes it is necessary to provide in the folder **`data`** a file `boundary_xy.txt` with the coordinates of the polygon vertices of the region in which the bathymetry is going to be extracted and to set in the file `parameters.json` the spacing between the nodes of the different meshes.

The structure of each of these files is the following:
* `boundary_xy.txt`: For each vertex point one line with (minimum 3)
>`x-coordinate`, `y-coordinate`

Quantities must be separated by at least one blank space between. These points are to be given in the same coordinate system in which the bathymetry is going to be obtained
* `parameters.json`: Set the values of node distance for ech mesh

| Object-name | Description | Suggested value | Units | 
|:--|:--:|:--:|:--:|
| `delta_M` | approx. mode-node distance | 2.5 | _m_ |
| `delta_K` | wavenumber-node distance | 5.0 | _m_ |
| `delta_B` | bathymetry-node distance | 5.0| _m_ |

Note that `delta_M` is not used for videos of georeferenced planviews.

### Run meshes generation
To verify the arrangement of the mesh, images of the meshes and the position of the grid nodes relative to the video frames can be generated. Run the code to generate the meshes:


```python
ubathy.CreateMeshes(pathFldMain)
```

As a result, in the scratch folder **`output/mumerics`**, the file containing the mesh for the bathymetry `mesh_B.txt`, which is common for all the videos, will be generated. The meshes for obtaining the wave modes `mesh_M.npz` and the wavenumbers `mesh_K.npz` for each video `<videoFilename>` will be generated in the folders **`scratch/mumerics/<videoFilename>`**. In case the plots are generated, the corresponding figures will be found in the **`scratch/plots`** folders, following the same folder structure as the data files. 

## Mode decomposition
Once the frames of the videos are available and the meshes have been generated, the decomposition of the waves into modes can be performed. Prior to the decomposition into modes, an algorithm based on Principal Component Analysis (Robust PCA) can be applied to reduce the noise in the images. The code includes two algorithms for the decomposition of the waves into modes. One is based on Empirical Orthogonal Functions (EOF) and the other on Dynamic Mode Decomposition (DMD). These analyses are performed on sub-videos of the main video. The length of these sub-videos and their number will determine the wave periods that can be solved and the number of modes.
The following parameters need to be set in the `parameters.json` file: 

| Object-name | Description | Suggested value | Suggested range | Units | 
|:--|:--:|:--:|:--:|:--:|
| `time_step` | time steps for video analysis | 30.0 | 1.0-60 | _s_ |
| `time_windows` | temporal windows lengths | [60.0, 90.0, 120.0] | 30-150 | _s_ |
| `min_period` | minimum wave period  | 3 |  _-_  | _s_ |
| `max_period` | maximum wave period | 15 |  _-_ | _s_ |
| `candes_iterations` | iterations robust algorithm | 50 | 40-80 | _-_ |
| `decomposition_method` | type of mode decomposition  | DMD | DMD or EOF |  |
| `DMD_rank` | number of DMD modes | 6 | 4-10 | _-_ |
| `EOF_min_variance` | minimun variance of the EOF modes | 0.025 | 0.010-0.100 | _-_ |

If `candes_iterations=0` Robust PCA is not performed.

### Run mode decomposition
Set the values of the plot generation and overwrite parameters, and run the wave mode decomposition code:


```python
ubathy.ObtainWAndModes(pathFldMain)
```

As a result, for all the videos and for every time window, the wave modes that verify the conditions set in the `parameters.json` file will be obtained. Each of these wave modes has an associated wave period. For each mode, a `t<timeInVideo>_w<windowTime>_T<period>_<modeType>.npz` file will be created in the **`scratch/numerics/<videoFilename>/M_modes`** folder containing, among other quantities, the phase of the waves at the nodes of the `mesh_M.npz`. In case the plots have been generated, the corresponding figures with the phase and amplitude of the modes will be found in the  **`scratch/plots/<videoFilename>/M_modes`** folder, following the same folder structure as the data files. 

## Wavenumber computation
The spatial structure of the modes is analysed to extract the wavenumber corresponding to each spatial point of the mode. The wavenumber is determined at each point from the values of the phase of the mode in its vicinity. The size of the spatial neighborhood for this analysis is determined by the range of depths to be measured and the number of neighborhoods to use. The following parameters need to be set in the `parameters.json` file:

| Object-name | Description | Suggested value | Suggested range | Units | 
|:--|:--:|:--:|:--:|:--:|
| `min_depth` | minimum depth of inversion  | 0.5 |  _-_ | _m_ |
| `max_depth` | maximum depth of inversion  | 6.0 |  _-_ | _m_ |
| `n_neighborhoods_K` | number of space neighborhoods for wavenumber calculation | 3 | 2-5 | _-_ |
| `coeff_radius_K` | wavelength factor for neighbourhood radius | 0.60 | 0.40-0.60 | _-_ |

### Run wavenumber computation
Run the wavenumber computation code:


```python
ubathy.ObtainK(pathFldMain)
```

As a result, for all the modes obtained after analising the videos, the wavenumber will be obtained. For each mode, a `t<timeInVideo>_w<windowTime>_T<period>_<modeType>.npz` file will be created in the **`scratch/numerics/<videoFilename>/K_modes`** folder containing, among other quantities, the wave period and wavenumber at the nodes of the `mesh_K.npz`. In case the plots have been generated, the corresponding figures with the phase and amplitude of the modes will be found in the **`scratch/plots/<videoFilename>/K_modes`** folder, following the same folder structure as the data files. 

## Bathymetry estimation
Finally, once the wave modes have been extracted from the videos, and from these, the wavenumbers, the bathymetries can be estimated. These bathymetries will be made by composing periods and wavenumbers from different videos listed in the file `videos4dates.json` in the folder **`data`**. The fields in this file have the following format:
* `videos4dates.json`: For each bathymetry the name of all the `<videoFilename>`'s to compose a `<date>` with format `"yyyyMMddhhmm"`
>  `"<date01>"`: [`"<videoFilename01>"`, `"<videoFilename02>"`, `"<videoFilename03>"`, . . .]

To compose the bathymetry at a point in space, frequency and wavenumber pairs are used in an environment determined by the wavelength at that point. In the composition of the bathymetry the standard deviation of the _gamma_ parameter of the different modes is used as a quality criterion. The following parameters need to be set in the `parameters.json` file:

| Object-name | Description | Default value | Suggested range | Units | 
|:--|:--:|:--:|:--:|:--:|
| `gamma_std_cutoff` | critical value for the standart deviation of the _gamma_ parameter | 0.075 | 0.060-0.090 | _-_ |
| `coeff_radius_B` | wavelength factor for neighbourhood radius | 0.20 | 0.10-0.30 | _-_ |

In case a Ground Truth bathymetry is available for a date `<date>`, the relative difference between the estimated bathymetry and the Ground Truth bathymetry is computed. This bathymetry must be located in the folder **`data/ground_truth`** with the name `<date>_GT_xyz.txt`. The structure of this file is the following:
* `<date>_GT_xyz.txt`: For each points reference bathymetry one line with
>`x-coordinate`, `y-coordinate`, `z-coordinate-bathymetry`

These points are to be given in the same coordinate system in which the bathymetry is going to be obtained. The `z`-axis is assumed to be upward directed.

### Run bathymetry estimation
Run the bathymetry estimation code:


```python
ubathy.ObtainB(pathFldMain)
```

As a result, for each `<date>` in the file `videos4dates.json`, a bathymetry `<date>_<modeType>.txt` at the nodes of the `mesh_B.txt` will be created in the folder **`output/numerics/bathymetries`**. If the file `<date>_GT_xyz.txt` exists for a `<date>` for which the bathymetry has been estimated, an `<date>_GT.txt` file will also be created with the bathymetry interpolated at the nodes of the `mesh_B.txt`. In case the plots have been generated, the corresponding figures with the bathymetry will be found in the **`output/plots/bathymetries`** folder, following the same folder structure as the data files.

The order of the grid points in the mesh and bathymetry files is the same. Therefore, to the same line number belongs the coordinates and the bathymetry of the same point in the corresponding files. The values of the bathymetry at the points where it has not been possible to evaluate are indicated by _`NaN`_ values.

## Kalman filtering
Bathymetries obtained at different times with a unique mesh for the bathymetry can be aggregate through a Kalman filter. Bathymetries `<date>.txt` located in the folder **`output/numerics/bathymetries`** that belong to the interval between `Kalman_start` and `Kalman_end`, with format `"yyyyMMddhhmm"`, will be aggregated correlatively. All these bathymetries must have been obtained with the same `mesh_B.txt` which will be located in the same folder. The following parameters need to be set in the `parameters.json` file:

| Object-name | Description | Default value | Suggested range | Units | 
|:--|:--:|:--:|:--:|:--:|
| `Kalman_ini` | initial data | _-_ | 202007250800 |`"yyyyMMddhhmm"` |
| `Kalman_fin` | final data | _-_ | 202008010900 | `"yyyyMMddhhmm"` |
| `variance_per_day` | daily bottom variability | 0.10 | 0.05-0.25 | _m/day_ |

In case Ground Truth bathymetries are available for several `<date>`'s, they must be located in the same folder with the name `<date>_GT.txt` and they should be interpolated at the nodes the same `mesh_B.txt`.

### Run Kalman filtering
Run the bathymetry estimation code:


```python
ubathy.PerformKalman(pathFldMain)
```

The Kalman filter runs on the **`output/numerics/bathymetries`** folder. In this example pre-computed bathymetries of different dates are provided in that folder. As a result, for all the bathymetries `<date>.txt` a filtered bathymetry `<date>.txt`, at the nodes of `mesh_B.txt`, that agregates all of the preceding bathymetries starting from `Kalman_start` will be created. The structure filtered bathymetry is the following:
* `<date>.txt`: For each grid points of the mesh one line with
>`z-coordinate`, `standart deviation`

The order of the grid points in the mesh and bathymetry files is the same. The values of the bathymetry at the points where it has not been possible to evaluate are indicated by _`NaN`_ values. In case the plots have been generated, the corresponding figures with the filtered bathymetries will be found in the **`output/plots/bathymetries_Kalman`** folder.


## Contact us

Are you experiencing problems? Do you want to give us a comment? Do you need to get in touch with us? Please contact us!

To do so, we ask you to use the [Issues section](https://github.com/Ulises-ICM-UPC/UBathy/issues) instead of emailing us.

## Contributions

Contributions to this project are welcome. To do a clean pull request, please follow these [guidelines](https://github.com/MarcDiethelm/contributing/blob/master/README.md)

## License

UBathy is released under a [AGPL-3.0 license](https://github.com/Ulises-ICM-UPC/UBathy/blob/master/LICENSE). If you use UBathy in an academic work, please cite:

    @Article{rs14236139,
      AUTHOR = {Simarro, Gonzalo and Calvete, Daniel},
      TITLE = {UBathy (v2.0): A Software to Obtain the Bathymetry from Video Imagery},
      JOURNAL = {Remote Sensing},
      VOLUME = {14},
      YEAR = {2022},
      NUMBER = {23},
      ARTICLE-NUMBER = {6139},
      URL = {https://www.mdpi.com/2072-4292/14/23/6139},
      ISSN = {2072-4292},
      DOI = {10.3390/rs14236139}
      }

    @Online{ubathyZenodo, 
      author = {Simarro, Gonzalo and Calvete, Daniel},
      title = {UBathy: A software to obtain the bathymetry from video imagery (version 2.0.0)},
      year = 2022,
      url = {10.5281/zenodo.7360216}
      }
    
    @Online{ulisesbathy, 
      author = {Simarro, Gonzalo and Calvete, Daniel},
      title = {UBathy},
      year = 2022,
      url = {https://github.com/Ulises-ICM-UPC/UBathy}
      }

