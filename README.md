This is an example code showing a simple SFM pipeline using OpenCV + GTSAM + PMVS2. OpenCV is used for feature matching, motion recovery and triangulation. GTSAM is used for bundle adjustment. The code output files for consumption by PMVS2, which does dense 3d point reconstruction. The code assumes the images are taken sequentialy apart by the same camera, with no change in zoom.

This code is meant to be educational so no attempts have been made to make it production friendly.

### REQUIREMENTS
- GTSAM (https://bitbucket.org/gtborg/gtsam)
- PMVS2 (www.di.ens.fr/pmvs/ or https://github.com/nghiaho12/RunSFM)
- Linux

I recommend using my RunSFM repo, which includes PMVS2 and all its dependencies. Clone it and type make. If it all works out the PMVS2 binary will be at RunSFM/cmvs/program/main/pmvs2.

This code only works on Linux at the moment because I use some system() calls to run mkdir and cp commands. It's pretty trivial to replace with say Boost.

### USAGE
Edit main.cpp and change the following variables for your dataset

- IMAGE_DOWNSAMPLE
- FOCAL_LENGTH
- IMAGE_DIR
- IMAGES

Alternatively, you can download the default dataset from

[http://nghiaho.com/uploads/desk.zip](http://nghiaho.com/uploads/desk.zip)

Unzip it in the root directory of the project.

Compile by running the script compile.sh and excuting ./main afterwards.

To recover the motion of the camera the pipeline doesn't require very high resolution images. So I would select an IMAGE_DOWNSAMPLE such that the final image width is somewhere around 1000 pixels.

The focal length for the images can be guessed from the EXIF data found in most jpeg files. You don't have to be too precise because GTSAM will optimize for the focal length, just as long as the initial guess is in the ballpark.

On a successful run, data is saved to the sub-directory root/ for PMVS2 as shown below, where the binary was ran. PMVS2 will run with the original resolution of the image, not the downsampled.

```
root/options.txt
root/models
root/txt
root/visualize
```

To run PMVS2 do
```
$ PATH_TO_PMVS2/pmvs2 root/ options.txt
```

Don't forget the trailing forward slash on root/, PMVS2 is a bit picky!

PMVS2 will output a PLY file in root/models/options.txt.ply. You can view the PLY in Meshlab.

### TIPS
If your editor supports code folding you can use that to see the individual parts of the SFM pipeline for easier digesting. There are four scoped sections

- Feature matching
- Motion recovery and triangulation
- Bundle adjustment
- Output for PMVS2

When using Meshlab be aware the default camera projection is not orthographic. This means angles may appear incorrect, eg. right angles made by two walls. I would recommend viewing in orthographic as this will allow you to verify the accuracy of the model visually.
