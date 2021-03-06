# Q & A

## Phase 1
### Part 1

Q:  How did you verify that you are parsing the contours correctly?

A:  A unittest was created to visualize several examples of dicom - icontour annotation pairs. Visual validation of the separate
and overlaid image pairs suggests that parsing occurred as expected.

Q:  What changes did you make to the code, if any, in order to integrate it into our production code base?

A:  Simplified parse_dicom_file function to return the numpy array, instead of a dictionary with a single entry, which is
    unnecessary usage of dictionary data structure.


### Part 2

Q.  Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2?
    If so, what? If not, is there anything that you can imagine changing in the future?

A.  Data structure for matching subject ID's was converted to a dictionary with contour subject as key, as it became
    clear that contour file names were to be used for querying dicom file names.

Q. How do you/did you verify that the pipeline was working correctly?

A.  Unittest with visual validation was created to check for proper matching between image and annotation, as well as
    proper random shuffling between epochs.

Q.  Given the pipeline you have built, can you see any deficiencies that you would change if you had more time?
    If not, can you think of any improvements/enhancements to the pipeline that you could build in?

A.  Matching between dicom and contour slices requires several non-robust design decisions, including specific
    regex pattern matching of the particular file naming convention for contour files.  Data robustness could be improved
    by implementing a database to host the set of images for a more robust file search approach.

## Phase 2

### Part 1

Q:  Discuss any changes that you made to the pipeline you built in Phase 1, and why you made those changes.

A:  Built out single image parser to parse image+icontour+ocontour, handling the case that no ocontour is available.
    Also, extended dataset filename aggregator to handle ocontour files.  Added toggleable feature to enable filtering the
    dataset to handle include those images/icontours with an ocontour corollary. Unittests extended accordingly.

### Part 2

Q:  Let’s assume that you want to create a system to outline the boundary of the blood pool (i-contours), and you
    already know the outer border of the heart muscle (o-contours). Compare the differences in pixel intensities inside
    the blood pool (inside the i-contour) to those inside the heart muscle (between the i-contours and o-contours);
    could you use a simple thresholding scheme to automatically create the i-contours, given the o-contours?
    Why or why not? Show figures that help justify your answer.

A:  The i-contour generally seems to align well with a area of lighter pixel threshold within the o-contour. We find a
    Threshold value relative to the maximum intensity pixel value of the o-contour region in each image. We find this to
    be 0.35. Using this threshold, we find the dice coefficient compared with the ground truth icontour to be 0.829 +/-
    0.029 (sd) based on a held-out test set of 50% of the data bootstrapped across 5 runs, which demonstrates solid
    performance. We visualize a few examples to confirm that the threshold does in fact predict modestly well.

![Visualization of threshold](img/threshold35-1.png)

    Two examples, one in each row. Left column: Ocontour baseline annotation overlaid with image; Mid column:
    Icontour ground truth overlaid with the image; Right column: Predicted Icontour with thresholding heuristic
    @0.35 maxval (described in detail above) overlaid with the image.


Q:  Do you think that any other heuristic (non-machine learning)-based approaches, besides simple thresholding,
    would work in this case? Explain.

A:  Calculating edge of the icontour by observing boundaries between light and dark boundaries near the edge of the
    ocontour could be another useful approach.  We know that there is typically a ring of dark pixels about 10-20%
    radius thickness of the ocontour region that signals the boundary of the icontour region. Finding the inner edge
    of this dark band, i.e. where pixels go from dark to light, could effectively mark the edge of the icounter region.
    This morphological implementation is beyond the scope of this task. However, we can clearly see this visual pattern
    in the image overlays above.

Q: What is an appropriate deep learning-based approach to solve this problem?

A:  An appropriate deep learning start to this problem could involve basic segmentation using a gold-standard segnet
    such as U-net. Standard data augmentation like random shear, rotation, translation, flip and even elastic
    deformation could benefit. Standard training techniques could apply such as using RMSProp/Adam optimizer and
    searching for optimal dropout, and using Batch Normalization appropriately. Input could be a simple two-channel
    image + ocontour annotation, or single channel image-only if we do not have access to the ocontour ground truth.
    A custom weighted pixel-wise cross entropy (weighted more heavily near the edge) could be used as a loss function,
    or simply dice coefficient loss. Transfer learning for model weight initialization could be done using a larger dataset.

Q:  What are some advantages and disadvantages of the deep learning approach compared your chosen heuristic method?

A:  Deep learning would automatically learn the edge detection (as well as higher level feature detection) mechanisms
    described above, leading to potentially improved performance, especially given larger dataset.  With appropriate
    annotation of surrounding anatomical structures, deep learning could learn how icontour region locates relative to
    other structure and gain even greater improved segmentation from this explaining away segmentation learning.

Visual review suggests that some ocontour annotations are erroneous -- for example, the ocontour annotation is
    incorrectly laterally translated and as a result is not a complete superset of the icontour annotation (see below).
    DL methods could learn icontour segmentation without requiring ocontour matching annotations, thus being more robust
    to human annotation error in this task.

Some disadvantages include requiring a decent (typically few hundred slices minimum) along with appropriate pixel-
    level annotations, which may be human-resource intensive to obtain. Deep learning prediction by slice can take
    several seconds of a 3D prediction is required, which may be a clinical issue depending on time-sensitvity, whereas
    a simple thresholding prediction could be calculated in real-time. Deep learning generally requires greater care to
    train appropriately than simple thresholding techniques.

![Erroneous examples](img/error35.png)

    Left Column: Ocontour baseline with image overlay. Mid Column: Icontour ground truth with image overlay. Right
    Column: Predicted Icontour with image overlay. An obvious error in Ocontour annotation leads to major error in
    Icontour thresholding prediction.


## Note

Full data is not uploaded for storage limit issues.  Code assumes identical file structure to given data file system.
