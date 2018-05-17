# Q & A

## Phase 1
### Part 1

Q: How did you verify that you are parsing the contours correctly?

A: A unittest was created to visualize several examples of dicom - icontour annotation pairs. Visual validation of the separate
and overlaid image pairs suggests that parsing occurred as expected.

Q: What changes did you make to the code, if any, in order to integrate it into our production code base?

A: Simplified parse_dicom_file function to return the numpy array, instead of a dictionary with a single entry, which is
unnecessary usage of dictionary data structure.


### Part 2

Q. Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2?
If so, what? If not, is there anything that you can imagine changing in the future?

A. Data structure for matching subject ID's was converted to a dictionary with contour subject as key, as it became
clear that contour file names were to be used for querying dicom file names.

Q. How do you/did you verify that the pipeline was working correctly?

Unittest with visual validation was created to check for proper matching between image and annotation, as well as
proper random shuffling between epochs.

Q. Given the pipeline you have built, can you see any deficiencies that you would change if you had more time?
If not, can you think of any improvements/enhancements to the pipeline that you could build in?

Matching between dicom and contour slices requires several non-robust design decisions, including specific
regex pattern matching of the particular file naming convention for contour files.  Data robustness could be improved
 by implementing a database to host the set of images for a more robust file search approach.

## Phase 2

### Part 1

Q: Discuss any changes that you made to the pipeline you built in Phase 1, and why you made those changes.

A: Built out single image parser to parse image+icontour+ocontour, handling the case that no ocontour is available.
    Also, extended dataset filename aggregator to handle ocontour files.  Added toggleable feature to enable filtering the
    dataset to handle include those images/icontours with an ocontour corollary. Unittests extended accordingly.

### Part 2

Q: Let’s assume that you want to create a system to outline the boundary of the blood pool (i-contours), and you
already know the outer border of the heart muscle (o-contours). Compare the differences in pixel intensities inside
the blood pool (inside the i-contour) to those inside the heart muscle (between the i-contours and o-contours);
could you use a simple thresholding scheme to automatically create the i-contours, given the o-contours?
Why or why not? Show figures that help justify your answer.

A: The i-contour generally seems to align well with a area of lighter pixel threshold within the o-contour. We find a
    Threshold value relative to the maximum intensity pixel value of the o-contour region in each image. We find this to
    be 0.35. Using this threshold, we find the dice coefficient compared with the ground truth icontour to be 0.82
    based on a held-out test set of 50% of the data. We visualize a few examples to confirm that the threshold does in
    fact predict decently well.

    [Visualization of threshold](img/threshold35.png)

    Here, the left column is the icontour ground truth overlaid with the image, and the right column is the heuristic
    threshold 0.35 prediction overlaid with the image.


Q: Do you think that any other heuristic (non-machine learning)-based approaches, besides simple thresholding,
would work in this case? Explain.

A:

Q: What is an appropriate deep learning-based approach to solve this problem?

Q: What are some advantages and disadvantages of the deep learning approach compared your chosen heuristic method?


## Note

Full data is not uploaded for storage limit issues.  Code assumes identical file structure to given data file system.
