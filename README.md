# Q & A

## Part 1

Q: How did you verify that you are parsing the contours correctly?

A: A unittest was created to visualize several examples of dicom - icontour annotation pairs. Visual validation of the separate
and overlaid image pairs suggests that parsing occurred as expected.

Q: What changes did you make to the code, if any, in order to integrate it into our production code base?

A: Simplified parse_dicom_file function to return the numpy array, instead of a dictionary with a single entry, which is
unnecessary usage of dictionary data structure.


## Part 2

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

# Note

Full data is not uploaded for storage limit issues.  Code assumes identical file structure to given data file system.
