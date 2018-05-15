# Q & A

## Part 1

Q: How did you verify that you are parsing the contours correctly?

A: A unittest was created to visualize several examples of dicom - icontour annotation pairs. Visual validation of the separate
and overlaid image pairs suggests that parsing occurred as expected.

Q: What changes did you make to the code, if any, in order to integrate it into our production code base?

A: Simplified parse_dicom_file function to return the numpy array, instead of a dictionary with a single entry, which is
superfluous usage of dictionary data structure.


## Part 2

Q. Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2?
If so, what? If not, is there anything that you can imagine changing in the future?

A.


Q. How do you/did you verify that the pipeline was working correctly?

Q. Given the pipeline you have built, can you see any deficiencies that you would change if you had more time?
If not, can you think of any improvements/enhancements to the pipeline that you could build in?
