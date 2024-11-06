                                                                     
                                                                     
                                                                     
                                             
The code and the algorithm are for non-commercial use only.

Paper : "Image Matting with KL-Divergence Based Sparse Sampling,ICCV 2015"

Author: Levent Karacan, Aykut Erdem, Erkut Erdem 

        (karacan@cs.hacettepe.edu.tr, aykut@cs.hacettepe.edu.tr, erkut@cs.hacettepe.edu.tr)

Date  : 26/04/2016

Version : 1.0 

Copyright 2016, Hacettepe University, Turkey.



Notes:
  1) DS3 Algorithm was taken from the author's website of "E. Elhamifar, G. Sapiro, and S. S. Sastry.
     Dissimilarity-based sparse subset selection. arXiv preprint arXiv:1407.6810, 2014"  
 		
     Please refer to Elhamifar et al's article when you use it.

  2) vl_feat computer vision library is required for superpixels.


  3) mtimesx library is used for multidimensional matrix multiplication, and requires Lapack and Blas libraries. 

     http://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support

  
  These libraries can be easily set by packet manager on linux systems

  % Linux Systems 

     matlabroot='MATLAB/R2013a'

     libmwblas.so and libmwlapack.so files may be under the glnxa64/ directory.

     lib_lapack = [matlabroot '/bin/glnxa64/lcc/libmwlapack.so'];

     lib_blas = [matlabroot '/bin/glnxa64/lcc/libmwblas.so'];

     mex('-DDEFINEUNIX','-largeArrayDims','mtimesx.c',lib_blas) ;



  % Windows Systems

     Visual Studio C++ Compiler

     mex mtimesx.c






 
