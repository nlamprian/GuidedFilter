GuidedFilter
============

`GuidedFilter` is an implementation of the [Guided Image Filtering](http://research.microsoft.com/en-us/um/people/kahe/eccv10/) algorithm in OpenCL. The `Guided Filter` is an image filter with many applications, one of which is **edge-preserving smoothing**. It has a non-approximate algorithm which is `O(1)` in the filter window size. It has excellent performance characteristics which make it a great alternative to the popular `Bilateral Filter`.

![hdwallpaperscool-com](https://github.com/nlamprian/GuidedFilter/wiki/assets/filter.png)

You can watch how the algorithm performs as a smoothing operator in the following two videos:
* [Guided Image Filtering on Kinect RGB stream with OpenCL](https://www.youtube.com/watch?v=cFQu10OsztI)
* [Guided Image Filtering on Kinect RGB and Depth streams with OpenCL](https://www.youtube.com/watch?v=PTLU1SiHCEY)

In the latter video, a **point cloud** is built from the `Kinect` streams. The Guided Image Filtering algorithm is applied to the `Depth` frame and each of the 3 channels of the `RGB` frame, separately. The frames are transfered to the GPU, processed with `OpenCL`, and then delivered directly to `OpenGL`. On my machine, I was able to get a **mean running time** of `5.2 ms`.

Note
----

The code was developed and tested on `Ubuntu 14.04.2`, on a system with an `AMD R9 270X` GPU.

The complete `documentation` is available [here](https://guided-filter.nlamprian.me).

For more details on the implemented algorithms, take a look at the project's [wiki](https://github.com/nlamprian/GuidedFilter/wiki/Algorithms).

Dependencies
------------

The project has a dependency on [CLUtils](https://github.com/nlamprian/CLUtils) (which is automatically downloaded by cmake). If you'd like to remove this dependency, you should be able to modify the kernel interface classes with minimal effort.

Currently, there are 3 example applications. For `guided_filter_image`, you'll need [OpenCV](https://github.com/jayrambhia/Install-OpenCV). For `guided_filter_kinect_rgb` and `guided_filter_kinect_point_cloud`, you'll need a Kinect and [libfreenect](https://github.com/OpenKinect/libfreenect/).

Compilation
-----------

```bash
git clone https://github.com/nlamprian/GuidedFilter.git
cd GuidedFilter

mkdir build
cd build

cmake -DBUILD_EXAMPLES=ON ..
# or to build the tests too
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON ..

make

# to run the examples (from the build directory!)
./bin/guided_filter_image
./bin/guided_filter_kinect_rgb
./bin/guided_filter_kinect_point_cloud

# to run the tests (e.g.)
./bin/guided_filter_tests_box
# or with profiling information
./bin/guided_filter_tests_box --profiling

# to install the libraries
sudo make install
# you'll need to copy manually the kernel
# files into your own projects

# to build the docs
make doxygen
firefox docs/html/index.html
```