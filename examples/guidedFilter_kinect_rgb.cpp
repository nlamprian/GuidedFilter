/*! \file guidedFilter_kinect_rgb.cpp
 *  \brief An example showcasing the effect of the `Guided Filter` algorithm on color images.
 *  \details This example demonstrates the performance of the [Guided Image Filtering]
 *           (http://research.microsoft.com/en-us/um/people/kahe/eccv10/) 
 *           algorithm on a live video stream. It processes the Kinect RGB  
 *           stream in OpenCL with the `GuidedFilter` pipeline.
 *  \author Nick Lamprianidis
 *  \version 1.0
 *  \date 2015
 *  \copyright The MIT License (MIT)
 *  \par
 *  Copyright (c) 2015 Nick Lamprianidis
 *  \par
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  \par
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  \par
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <mutex>
#include <CLUtils.hpp>
#include <GuidedFilter/algorithms.hpp>

#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <libfreenect.hpp>


// Window parameters
const int gl_win_width = 1280;
const int gl_win_height = 480;
int glWinId;

// GL texture IDs
GLuint glRGBTex, glRGBTexFilt;

// Freenect
class MyFreenectDevice;
Freenect::Freenect freenect;
MyFreenectDevice *device;
double freenectAngle = 0;

// OpenCL
const std::vector<std::string> kernel_files = { "kernels/imageSupport_kernels.cl", 
                                                "kernels/prefixSum_kernels.cl", 
                                                "kernels/transpose_kernels.cl", 
                                                "kernels/boxFilter_kernels.cl",
                                                "kernels/math_kernels.cl", 
                                                "kernels/guidedFilter_kernels.cl", 
                                                "kernels/examples_kernels.cl" };
const int imgWidth = 640;
const int imgHeight = 480;
const int dRadius = 5;
const float dEps = 0.02f;
class GFilterRGB;
GFilterRGB *gFilter;


/*! \brief Creates an OpenCL environment with CL-GL interoperability. */
class CLEnvGL : public clutils::CLEnv
{
public:
    /*! \brief Initializes the OpenCL environment. */
    CLEnvGL () : CLEnv ()
    {
        addContext (0, true);
        addQueueGL (0);
        addQueueGL (0);
        addProgram (0, kernel_files);
    }

private:
    /*! \brief Initializes the OpenGL memory buffers.
     *  \note Do not call directly. `initGLMemObjects` is called by `addContext`
     *        when creating the GL-shared CL context.
     */
    void initGLMemObjects ()
    {
        glGenTextures (1, &glRGBTex);
        glBindTexture (GL_TEXTURE_2D, glRGBTex);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, imgWidth, imgHeight,
                      0, GL_RGBA, GL_FLOAT, nullptr);

        glGenTextures (1, &glRGBTexFilt);
        glBindTexture (GL_TEXTURE_2D, glRGBTexFilt);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, imgWidth, imgHeight,
                      0, GL_RGBA, GL_FLOAT, nullptr);
        
        glBindTexture (GL_TEXTURE_2D, 0);
    }

};


/*! \brief Applies `Guided Image Filtering` on a `Kinect` RGB frame and 
 *         delivers the data to `OpenGL`.
 */
class GFilterRGB
{
public:
    GFilterRGB () : 
        env (), context (env.getContext (0)), 
        queue0 (env.getQueue (0, 0)), queue1 (env.getQueue (0, 1)), 
        kernelGLIn (env.getProgram (0), "combineRGBGL"), 
        kernelGLOut (env.getProgram (0), "combineRGBGL"), 
        global (imgWidth * imgHeight), info (0, 0, 0, { 0, 1 }, 0), 
        kGFRGB (env, info), bufferSize (imgWidth * imgHeight * sizeof (cl_float)), 
        waitListGLIn (1), waitListGLObj (1), normalizeRGB (0)
    {
        size_t wgMultiple = kernelGLIn.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[0][0]);

        // The local workspace should be a multiple of 3
        size_t wgM = wgMultiple;
        unsigned int pixels = imgWidth * imgHeight;
        while (pixels % (3 * wgM) != 0) wgM >>= 1;
        local = cl::NDRange (3 * wgM);

        // Initialize the Guided Image Filtering pipeline
        kGFRGB.get (cl_algo::Kinect::GuidedFilterRGB<cl_algo::Kinect::GuidedFilterRGBConfig::SEPARATED>
            ::Memory::D_OUT_R) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        kGFRGB.get (cl_algo::Kinect::GuidedFilterRGB<cl_algo::Kinect::GuidedFilterRGBConfig::SEPARATED>
            ::Memory::D_OUT_G) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        kGFRGB.get (cl_algo::Kinect::GuidedFilterRGB<cl_algo::Kinect::GuidedFilterRGBConfig::SEPARATED>
            ::Memory::D_OUT_B) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        kGFRGB.init (imgWidth, imgHeight, dRadius, dEps, cl_algo::Staging::I);

        // Create GL-shared images
        imagesGL.emplace_back (context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, glRGBTex);
        imagesGL.emplace_back (context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, glRGBTexFilt);

        // Set arguments for the kernel instance responsible for handling the original frame
        kernelGLIn.setArg (0, kGFRGB.get (cl_algo::Kinect::GuidedFilterRGB<cl_algo
            ::Kinect::GuidedFilterRGBConfig::SEPARATED>::Memory::D_INTR_R));
        kernelGLIn.setArg (1, kGFRGB.get (cl_algo::Kinect::GuidedFilterRGB<cl_algo
            ::Kinect::GuidedFilterRGBConfig::SEPARATED>::Memory::D_INTR_G));
        kernelGLIn.setArg (2, kGFRGB.get (cl_algo::Kinect::GuidedFilterRGB<cl_algo
            ::Kinect::GuidedFilterRGBConfig::SEPARATED>::Memory::D_INTR_B));
        kernelGLIn.setArg (3, imagesGL[0]);
        kernelGLIn.setArg (4, cl::Local (3 * local[0] * sizeof (cl_float)));
        kernelGLIn.setArg (5, imgWidth);
        kernelGLIn.setArg (6, normalizeRGB);

        // Set arguments for the kernel instance responsible for handling the processed frame
        kernelGLOut.setArg (0, kGFRGB.get (cl_algo::Kinect::GuidedFilterRGB<cl_algo
            ::Kinect::GuidedFilterRGBConfig::SEPARATED>::Memory::D_OUT_R));
        kernelGLOut.setArg (1, kGFRGB.get (cl_algo::Kinect::GuidedFilterRGB<cl_algo
            ::Kinect::GuidedFilterRGBConfig::SEPARATED>::Memory::D_OUT_G));
        kernelGLOut.setArg (2, kGFRGB.get (cl_algo::Kinect::GuidedFilterRGB<cl_algo
            ::Kinect::GuidedFilterRGBConfig::SEPARATED>::Memory::D_OUT_B));
        kernelGLOut.setArg (3, imagesGL[1]);
        kernelGLOut.setArg (4, cl::Local (3 * local[0] * sizeof (cl_float)));
        kernelGLOut.setArg (5, imgWidth);
        kernelGLOut.setArg (6, normalizeRGB);
    }

    /*! \brief Processes an RGB frame on the GPU.
     *  \details The processed frame is delivered directly to OpenGL from the GPU.
     *  
     *  \param[in] rgb frame to be processed.
     */
    void process (cl_uchar *rgb)
    {
        // Transfer data to device
        kGFRGB.write (cl_algo::Kinect::GuidedFilterRGB<cl_algo
            ::Kinect::GuidedFilterRGBConfig::SEPARATED>::Memory::D_IN, rgb);

        glFinish ();  // Wait for OpenGL pending operations on buffers to finish

        // Take ownership of OpenGL textures
        queue1.enqueueAcquireGLObjects ((std::vector<cl::Memory> *) &imagesGL);

        // Dispatch kernels
        kGFRGB.run (nullptr, &gfEvent); waitListGLIn[0] = gfEvent;
        queue1.enqueueNDRangeKernel (kernelGLIn, cl::NullRange, global, local, &waitListGLIn, &glEventIn);
        queue0.enqueueNDRangeKernel (kernelGLOut, cl::NullRange, global, local);
        waitListGLObj[0] = glEventIn;

        // Give up ownership of OpenGL textures
        queue0.enqueueReleaseGLObjects ((std::vector<cl::Memory> *) &imagesGL, &waitListGLObj);

        queue0.finish ();
    }

    void radiusUp ()
    {
        kGFRGB.setRadius (kGFRGB.getRadius () + 1);
    }

    void radiusDown ()
    {
        if (kGFRGB.getRadius () == 1) return;
        kGFRGB.setRadius (kGFRGB.getRadius () - 1);
    }

    void resetRadius ()
    {
        kGFRGB.setRadius (dRadius);
    }

    int getRadius ()
    {
        return kGFRGB.getRadius ();
    }

    void epsUp ()
    {
        kGFRGB.setEps (kGFRGB.getEps () + 0.005f);
    }

    void epsDown ()
    {
        float tmp = kGFRGB.getEps () - 0.005f;
        if (tmp < 0.f) return;
        kGFRGB.setEps (tmp);
    }

    void resetEps ()
    {
        kGFRGB.setEps (dEps);
    }

    float getEps ()
    {
        return kGFRGB.getEps ();
    }

    void toggleRGBNorm ()
    {
        normalizeRGB = !normalizeRGB;
        kernelGLIn.setArg (6, normalizeRGB);
        kernelGLOut.setArg (6, normalizeRGB);
    }

private:
    CLEnvGL env;
    cl::Context &context;
    cl::CommandQueue &queue0, &queue1;
    cl::Kernel kernelGLIn, kernelGLOut;
    std::vector<cl::ImageGL> imagesGL;  /*!< GL-shared images */
    cl::NDRange global, local;
    cl::Event gfEvent, glEventIn;
    std::vector<cl::Event> waitListGLIn, waitListGLObj;
    clutils::CLEnvInfo<2> info;
    cl_algo::Kinect::GuidedFilterRGB<cl_algo::Kinect::GuidedFilterRGBConfig::SEPARATED> kGFRGB;
    unsigned int bufferSize;
    int normalizeRGB;

};


/*! \brief A class hierarchy for manipulating a mutex. */
class Mutex
{
public:
    void lock () { freenectMutex.lock (); }
    void unlock () { freenectMutex.unlock (); }

    /*! \brief A class that automates the manipulation of 
     *         the outer class instance's mutex.
     *  \details Mutex's mutex is locked with the creation of a 
     *           ScopedLock instance and unlocked with the 
     *           destruction of the ScopedLock instance.
     */
    class ScopedLock
    {
    public:
        ScopedLock (Mutex &mtx) : mMutex (mtx) { mMutex.lock (); }
        ~ScopedLock () { mMutex.unlock (); }

    private:
        Mutex &mMutex;

    };

private:
    /*! A mutex for safely accessing a buffer updated by the freenect thread. */
    std::mutex freenectMutex;

};


/*! \brief A class that extends Freenect::FreenectDevice by defining 
 *         the VideoCallback function so we can be getting updates 
 *         with the latest RGB frame.
 */
class MyFreenectDevice : public Freenect::FreenectDevice
{
public:
    /*! \note The creation of the device is done through the Freenect class.
     *
     *  \param[in] ctx context to open device through (handled by the library).
     *  \param[in] idx index of the device on the bus.
     */
    MyFreenectDevice (freenect_context *ctx, int idx) : 
        Freenect::FreenectDevice (ctx, idx), newRGBFrame (false)
    {
        // setVideoFormat (FREENECT_VIDEO_YUV_RGB, FREENECT_RESOLUTION_MEDIUM);
        rgbBuffer = new cl_uchar[freenect_find_video_mode (
            FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes];
    }

    virtual ~MyFreenectDevice ()
    {
        delete[] rgbBuffer;
    }

    /*! \brief Delivers the latest RGB frame.
     *  \note Do not call directly, it's only used by the library.
     *  
     *  \param[in] rgb an array holding the rgb frame.
     *  \param[in] timestamp a time stamp.
     */
    void VideoCallback (void *rgb, uint32_t timestamp)
    {
        Mutex::ScopedLock lock (rgbMutex);
        
        std::copy ((cl_uchar *) rgb, (cl_uchar *) rgb + getVideoBufferSize (), rgbBuffer);
        newRGBFrame = true;
    }

    /*! \brief Processes the most recently received RGB frame.
     *  \note The frame is left on the GPU to be handled by OpenGL.
     *
     *  \return A flag to indicate whether a new frame was present.
     */
    bool updateFrame ()
    {
        Mutex::ScopedLock lock (rgbMutex);
        
        if (!newRGBFrame)
            return false;

        gFilter->process (rgbBuffer);

        newRGBFrame = false;

        return true;
    }

private:
    Mutex rgbMutex;
    cl_uchar *rgbBuffer;
    bool newRGBFrame;

};


/*! \brief Display callback for the window. */
void drawGLScene ()
{
    static uint8_t frameCount = 0;

    if (device->updateFrame ())
        frameCount++;

    glClear (GL_COLOR_BUFFER_BIT);
    glEnable (GL_TEXTURE_2D);

    glBindTexture (GL_TEXTURE_2D, glRGBTex);

    glBegin (GL_QUADS);
    glColor4f (1.f, 1.f, 1.f, 1.f);
    glVertex2i (0, 0); glTexCoord2f (1.f, 0.f);
    glVertex2i (imgWidth, 0); glTexCoord2f (1.f, 1.f);
    glVertex2i (imgWidth, imgHeight); glTexCoord2f (0.f, 1.f);
    glVertex2i (0, imgHeight); glTexCoord2f (0.f, 0.f);
    glEnd ();

    glBindTexture (GL_TEXTURE_2D, glRGBTexFilt);

    glBegin (GL_QUADS);
    glColor4f (1.f, 1.f, 1.f, 1.f);
    glVertex2i (imgWidth, 0); glTexCoord2f (1.f, 0.f);
    glVertex2i (imgWidth + imgWidth, 0); glTexCoord2f (1.f, 1.f);
    glVertex2i (imgWidth + imgWidth, imgHeight); glTexCoord2f (0.f, 1.f);
    glVertex2i (imgWidth, imgHeight); glTexCoord2f (0.f, 0.f);
    glEnd ();


    static float fps;
    std::ostringstream fpsBuffer;
    static std::string fpsStr;
    static std::chrono::time_point<std::chrono::system_clock> 
        timeRef = std::chrono::system_clock::now ();
    
    // Calculate and display the frame rate
    if (frameCount == 10)
    {
        std::chrono::duration<float> elapsed = 
            std::chrono::system_clock::now () - timeRef;
        fps = 10.f / elapsed.count ();

        fpsBuffer << std::fixed << std::setprecision (2) 
                  << std::setfill ('0') << std::setw (5) << fps << " fps";
        fpsStr = fpsBuffer.str ();

        frameCount = 0;
        timeRef = std::chrono::system_clock::now ();
    }
    
    glRasterPos2f (1210.f, 25.f);
    for (auto c : fpsStr)
        glutBitmapCharacter (GLUT_BITMAP_HELVETICA_12, c);

    // Display the filter parameters
    std::ostringstream paramStr;
    paramStr << "Radius: " << gFilter->getRadius ();
    glRasterPos2f (20.f, 445.f);
    for (auto c : paramStr.str ())
        glutBitmapCharacter (GLUT_BITMAP_HELVETICA_12, c);
    paramStr.str (std::string ());
    paramStr.clear ();
    paramStr << "Eps: " << std::fixed << std::setprecision (3) << gFilter->getEps ();
    glRasterPos2f (20.f, 460.f);
    for (auto c : paramStr.str ())
        glutBitmapCharacter (GLUT_BITMAP_HELVETICA_12, c);

    glutSwapBuffers ();
}


/*! \brief Idle callback for the window. */
void idleGLScene ()
{
    glutPostRedisplay ();
}


/*! \brief Reshape callback for the window. */
void resizeGLScene (int width, int height)
{
    glViewport (0, 0, width, height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    glOrtho (0.0, gl_win_width, gl_win_height, 0.0, -1.0, 1.0);
    glMatrixMode (GL_MODELVIEW);
}


/*! \brief Keyboard callback for the window. */
void keyPressed (unsigned char key, int x, int y)
{
    switch (key)
    {
        case 0x1B:  // ESC
        case  'Q':
        case  'q':
            glutDestroyWindow (glWinId);
            break;
        case  'I':
        case  'i':
            gFilter->radiusDown ();
            break;
        case  'O':
        case  'o':
            gFilter->resetRadius ();
            break;
        case  'P':
        case  'p':
            gFilter->radiusUp ();
            break;
        case  'J':
        case  'j':
            gFilter->epsDown ();
            break;
        case  'K':
        case  'k':
            gFilter->resetEps ();
            break;
        case  'L':
        case  'l':
            gFilter->epsUp ();
            break;
        case  'N':
        case  'n':
            gFilter->toggleRGBNorm ();
            break;
        case  'W':
        case  'w':
            if (++freenectAngle > 30)
                freenectAngle = 30;
            device->setTiltDegrees (freenectAngle);
            break;
        case  'S':
        case  's':
            if (--freenectAngle < -30)
                freenectAngle = -30;
            device->setTiltDegrees (freenectAngle);
            break;
        case  'R':
        case  'r':
            freenectAngle = 0;
            device->setTiltDegrees (freenectAngle);
            break;
        case  '1':
            device->setLed (LED_GREEN);
            break;
        case  '2':
            device->setLed (LED_RED);
            break;
        case  '3':
            device->setLed (LED_YELLOW);
            break;
        case  '4':
        case  '5':
            device->setLed (LED_BLINK_GREEN);
            break;
        case  '6':
            device->setLed (LED_BLINK_RED_YELLOW);
            break;
        case  '0':
            device->setLed (LED_OFF);
            break;
    }
}


/*! \brief Initializes GLUT. */
void initGL (int argc, char **argv)
{
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
    glutInitWindowSize (gl_win_width, gl_win_height);
    glutInitWindowPosition ((glutGet (GLUT_SCREEN_WIDTH) - gl_win_width) / 2,
                            (glutGet (GLUT_SCREEN_HEIGHT) - gl_win_height) / 2 - 70);
    glWinId = glutCreateWindow ("Guided Image Filtering on an RGB stream");

    glutDisplayFunc (&drawGLScene);
    glutIdleFunc (&idleGLScene);
    glutReshapeFunc (&resizeGLScene);
    glutKeyboardFunc (&keyPressed);

    glClearColor (0.f, 0.f, 0.f, 0.f);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glShadeModel (GL_SMOOTH);
}


// Displays the available controls 
void printInfo ()
{
    std::cout << "\nAvailable Controls:\n";
    std::cout << "===================\n";
    std::cout << " Adjust Filter Radius  [-/r/+] :  I/O/P\n";
    std::cout << " Adjust Filter Epsilon [-/r/+] :  J/K/L\n";
    std::cout << " Toggle RGB Normalization      :  N\n";
    std::cout << " Kinect Tilt Angle     [-/r/+] :  S/R/W\n";
    std::cout << " Update LED State              :  0-6\n";
    std::cout << " Quit                          :  Q or Esc\n\n";
}


int main (int argc, char **argv)
{
    try
    {
        printInfo ();

        device = &freenect.createDevice<MyFreenectDevice> (0);
        device->startVideo ();

        initGL (argc, argv);

        // The OpenCL environment must be created after the OpenGL environment 
        // has been initialized and before OpenGL starts rendering
        gFilter = new GFilterRGB ();
        
        glutMainLoop ();

        device->stopVideo ();
        delete gFilter;

        return 0;
    }
    catch (const std::runtime_error &error)
    {
        std::cerr << "Kinect: " << error.what () << std::endl;
    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
    }
    exit (EXIT_FAILURE);
}
