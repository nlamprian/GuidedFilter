/*! \file guidedFilter_kinect_point_cloud.cpp
 *  \brief An example showcasing the effect of the `Guided Filter` algorithm on depth images.
 *  \details This example demonstrates the performance of the [Guided Image Filtering]
 *           (http://research.microsoft.com/en-us/um/people/kahe/eccv10/) 
 *           algorithm on a live video stream. It processes the Kinect RGB 
 *           and Depth streams in OpenCL with the `GuidedFilter` pipeline. Then,
 *           it creates a point cloud and displays it in an OpenGL window.
 *  \author Nick Lamprianidis
 *  \version 1.2.0
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
#include <GL/glew.h>  // Add before CLUtils.hpp
#include <CLUtils.hpp>
#include <GuidedFilter/algorithms.hpp>

#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <libfreenect.hpp>


// Window parameters
const int gl_win_width = 640;
const int gl_win_height = 480;
int glWinId;

// Model parameters
int mouseX = -1, mouseY = -1;
float angleX = 0.f, angleY = 0.f;
float zoom = 1.f;

// GL texture IDs
GLuint glRGBBuf, glDepthBuf;

// Freenect
class MyFreenectDevice;
Freenect::Freenect freenect;
MyFreenectDevice *device;
double freenectAngle = 0;
float focalLength = 595.f;

// OpenCL
const std::vector<std::string> kernel_files = { "kernels/imageSupport_kernels.cl", 
                                                "kernels/scan_kernels.cl", 
                                                "kernels/transpose_kernels.cl", 
                                                "kernels/boxFilter_kernels.cl",
                                                "kernels/math_kernels.cl", 
                                                "kernels/guidedFilter_kernels.cl", 
                                                "kernels/examples_kernels.cl" };
const int imgWidth = 640;
const int imgHeight = 480;
const int dRadius = 5;
const float dEps = 0.02f;
const float dScaling = 1e-3f;
class GFilterPC;
GFilterPC *gFilter;

enum class Stream : uint8_t { RGB, DEPTH };


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
        glGenBuffers (1, &glRGBBuf);
        glBindBuffer (GL_ARRAY_BUFFER, glRGBBuf);
        glBufferData (GL_ARRAY_BUFFER, 4 * imgWidth * imgHeight * sizeof (cl_float), NULL, GL_DYNAMIC_DRAW);
        glGenBuffers (1, &glDepthBuf);
        glBindBuffer (GL_ARRAY_BUFFER, glDepthBuf);
        glBufferData (GL_ARRAY_BUFFER, 4 * imgWidth * imgHeight * sizeof (cl_float), NULL, GL_DYNAMIC_DRAW);
        glBindBuffer (GL_ARRAY_BUFFER, 0);
    }

};


/*! \brief Applies `Guided Image Filtering` on a `Kinect` RGB frame and 
 *         delivers the data to `OpenGL`.
 */
class GFilterPC
{
public:
    GFilterPC () : 
        env (), context (env.getContext (0)), 
        queue0 (env.getQueue (0, 0)), queue1 (env.getQueue (0, 1)), 
        kernelRGBGL (env.getProgram (0), "combineRGBGL_PC"), 
        global (imgWidth * imgHeight), info (0, 0, 0, { 0, 1 }, 0), 
        kGFRGB (env, info), kGFDepth (env, info), to3D (env, info.getCLEnvInfo (0)), 
        bufferSize (imgWidth * imgHeight * sizeof (cl_float)), 
        waitList (1), normalizeRGB (0)
    {
        size_t wgMultiple = kernelRGBGL.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[0][0]);

        // The local workspace should be a multiple of 3
        size_t wgM = wgMultiple;
        unsigned int pixels = imgWidth * imgHeight;
        while (pixels % (3 * wgM) != 0) wgM >>= 1;
        local = cl::NDRange (3 * wgM);

        // Create GL-shared buffers
        buffersGL.emplace_back (context, CL_MEM_WRITE_ONLY, glRGBBuf);
        buffersGL.emplace_back (context, CL_MEM_WRITE_ONLY, glDepthBuf);

        // Initialize the Guided Image Filtering pipeline
        kGFRGB.get (cl_algo::GF::Kinect::GuidedFilterRGB<cl_algo::GF::Kinect::GuidedFilterRGBConfig::SEPARATED>
            ::Memory::D_OUT_R) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        kGFRGB.get (cl_algo::GF::Kinect::GuidedFilterRGB<cl_algo::GF::Kinect::GuidedFilterRGBConfig::SEPARATED>
            ::Memory::D_OUT_G) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        kGFRGB.get (cl_algo::GF::Kinect::GuidedFilterRGB<cl_algo::GF::Kinect::GuidedFilterRGBConfig::SEPARATED>
            ::Memory::D_OUT_B) = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        kGFRGB.init (imgWidth, imgHeight, dRadius, dEps, cl_algo::GF::Staging::I);

        // Set arguments for the kernel responsible for handling the filtered RGB frame
        kernelRGBGL.setArg (0, kGFRGB.get (cl_algo::GF::Kinect::GuidedFilterRGB<cl_algo::GF
            ::Kinect::GuidedFilterRGBConfig::SEPARATED>::Memory::D_OUT_R));
        kernelRGBGL.setArg (1, kGFRGB.get (cl_algo::GF::Kinect::GuidedFilterRGB<cl_algo::GF
            ::Kinect::GuidedFilterRGBConfig::SEPARATED>::Memory::D_OUT_G));
        kernelRGBGL.setArg (2, kGFRGB.get (cl_algo::GF::Kinect::GuidedFilterRGB<cl_algo::GF
            ::Kinect::GuidedFilterRGBConfig::SEPARATED>::Memory::D_OUT_B));
        kernelRGBGL.setArg (3, buffersGL[0]);
        kernelRGBGL.setArg (4, cl::Local (3 * local[0] * sizeof (cl_float)));
        kernelRGBGL.setArg (5, imgWidth);
        kernelRGBGL.setArg (6, normalizeRGB);

        // The depth frames are being scaled down before processing by 
        // GuidedFilter and are being scaled back up after GuidedFilter and 
        // before building the point cloud. The actual scale of the data has 
        // a strong effect on the resulting point cloud.
        kGFDepth.get (cl_algo::GF::Kinect::GuidedFilterDepth::Memory::D_OUT) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);
        kGFDepth.init (imgWidth, imgHeight, dRadius, dEps, dScaling, cl_algo::GF::Staging::I);

        to3D.get (cl_algo::GF::DepthTo3D::Memory::D_IN) = 
            kGFDepth.get (cl_algo::GF::Kinect::GuidedFilterDepth::Memory::D_OUT);
        to3D.get (cl_algo::GF::DepthTo3D::Memory::D_OUT) = buffersGL[1];
        to3D.init (imgWidth, imgHeight, focalLength, 1.f, cl_algo::GF::Staging::NONE);
    }

    /*! \brief Processes RGB and Depth frames on the GPU.
     *  \details The processed frame is delivered directly to OpenGL from the GPU.
     *  
     *  \param[in] rgb RGB frame to be processed.
     *  \param[in] depth depth frame to be processed.
     */
    void process (cl_uchar *rgb, cl_ushort *depth)
    {
        // Transfer data to device
        kGFRGB.write (cl_algo::GF::Kinect::GuidedFilterRGB<cl_algo::GF
            ::Kinect::GuidedFilterRGBConfig::SEPARATED>::Memory::D_IN, rgb);
        kGFDepth.write (cl_algo::GF::Kinect::GuidedFilterDepth::Memory::D_IN, depth);

        glFinish ();  // Wait for OpenGL pending operations on buffers to finish

        // Take ownership of OpenGL textures
        queue1.enqueueAcquireGLObjects ((std::vector<cl::Memory> *) &buffersGL);

        // Dispatch kernels
        kGFRGB.run ();
        kGFDepth.run ();
        queue1.enqueueNDRangeKernel (kernelRGBGL, cl::NullRange, global, local, nullptr, &event);
        to3D.run ();
        waitList[0] = event;

        // Give up ownership of OpenGL textures
        queue0.enqueueReleaseGLObjects ((std::vector<cl::Memory> *) &buffersGL, &waitList);

        queue0.finish ();
    }

    void radiusUp (Stream frame)
    {
        if (frame == Stream::RGB)
            kGFRGB.setRadius (kGFRGB.getRadius () + 1);
        else
            kGFDepth.setRadius (kGFDepth.getRadius () + 1);
    }

    void radiusDown (Stream frame)
    {
        if (frame == Stream::RGB)
        {
            if (kGFRGB.getRadius () == 1) return;
            kGFRGB.setRadius (kGFRGB.getRadius () - 1);
        }
        else
        {
            if (kGFDepth.getRadius () == 1) return;
            kGFDepth.setRadius (kGFDepth.getRadius () - 1);
        }
    }

    void resetRadius (Stream frame)
    {
        if (frame == Stream::RGB)
            kGFRGB.setRadius (dRadius);
        else
            kGFDepth.setRadius (dRadius);
    }

    int getRadius (Stream frame)
    {
        if (frame == Stream::RGB)
            return kGFRGB.getRadius ();
        else
            return kGFDepth.getRadius ();
    }

    void epsUp (Stream frame)
    {
        if (frame == Stream::RGB)
            kGFRGB.setEps (kGFRGB.getEps () + 0.005f);
        else
            kGFDepth.setEps (kGFDepth.getEps () + 0.005f);
    }

    void epsDown (Stream frame)
    {
        if (frame == Stream::RGB)
        {
            float tmp = kGFRGB.getEps () - 0.005f;
            if (tmp < 0.f) return;
            kGFRGB.setEps (tmp);
        }
        else
        {
            float tmp = kGFDepth.getEps () - 0.005f;
            if (tmp < 0.f) return;
            kGFDepth.setEps (tmp);
        }
    }

    void resetEps (Stream frame)
    {
        if (frame == Stream::RGB)
            kGFRGB.setEps (dEps);
        else
            kGFDepth.setEps (dEps);
    }

    float getEps (Stream frame)
    {
        if (frame == Stream::RGB)
            return kGFRGB.getEps ();
        else
            return kGFDepth.getEps ();
    }

    void toggleRGBNorm ()
    {
        normalizeRGB = !normalizeRGB;
        kernelRGBGL.setArg (6, normalizeRGB);
    }

private:
    CLEnvGL env;
    cl::Context &context;
    cl::CommandQueue &queue0, &queue1;
    cl::Kernel kernelRGBGL;
    std::vector<cl::BufferGL> buffersGL;  /*!< GL-shared buffers */
    cl::NDRange global, local;
    cl::Event event;
    std::vector<cl::Event> waitList;
    clutils::CLEnvInfo<2> info;
    cl_algo::GF::Kinect::GuidedFilterRGB<cl_algo::GF::Kinect::GuidedFilterRGBConfig::SEPARATED> kGFRGB;
    cl_algo::GF::Kinect::GuidedFilterDepth kGFDepth;
    cl_algo::GF::DepthTo3D to3D;
    unsigned int bufferSize;
    int normalizeRGB;

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
        Freenect::FreenectDevice (ctx, idx), newRGBFrame (false), newDepthFrame (false)
    {
        // setVideoFormat (FREENECT_VIDEO_YUV_RGB, FREENECT_RESOLUTION_MEDIUM);
        setDepthFormat (FREENECT_DEPTH_REGISTERED);

        rgbBuffer = new cl_uchar[freenect_find_video_mode (
            FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes];
        depthBuffer = new cl_ushort[freenect_find_depth_mode (
            FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED).bytes / 2];
    }

    virtual ~MyFreenectDevice ()
    {
        delete[] rgbBuffer;
        delete[] depthBuffer;
    }

    /*! \brief Delivers the latest RGB frame.
     *  \note Do not call directly, it's only used by the library.
     *  
     *  \param[in] rgb an array holding the rgb frame.
     *  \param[in] timestamp a time stamp.
     */
    void VideoCallback (void *rgb, uint32_t timestamp)
    {
        std::lock_guard<std::mutex> lock (rgbMutex);
        
        std::copy ((cl_uchar *) rgb, (cl_uchar *) rgb + getVideoBufferSize (), rgbBuffer);
        newRGBFrame = true;
    }


    /*! \brief Delivers the latest Depth frame.
     *  \note Do not call directly, it's only used by the library.
     *  
     *  \param[in] depth an array holding the depth frame.
     *  \param[in] timestamp a time stamp.
     */
    void DepthCallback (void *depth, uint32_t timestamp)
    {
        std::lock_guard<std::mutex> lock (depthMutex);
        
        std::copy ((cl_ushort *) depth, (cl_ushort *) depth + getDepthBufferSize () / 2, depthBuffer);
        newDepthFrame = true;
    }


    /*! \brief Processes the most recently received RGB and Depth frames.
     *  \note The frames are left on the GPU to be handled by OpenGL.
     *
     *  \return A flag to indicate whether new frames were present.
     */
    bool updateFrames ()
    {
        std::lock_guard<std::mutex> lockRGB (rgbMutex);
        std::lock_guard<std::mutex> lockDepth (depthMutex);
        
        if (!newRGBFrame || !newDepthFrame)
            return false;

        gFilter->process (rgbBuffer, depthBuffer);

        newRGBFrame = false;
        newDepthFrame = false;

        return true;
    }

private:
    std::mutex rgbMutex, depthMutex;
    cl_uchar *rgbBuffer;
    cl_ushort *depthBuffer;
    bool newRGBFrame, newDepthFrame;

};


/*! \brief Display callback for the window. */
void drawGLScene ()
{
    static uint8_t frameCount = 0;

    if (device->updateFrames ())
        frameCount++;

    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindBuffer (GL_ARRAY_BUFFER, glDepthBuf);
    glVertexPointer (4, GL_FLOAT, 0, NULL);
    glEnableClientState (GL_VERTEX_ARRAY);
    
    glBindBuffer (GL_ARRAY_BUFFER, glRGBBuf);
    glColorPointer (4, GL_FLOAT, 0, NULL);
    glEnableClientState (GL_COLOR_ARRAY);
    
    glDrawArrays (GL_POINTS, 0, imgWidth * imgHeight);

    glDisableClientState (GL_VERTEX_ARRAY);
    glDisableClientState (GL_COLOR_ARRAY);
    glBindBuffer (GL_ARRAY_BUFFER, 0);

    // Draw the world coordinate frame
    glLineWidth (2.f);
    glBegin (GL_LINES);
    glColor3ub (255, 0, 0);
    glVertex3i (  0, 0, 0);
    glVertex3i ( 50, 0, 0);

    glColor3ub (0, 255, 0);
    glVertex3i (0,   0, 0);
    glVertex3i (0,  50, 0);

    glColor3ub (0, 0, 255);
    glVertex3i (0, 0,   0);
    glVertex3i (0, 0,  50);
    glEnd ();

    // Position the camera
    glMatrixMode (GL_MODELVIEW);
    glLoadIdentity ();
    glScalef (zoom, zoom, 1);
    gluLookAt ( -7*angleX, -7*angleY, -1000.0,
                      0.0,       0.0,  2000.0,
                      0.0,      -1.0,     0.0 );


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
    
    glColor3ub (255, 255, 0);
    glRasterPos3f (20.f, 30.f, -20.f);
    for (auto c : fpsStr)
        glutBitmapCharacter (GLUT_BITMAP_HELVETICA_12, c);

    // Display the filter parameters
    std::ostringstream paramStr;
    paramStr << "C.Radius: " << gFilter->getRadius (Stream::RGB);
    glRasterPos3f (20.f, 60.f, -20.f);
    for (auto c : paramStr.str ())
        glutBitmapCharacter (GLUT_BITMAP_HELVETICA_12, c);
    paramStr.str (std::string ());
    paramStr.clear ();
    paramStr << "C.Eps: " << std::fixed << std::setprecision (3) << gFilter->getEps (Stream::RGB);
    glRasterPos3f (20.f, 90.f, -20.f);
    for (auto c : paramStr.str ())
        glutBitmapCharacter (GLUT_BITMAP_HELVETICA_12, c);
    paramStr.str (std::string ());
    paramStr.clear ();
    paramStr << "D.Radius: " << gFilter->getRadius (Stream::DEPTH);
    glRasterPos3f (20.f, 120.f, -20.f);
    for (auto c : paramStr.str ())
        glutBitmapCharacter (GLUT_BITMAP_HELVETICA_12, c);
    paramStr.str (std::string ());
    paramStr.clear ();
    paramStr << "D.Eps: " << std::fixed << std::setprecision (3) << gFilter->getEps (Stream::DEPTH);
    glRasterPos3f (20.f, 150.f, -20.f);
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
    gluPerspective (50.0, width / (float) height, 900.0, 11000.0);
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
            gFilter->radiusDown (Stream::RGB);
            break;
        case  'O':
        case  'o':
            gFilter->resetRadius (Stream::RGB);
            break;
        case  'P':
        case  'p':
            gFilter->radiusUp (Stream::RGB);
            break;
        case  'J':
        case  'j':
            gFilter->epsDown (Stream::RGB);
            break;
        case  'K':
        case  'k':
            gFilter->resetEps (Stream::RGB);
            break;
        case  'L':
        case  'l':
            gFilter->epsUp (Stream::RGB);
            break;
        case  'N':
        case  'n':
            gFilter->toggleRGBNorm ();
            break;
        case  'F':
        case  'f':
            gFilter->radiusDown (Stream::DEPTH);
            break;
        case  'G':
        case  'g':
            gFilter->resetRadius (Stream::DEPTH);
            break;
        case  'H':
        case  'h':
            gFilter->radiusUp (Stream::DEPTH);
            break;
        case  'C':
        case  'c':
            gFilter->epsDown (Stream::DEPTH);
            break;
        case  'V':
        case  'v':
            gFilter->resetEps (Stream::DEPTH);
            break;
        case  'B':
        case  'b':
            gFilter->epsUp (Stream::DEPTH);
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


/*! \brief Mouse callback for the window. */
void mouseMoved (int x, int y)
{
    if (mouseX >= 0 && mouseY >= 0)
    {
        angleX += x - mouseX;
        angleY += y - mouseY;
    }

    mouseX = x;
    mouseY = y;
}


/*! \brief Mouse button callback for the window. */
void mouseButtonPressed (int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        switch (button)
        {
            case GLUT_LEFT_BUTTON:
                mouseX = x;
                mouseY = y;
                break;
            case 3:  // Scroll Up
                zoom *= 1.2f;
                break;
            case 4:  // Scroll Down
                zoom /= 1.2f;
                break;
        }
    }
    else if (state == GLUT_UP && button == GLUT_LEFT_BUTTON)
    {
        mouseX = -1;
        mouseY = -1;
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
    glWinId = glutCreateWindow ("Guided Image Filtering on RGB and Depth streams");

    glutDisplayFunc (&drawGLScene);
    glutIdleFunc (&idleGLScene);
    glutReshapeFunc (&resizeGLScene);
    glutKeyboardFunc (&keyPressed);
    glutMotionFunc (&mouseMoved);
    glutMouseFunc (&mouseButtonPressed);

    glewInit ();

    glClearColor (0.65f, 0.65f, 0.65f, 1.f);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_ALPHA_TEST);
    glAlphaFunc (GL_GREATER, 0.f);
    glEnable (GL_DEPTH_TEST);
    glShadeModel (GL_SMOOTH);
}


// Displays the available controls 
void printInfo ()
{
    std::cout << "\nAvailable Controls:\n";
    std::cout << "===================\n";
    std::cout << " Rotate                              :  Mouse Left Button\n";
    std::cout << " Zoom In/Out                         :  Mouse Wheel\n";
    std::cout << " Adjust RGB Filter Radius    [-/r/+] :  I/O/P\n";
    std::cout << " Adjust RGB Filter Epsilon   [-/r/+] :  J/K/L\n";
    std::cout << " Toggle RGB Normalization            :  N\n";
    std::cout << " Adjust Depth Filter Radius  [-/r/+] :  F/G/H\n";
    std::cout << " Adjust Depth Filter Epsilon [-/r/+] :  C/V/B\n";
    std::cout << " Kinect Tilt Angle           [-/r/+] :  S/R/W\n";
    std::cout << " Update LED State                    :  0-6\n";
    std::cout << " Quit                                :  Q or Esc\n\n";
}


int main (int argc, char **argv)
{
    try
    {
        printInfo ();

        device = &freenect.createDevice<MyFreenectDevice> (0);
        device->startVideo ();
        device->startDepth ();

        initGL (argc, argv);

        // The OpenCL environment must be created after the OpenGL environment 
        // has been initialized and before OpenGL starts rendering
        gFilter = new GFilterPC ();
        
        glutMainLoop ();

        device->stopVideo ();
        device->stopDepth ();
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
