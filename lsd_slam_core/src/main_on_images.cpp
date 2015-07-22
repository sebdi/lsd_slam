/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LiveSLAMWrapper.h"

#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "SlamSystem.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "IOWrapper/ROS/ROSOutput3DWrapper.h"
#include "IOWrapper/ROS/rosReconfigure.h"

#include "util/Undistorter.h"
#include <ros/package.h>

#include "opencv2/opencv.hpp"
#include "DepthEstimation/DepthMapPixelHypothesis.h"

struct veloPoint{
    float x;
    float y;
    float z;
    float i;
};

using namespace lsd_slam;

DepthMapPixelHypothesis* projectLidarInImage(int width,int height, std::vector<veloPoint> &velpoints)
{
    Eigen::Matrix3d velo_to_cam_R;
    velo_to_cam_R << 7.967514e-03, -9.999679e-01, -8.462264e-04, -2.771053e-03, 8.241710e-04, -9.999958e-01, 9.999644e-01, 7.969825e-03, -2.764397e-03;
    Eigen::Vector3d velo_to_cam_t;
    velo_to_cam_t << -1.377769e-02, -5.542117e-02, -2.918589e-01;
    Eigen::Matrix4d T;
    T << velo_to_cam_R(0,0), velo_to_cam_R(0,1), velo_to_cam_R(0,2), velo_to_cam_t(0),
            velo_to_cam_R(1,0), velo_to_cam_R(1,1), velo_to_cam_R(1,2), velo_to_cam_t(1),
            velo_to_cam_R(2,0), velo_to_cam_R(2,1), velo_to_cam_R(2,2), velo_to_cam_t(2),
            0 , 0 , 0 ,1;
    Eigen::Matrix3Xd P0(3,4);
    P0 << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00, 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00;
    Eigen::Vector4d X;
    Eigen::Vector4d X_temp;
    Eigen::Vector3d x;
    DepthMapPixelHypothesis * depth = new DepthMapPixelHypothesis[width*height];

    for (size_t j=0;j<velpoints.size();j++)
    {
        X << velpoints[j].x, velpoints[j].y, velpoints[j].z, 1;
        X_temp = T * X;
        if (X_temp(2)>0)
        {
            x = P0 * T * X;
            x = x/x(2);
            int u = (int)ceil(x(0));
            int v = (int)ceil(x(1));
            if (u<width-15 && v<height-15 && u>15 && v>15)
            {
                depth[u+v*width].idepth = depth[u+v*width].idepth_smoothed = 1.0f/(double)X_temp(2);
                depth[u+v*width].idepth_var = depth[u+v*width].idepth_var_smoothed = 0.01f;
                depth[u+v*width].isValid = true;
                for (size_t j = 1;j<2;j++)
                {
                    depth[(u-j)+v*width].idepth = depth[(u-j)+v*width].idepth_smoothed = 1.0f/(double)X_temp(2);
                    depth[(u-j)+v*width].idepth_var = depth[(u-j)+v*width].idepth_var_smoothed = 0.4f*j;
                    depth[(u-j)+v*width].isValid = true;
                    depth[(u+j)+v*width].idepth = depth[(u+j)+v*width].idepth_smoothed = 1.0f/(double)X_temp(2);
                    depth[(u+j)+v*width].idepth_var = depth[(u+j)+v*width].idepth_var_smoothed = 0.4f*j;
                    depth[(u+j)+v*width].isValid = true;
                    depth[u+(v+j)*width].idepth = depth[u+(v+j)*width].idepth_smoothed = 1.0f/(double)X_temp(2);
                    depth[u+(v+j)*width].idepth_var = depth[u+(v+j)*width].idepth_var_smoothed = 0.4f*j;
                    depth[u+(v+j)*width].isValid = true;
                    depth[u+(v-j)*width].idepth = depth[u+(v-j)*width].idepth_smoothed = 1.0f/(double)X_temp(2);
                    depth[u+(v-j)*width].idepth_var = depth[u+(v-j)*width].idepth_var_smoothed = 0.4f*j;
                    depth[u+(v-j)*width].isValid = true;
                    depth[(u-j)+(v-j)*width].idepth = depth[(u-j)+(v-j)*width].idepth_smoothed = 1.0f/(double)X_temp(2);
                    depth[(u-j)+(v-j)*width].idepth_var = depth[(u-j)+(v-j)*width].idepth_var_smoothed = 0.4f*j;
                    depth[(u-j)+(v-j)*width].isValid = true;
                    depth[(u+j)+(v+j)*width].idepth = depth[(u+j)+(v+j)*width].idepth_smoothed = 1.0f/(double)X_temp(2);
                    depth[(u+j)+(v+j)*width].idepth_var = depth[(u+j)+(v+j)*width].idepth_var_smoothed = 0.4f*j;
                    depth[(u+j)+(v+j)*width].isValid = true;
                }
                //depthPoints++;
            }
        }
    }
    return depth;
}


int getVel(std::vector<std::string> &files, std::vector<std::vector<veloPoint>> &points, int num)
{
    for (size_t j=0; j<files.size() && j<num;j++)
    {
        // allocate 4 MB buffer (only ~130*4*4 KB are needed)
        int32_t num = 1000000;
        float *data = (float*)malloc(num*sizeof(float));

        // pointers
        float *px = data+0;
        float *py = data+1;
        float *pz = data+2;
        float *pr = data+3;

        // load point cloud
        FILE *stream;
        stream = fopen (files[j].c_str(),"rb");
        num = fread(data,sizeof(float),num,stream)/4;
        std::vector<veloPoint> temp;
        for (int32_t i=0; i<num; i++) {
            //std::cout << "i=" << i << std::endl;
            //point_cloud.points.push_back(tPoint(*px,*py,*pz,*pr));
            veloPoint point;
            point.x = *px;
            point.y = *py;
            point.z = *pz;
            point.i = *pr;
            temp.push_back(point);
            px+=4; py+=4; pz+=4; pr+=4;
        }
        points.push_back(temp);
        fclose(stream);
    }
    return points.size();
}

std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}
std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}
std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}
int getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
        std::string name = std::string(dirp->d_name);

        if(name != "." && name != "..")
            files.push_back(name);
    }
    closedir(dp);


    std::sort(files.begin(), files.end());

    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
    for(unsigned int i=0;i<files.size();i++)
    {
        if(files[i].at(0) != '/')
            files[i] = dir + files[i];
    }

    return files.size();
}

int getFile (std::string source, std::vector<std::string> &files)
{
    std::ifstream f(source.c_str());

    if(f.good() && f.is_open())
    {
        while(!f.eof())
        {
            std::string l;
            std::getline(f,l);

            l = trim(l);

            if(l == "" || l[0] == '#')
                continue;

            files.push_back(l);
        }

        f.close();

        size_t sp = source.find_last_of('/');
        std::string prefix;
        if(sp == std::string::npos)
            prefix = "";
        else
            prefix = source.substr(0,sp);

        for(unsigned int i=0;i<files.size();i++)
        {
            if(files[i].at(0) != '/')
                files[i] = prefix + "/" + files[i];
        }

        return (int)files.size();
    }
    else
    {
        f.close();
        return -1;
    }

}



int main( int argc, char** argv )
{
    ros::init(argc, argv, "LSD_SLAM");

    dynamic_reconfigure::Server<lsd_slam_core::LSDParamsConfig> srv(ros::NodeHandle("~"));
    srv.setCallback(dynConfCb);

    dynamic_reconfigure::Server<lsd_slam_core::LSDDebugParamsConfig> srvDebug(ros::NodeHandle("~Debug"));
    srvDebug.setCallback(dynConfCbDebug);

    packagePath = ros::package::getPath("lsd_slam_core")+"/";



    // get camera calibration in form of an undistorter object.
    // if no undistortion is required, the undistorter will just pass images through.
    std::string calibFile;
    Undistorter* undistorter = 0;
    if(ros::param::get("~calib", calibFile))
    {
        undistorter = Undistorter::getUndistorterForFile(calibFile.c_str());
        ros::param::del("~calib");
    }

    if(undistorter == 0)
    {
        printf("need camera calibration file! (set using _calib:=FILE)\n");
        exit(0);
    }

    int w = undistorter->getOutputWidth();
    int h = undistorter->getOutputHeight();

    int w_inp = undistorter->getInputWidth();
    int h_inp = undistorter->getInputHeight();

    float fx = undistorter->getK().at<double>(0, 0);
    float fy = undistorter->getK().at<double>(1, 1);
    float cx = undistorter->getK().at<double>(2, 0);
    float cy = undistorter->getK().at<double>(2, 1);
    Sophus::Matrix3f K;
    K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;


    // make output wrapper. just set to zero if no output is required.
    Output3DWrapper* outputWrapper = new ROSOutput3DWrapper(w,h);






    // open image files: first try to open as file.
    std::string source;
    std::vector<std::string> files;
    if(!ros::param::get("~files", source))
    {
        printf("need source files! (set using _files:=FOLDER)\n");
        exit(0);
    }
    ros::param::del("~files");


    if(getdir(source, files) >= 0)
    {
        printf("found %d image files in folder %s!\n", (int)files.size(), source.c_str());
    }
    else if(getFile(source, files) >= 0)
    {
        printf("found %d image files in file %s!\n", (int)files.size(), source.c_str());
    }
    else
    {
        printf("could not load file list! wrong path / file?\n");
    }

    // read velo data
    std::string path_to_velo = "/home/sebastian/Dropbox/KITTI/velo/00/velodyne";
    std::vector<std::string> velo_files;
    if(getdir(path_to_velo, velo_files) >= 0)
    {
        printf("found %d point cloud files in folder %s!\n", (int)velo_files.size(), path_to_velo.c_str());
    }
    else if(getFile(path_to_velo, velo_files) >= 0)
    {
        printf("found %d point cloud files in file %s!\n", (int)velo_files.size(), path_to_velo.c_str());
    }
    else
    {
        printf("could not load file list! wrong path / file?\n");
    }
    std::vector<std::vector<veloPoint>> velpoints;
    getVel(velo_files,velpoints,100);


    // get HZ
    double hz = 0;
    if(!ros::param::get("~hz", hz))
        hz = 0;
    ros::param::del("~hz");



    cv::Mat image = cv::Mat(h,w,CV_8U);
    int runningIDX=0;
    float fakeTimeStamp = 0;

    ros::Rate r(hz);

    // make slam system
    SlamSystem* system = new SlamSystem(w, h, K, doSlam);
    system->setVisualization(outputWrapper);

    for(unsigned int i=0;i<files.size();i++)
    {
        cv::Mat imageDist = cv::imread(files[i], CV_LOAD_IMAGE_GRAYSCALE);

        if(imageDist.rows != h_inp || imageDist.cols != w_inp)
        {
            if(imageDist.rows * imageDist.cols == 0)
                printf("failed to load image %s! skipping.\n", files[i].c_str());
            else
                printf("image %s has wrong dimensions - expecting %d x %d, found %d x %d. Skipping.\n",
                       files[i].c_str(),
                       w_inp,h_inp,imageDist.cols, imageDist.rows);
            continue;
        }
        assert(imageDist.type() == CV_8U);

        undistorter->undistort(imageDist, image);
        assert(image.type() == CV_8U);

        DepthMapPixelHypothesis * depthMapPixel;
        float * depth = new float[w*h];
        for (int x=0;x<w;x++)
        {
            for (int y=0;y<h;y++)
            {
                depth[x+y*w] = -1.0f;
            }
        }
        if(runningIDX == 0)
        {

            if (velpoints.size() !=0)
            {

                depthMapPixel = projectLidarInImage(w,h,velpoints[i]);
                for (int x=0;x<w;x++)
                {
                    for (int y=0;y<h;y++)
                    {
                        depth[x+y*w] = 1.0f/depthMapPixel[x+y*w].idepth;
                    }
                }

            }
            else
            {
                depth = 0;
            }
            //system->gtDepthInit(image.data,depth,fakeTimeStamp,runningIDX);
            system->randomInit(image.data, fakeTimeStamp, runningIDX);
        }
        else
        {


            if (velpoints.size() !=0)
            {

                depthMapPixel = projectLidarInImage(w,h,velpoints[i]);


            }
            else
            {
                depthMapPixel = 0;
            }

            system->trackFrame(image.data, runningIDX ,hz == 0,fakeTimeStamp,depthMapPixel);
        }
        runningIDX++;
        fakeTimeStamp+=0.03;

        if(hz != 0)
            r.sleep();

        if(fullResetRequested)
        {

            printf("FULL RESET!\n");
            delete system;

            system = new SlamSystem(w, h, K, doSlam);
            system->setVisualization(outputWrapper);

            fullResetRequested = false;
            runningIDX = 0;
        }

        ros::spinOnce();

        if(!ros::ok())
            break;
    }


    system->finalize();



    delete system;
    delete undistorter;
    delete outputWrapper;
    return 0;
}
