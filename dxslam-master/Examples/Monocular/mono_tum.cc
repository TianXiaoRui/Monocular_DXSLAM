/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>
#include <unistd.h>
#include<opencv2/core/core.hpp>
#include "cnpy.h"


using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

void getdescriptor(string filename,cv::Mat & descriptor,int nkeypoints);
void getGlobaldescriptor(string filename,cv::Mat & descriptor);
void getKeyPoint(string filename , vector<cv::KeyPoint> & keyPoints);


int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence path_to_feature" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    string featureFolder = string(argv[4]);


    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    DXSLAM::System SLAM(argv[1],argv[2],DXSLAM::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif
// Pass the image and hf-net output to the SLAM system
        cv::Mat local_desc;
        cv::Mat global_desc;
        vector<cv::KeyPoint> keypoints;

        // Get keyPoint,local descriptor and global descriptor
        getKeyPoint(featureFolder + "/point-txt/" + to_string(vTimestamps[ni]) + ".txt" , keypoints);
        local_desc.create(keypoints.size(), 256, CV_32F);
        getdescriptor(featureFolder + "/des/" + to_string(vTimestamps[ni]) + ".npy" , local_desc , keypoints.size());
        global_desc.create(4096, 1, CV_32F);
        getGlobaldescriptor(featureFolder + "/glb/" + to_string(vTimestamps[ni]) + ".npy" , global_desc);

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe,keypoints , local_desc , global_desc);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("FrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

void getdescriptor(string filename,cv::Mat & descriptor,int nkeypoints){
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    for(int i = 0 ; i < nkeypoints ; i ++){
        float* pdata= descriptor.ptr<float>(i);
        for(int j = 0 ; j < 256 ; j ++ ){
            float temp = arr.data<float>()[i *256 + j];
            pdata[j]= temp;
        }
    }
}

void getGlobaldescriptor(string filename,cv::Mat & descriptor){
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    float* pdata= descriptor.ptr<float>(0);
    for(int j = 0 ; j < 4096 ; j ++ ){
        pdata[j]= arr.data<float>()[j];
    }
}

void getKeyPoint(string filename , vector<cv::KeyPoint> & keyPoints){
    ifstream getfile(filename);

    for(int i = 0 ; i < 550 && !getfile.eof()  ; i++)
    {
        string s;
        getline(getfile,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t_x;
            double t_y;
            ss >> t_x;
            ss >> t_y;
            cv::KeyPoint keyPoint (t_x,t_y,1);
            keyPoint.octave = 0;
            keyPoints.push_back(keyPoint);
        }
    }
}