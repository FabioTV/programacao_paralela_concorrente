#pragma GCC optimize("O2")
#include <iostream>
#include <ctime>
#include <cmath>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/cudacodec.hpp"

int main() {
	cv::TickMeter tm;
	tm.start();

    cv::String filename = "DJI_0012.MOV";
    cv::VideoCapture cap(filename, cv::CAP_FFMPEG);
    int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter video_out("teste_gpu.mp4", codec, fps, cv::Size(frame_width, frame_height));

    cv::Mat frame, aux, aux2;
    std::queue<cv::Mat> frames;
    cv::cuda::GpuMat frame_gpu, frame2_gpu, frame3_gpu;

    //#pragma omp parallel 
    //#pragma omp for
    for (long i = 0; i < 500; i++) {
                
        cap >> frame;
        if (frame.empty())
            continue;
        frame_gpu.upload(frame);

        //cv::cuda::fastNlMeansDenoisingColored(frame_gpu, frame3_gpu, 1, 8, 21, 7);
        auto filter = cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(9, 9), 8);
        filter->apply(frame_gpu, frame3_gpu);
        std::cout << "1" << std::endl;
              

               
        frame3_gpu.download(aux);
        frames.push(aux);
        std::cout << "2" << std::endl;
        

        aux2 = frames.back();
        video_out.write(aux2);
        //std::cout << aux2;
        frames.pop();
        std::cout << "3" << std::endl;
               
    }
    


	tm.stop();
	std::cout << "Tempo total (segundos):" << tm.getTimeSec() << std::endl;

    return 0;
}