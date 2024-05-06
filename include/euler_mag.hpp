#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>

#include "timer.h"

class EulerMag
{
    public:
    
    EulerMag();
    ~EulerMag();

    bool init();
    void run();

    std::string getInputFilename() const { return _input_file_name; }
    void setInputFilename(const std::string input_file_name) { _input_file_name = input_file_name; }

    std::string getOutputFileName() const { return _output_file_name; }
    void setOutputFileName(const std::string output_file_name) { _output_file_name = output_file_name; }

    double getAlpha() const { return _alpha; }
    void setAlpha(double alpha) { _alpha = alpha; }

    double getChromAttenuation() const { return _chromatic_attenuation; }
    void setChromAttenuation(double chromAttenuation) { _chromatic_attenuation = chromAttenuation; }

    double getCutoffFreqHigh() const { return _cutoff_frequency_high; }
    void setCutoffFreqHigh(double cutoffFreqHigh) { _cutoff_frequency_high = cutoffFreqHigh; }

    double getCutoffFreqLow() const { return _cutoff_frequency_low; }
    void setCutoffFreqLow(double cutoffFreqLow) { _cutoff_frequency_low = cutoffFreqLow; }

    double getDelta() const { return _delta; }
    void setDelta(double delta) { _delta = delta; }

    double getExaggerationFactor() const { return _exaggeration; }
    void setExaggerationFactor(double exaggerationFactor) { _exaggeration = exaggerationFactor; }

    double getLambda() const { return _lambda; }
    void setLambda(double lambda) { _lambda = lambda; }

    double getLambdaC() const { return _lambda_c; }
    void setLambdaC(double lambdaC) { _lambda_c = lambdaC; }

    int getLapPyramidLevels() const { return _laplacian_pyramid_depth; }
    void setLapPyramidLevels(int lapPyramidLevels) { _laplacian_pyramid_depth = lapPyramidLevels; }
    
    private:

    int getCodecNumber(std::string file_name);
    void makeLaplacianPyramid();
    void temporalFilter();
    void amplify();
    void reconstructFromLaplacianPyramid();
    void attenuate();

    std::string _input_file_name;
    int _input_img_width;
    int _input_img_height;

    cv::VideoCapture* _input;
    int _frame_count;
    int _input_fps;
    cv::cuda::GpuMat _input_frame;

    // Laplacian Pyramid
    std::vector<cv::cuda::GpuMat> _laplacian_pyramid;
    int _laplacian_pyramid_depth;

    // Temporal Filter
    std::vector<cv::cuda::GpuMat> _lowpass_1;
    std::vector<cv::cuda::GpuMat> _lowpass_2;
    std::vector<cv::cuda::GpuMat> _filtered;
    double _cutoff_frequency_high;
    double _cutoff_frequency_low;
    cv::cuda::GpuMat _motion;

    // Chromatic attenuation
    double _chromatic_attenuation;

    cv::VideoWriter* _output;
    cv::Mat _output_frame;
    std::string _output_file_name;
    int _output_img_width;
    int _output_img_height;

    // General parameters
    double _lambda_c;
    double _alpha;
    double _delta;
    double _exaggeration;
    double _lambda;

    Timer timer_;

    

};