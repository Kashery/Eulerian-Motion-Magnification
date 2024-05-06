#include "euler_mag.hpp"

#define DISPLAY_WINDOW "Euler Mag"

EulerMag::EulerMag()
    : _input_file_name()
    , _output_file_name("result.mp4")
    , _cutoff_frequency_high(1.0)
    , _cutoff_frequency_low(0.01)
    , _laplacian_pyramid_depth(5)
    , _lambda_c(16.0)
    , _alpha(10.0)
    , _exaggeration(10.0)
    , _chromatic_attenuation(0.1)
{
}

EulerMag::~EulerMag()
{
    if (_input != NULL)
        _input->release();

    if (_output != NULL)
        _output->release();
}

bool EulerMag::init()
{
    // Calculate the spatial frequency bands
    

    // Instantiate video capture
    _input = new cv::VideoCapture(_input_file_name);

    if (!_input->isOpened())
    {
        std::cerr << "Error: cannot open file \"" << _input_file_name << "\"" << std::endl;
        return false;
    }

    // Set input size
    _input_img_width = _input->get(cv::CAP_PROP_FRAME_WIDTH);
    
    _input_img_height = _input->get(cv::CAP_PROP_FRAME_HEIGHT);


    // Set frame info about the video capture
    _frame_count = _input->get(cv::CAP_PROP_FRAME_COUNT);
    _input_fps = _input->get(cv::CAP_PROP_FPS);

    // Instantiate output preview window
    cv::namedWindow(DISPLAY_WINDOW, cv::WINDOW_AUTOSIZE);

    // Check if output file was provided
    if (!_output_file_name.empty())
    {
        // Set output file info
        _output_img_width = _input_img_width;
        _output_img_height = _input_img_height;

        // Instantiate video writer
        _output = new cv::VideoWriter(
            _output_file_name,
            getCodecNumber(_output_file_name),
            _input_fps,
            cv::Size(_output_img_width, _output_img_height),
            true
        );

        if (!_output->isOpened())
        {
            std::cerr << "Error: Unable to create output video file \"" << _output_file_name << "\"" << std::endl;
            return false;
        }
    }

    return true;
}

void EulerMag::run()
{int frame =0;
    while (1)
    {
        timer_.start();
        // Fetch new frame
        cv::Mat tmp_mat;
        _input->read(tmp_mat);
        if (tmp_mat.empty())
            break;

        // upload to gpu
        _input_frame.upload(tmp_mat);
        _input_frame.convertTo(_input_frame, CV_32FC3, 1.0/255.0f);
        // Convert to lab color space
        cv::cuda::cvtColor(_input_frame, _input_frame, cv::COLOR_BGR2Lab);
        
        
        
        // Spatial filter
        makeLaplacianPyramid();
        // Temporal filter
        if (frame== 0)
        {
            
            _lowpass_1 = _laplacian_pyramid;
            _lowpass_2 = _laplacian_pyramid;
            _filtered = _laplacian_pyramid;
        }
        else
        {
            temporalFilter();
            
            
            _delta = _lambda_c / 8.0 / (1.0 + _alpha);
            _lambda = sqrt((float)(_input_img_width * _input_img_width + _input_img_height * _input_img_height)) / 3.0;
            amplify();
        }
        reconstructFromLaplacianPyramid();
        attenuate();
        if (frame != 0)
        {
            cv::cuda::add(_input_frame, _motion, _input_frame);
        }
        frame ++;
        // _input_frame = ;
        cv::cuda::cvtColor(_input_frame, _input_frame, cv::COLOR_Lab2BGR);
        _input_frame.convertTo(_input_frame, CV_8UC3, 255.0, 1.0 / 255.0);
        _input_frame.download(_output_frame);
        cv::resize(_output_frame, _output_frame, cv::Size(_output_img_width,_output_img_height));
        imshow(DISPLAY_WINDOW, _output_frame);
        if (!_output_file_name.empty())
            _output->write(_output_frame);

        double loop_time_ms_ = timer_.getTimeMilliSec();
        std::cout << " | Time taken: " << loop_time_ms_ << " ms" << std::endl;
        char c = cv::waitKey(1);
        if (c == 27)
            break;
    }
    
}

int EulerMag::getCodecNumber(std::string file_name)
{
    std::string file_extn = file_name.substr(file_name.find_last_of('.') + 1);

    // Currently supported video formats are AVI and MPEG-4
    if (file_extn == "avi")
        return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    else if (file_extn == "mp4")
        return cv::VideoWriter::fourcc('D', 'I', 'V', 'X');
    else
        return -1;
}

void EulerMag::makeLaplacianPyramid()
{
    _laplacian_pyramid.clear();
    
    cv::cuda::GpuMat tmp_img;
    
    
    _input_frame.copyTo(tmp_img);
    for (int i = 0; i < _laplacian_pyramid_depth; ++i)
    {
        cv::cuda::GpuMat l1, l0;
        cv::cuda::GpuMat layer;
        
        cv::cuda::pyrDown(tmp_img, l1);
        cv::cuda::pyrUp(l1, l0);
        cv::cuda::subtract(tmp_img, l0, layer);
        
        // TODO: initializing cv::cuda::GpuMat by push back may increase the runtime
        _laplacian_pyramid.push_back(layer);
        l1.copyTo(tmp_img);
    }
        
    
    _laplacian_pyramid.push_back(tmp_img);
}

void EulerMag::temporalFilter()
{
    for (size_t i = 0; i < _laplacian_pyramid_depth; ++i)
    {
        cv::cuda::GpuMat tmp_1, tmp_2;
        cv::cuda::addWeighted(_lowpass_1[i],(1 - _cutoff_frequency_high),_laplacian_pyramid[i], _cutoff_frequency_high, 0, tmp_1);
        cv::cuda::addWeighted(_lowpass_2[i],(1 - _cutoff_frequency_low),_laplacian_pyramid[i], _cutoff_frequency_low, 0, tmp_2);
        _lowpass_1[i] = tmp_1;
        _lowpass_2[i] = tmp_2;
        cv::cuda::subtract(_lowpass_1[i],_lowpass_2[i],_filtered[i]);
    }
    
}

void EulerMag::amplify()
{
    for (int i = _laplacian_pyramid_depth; i >= 0; i--)
    {
        double alpha;
        alpha = _lambda / _delta / 8.0 - 1.0;
        alpha *= _exaggeration;
        if (i == _laplacian_pyramid_depth || i == 0)
        {
            _filtered[i].convertTo(_filtered[i],_filtered[i].type(), 0);
        }else
        {
            _filtered[i].convertTo(_filtered[i],_filtered[i].type(), std::min(alpha,_alpha));
        }
        _lambda /= 2.0;
    }
    
}

void EulerMag::reconstructFromLaplacianPyramid()
{   
    cv::cuda::GpuMat tmp;
    _filtered[_laplacian_pyramid_depth].copyTo(tmp);
    for (int i = _laplacian_pyramid_depth - 1; i >=0; --i)
    {
        cv::cuda::GpuMat l;
        cv::cuda::pyrUp(tmp, l);
        
        cv::cuda::add(l, _filtered[i], tmp);
        
    }
    tmp.copyTo(_motion);
    
}

void EulerMag::attenuate()
{
    cv::cuda::GpuMat channels[3];
    cv::cuda::split(_motion, channels);
    channels[1].convertTo(channels[1], channels[1].type(), _chromatic_attenuation);
    channels[2].convertTo(channels[2], channels[2].type(), _chromatic_attenuation);
    cv::cuda::merge(channels, 3, _motion);
}
