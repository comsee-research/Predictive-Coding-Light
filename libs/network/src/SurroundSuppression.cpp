//
// Created by antony on 03/05/2022
//

#include "SurroundSuppression.hpp"
const int width_ = 346;
const int height_ = 260;

/**
 * Creates an object of the class Surround Suppression. 
 * @param networkPath 
 * @param path 
 * @param network 
 */
SurroundSuppression::SurroundSuppression(const std::string &networkPath, const paths_container &path, NetworkHandle& network) : m_network(network) {
    for(const auto& p: path)
    {
        m_path.emplace_back(p);
    }
    m_networkPath = networkPath;

    for(int layer=0; layer<m_network.getNetworkConfig().getLayerConnectivity().size(); layer++){
        std::vector<receptivefield> to_concatenate;
        int numberOfNeuronsInLayer = m_network.getNetworkConfig().getLayerConnectivity()[layer].patches[0].size()
                * m_network.getNetworkConfig().getLayerConnectivity()[layer].sizes[0]
                * m_network.getNetworkConfig().getLayerConnectivity()[layer].patches[1].size()
                * m_network.getNetworkConfig().getLayerConnectivity()[layer].sizes[1]
                * m_network.getNetworkConfig().getLayerConnectivity()[layer].patches[2].size()
                * m_network.getNetworkConfig().getLayerConnectivity()[layer].sizes[2];
        for(int j=0; j<numberOfNeuronsInLayer; j++){
            receptivefield temp;
            getReceptiveField(layer, j, temp);
            to_concatenate.emplace_back(temp);
        }
        m_neuronsReceptiveField.emplace_back(to_concatenate);
    }
    m_time_gap=1000;
    m_log_threshold=0;
    m_map_threshold=0.4;
    m_n_max=5;
    m_adapt_thresh_coef_shift=0.05;
    m_timestamp_noise=50;
}

/**
 * Trains the network on the dataset.
 * @param typeOfTraining 
 * @param numberOfTimes 
 * @param epochs 
 */
void SurroundSuppression::train(const std::string &typeOfTraining, int numberOfTimes, int epochs){
    if(typeOfTraining==m_network.getSimpleNeuronConfig().STDP_LEARNING){
        auto rng = std::default_random_engine {};
        std::cout << "Training is about to start..." << std::endl;
        events_list events;
        for(int j=0; j<epochs; j++){
            std::shuffle(std::begin(m_path), std::end(m_path), rng);
            std::cout << "It's epoch number : " << j << " !" << std::endl;
            for(int i=0; i<m_path.size();i++) {
                std::cout << "Training of event folder number : " << i+1 << " !" << std::endl;
                while (m_network.loadEvents(events, numberOfTimes)) {
                    m_network.feedEvents(events);
                }
                m_network.save(m_path[i], numberOfTimes);
                events.clear();
                if(i!=m_path.size()-1){
                    m_network.setEventPath(m_path[i+1]);
                }
            }
            m_network.setEventPath(m_path[0]);
        }
    }
    else{

        std::cout << "Please, verify that the type of learning is correct." << std::endl;
    }        
}

/**
 * Identifies the receptive field of a simple cell.
 * @param layer 
 * @param neuron_id 
 * @param to_return 
 */
void SurroundSuppression::getReceptiveField(int layer, int neuron_id, receptivefield &to_return) {
    if(layer>=m_network.getNetworkConfig().getLayerConnectivity().size() || layer < 0){
        std::cout << "Please verify the value you gave to the parameter 'layer'." << std::endl;
    }
    else{
        int totalNumberOfNeurons = m_network.getNetworkConfig().getLayerConnectivity()[layer].patches[0].size()
                * m_network.getNetworkConfig().getLayerConnectivity()[layer].sizes[0]
                * m_network.getNetworkConfig().getLayerConnectivity()[layer].patches[1].size()
                * m_network.getNetworkConfig().getLayerConnectivity()[layer].sizes[1]
                * m_network.getNetworkConfig().getLayerConnectivity()[layer].patches[2].size()
                * m_network.getNetworkConfig().getLayerConnectivity()[layer].sizes[2];
        if(neuron_id<0 || neuron_id>=totalNumberOfNeurons){
            std::cout << "Please verify the value you gave to the parameter 'neuron_id'." << std::endl;
        }
        else{
            int id = neuron_id;
            int actual_x = 0;
            while(id>=0){
                id=id - m_network.getNetworkConfig().getLayerConnectivity()[layer].sizes[1] * m_network.getNetworkConfig().getLayerConnectivity()[layer].sizes[2];
                actual_x++;
            }
            actual_x = actual_x -1;
            id+=m_network.getNetworkConfig().getLayerConnectivity()[layer].sizes[1] * m_network.getNetworkConfig().getLayerConnectivity()[layer].sizes[2];
            int actual_y = 0;
            while(id>=0){
                id = id - m_network.getNetworkConfig().getLayerConnectivity()[layer].sizes[2];
                actual_y++;            
            }
            actual_y = actual_y -1;
            int x_minus=0; 
            int y_minus=0;
            int x_plus=0;
            int y_plus=0;
            int neursizes_x = m_network.getNetworkConfig().getLayerConnectivity()[layer].neuronSizes[0][0];
            int overlap_x = m_network.getNetworkConfig().getLayerConnectivity()[layer].neuronOverlap[0];
            if(overlap_x!=0)
            {
                x_minus= (actual_x)*(neursizes_x-overlap_x);
            }
            else
            {
                x_minus=(actual_x)*neursizes_x;
            }
            
            x_plus = x_minus + m_network.getNetworkConfig().getLayerConnectivity()[layer].neuronSizes[0][0];

            int neursizes_y = m_network.getNetworkConfig().getLayerConnectivity()[layer].neuronSizes[0][1];
            int overlap_y = m_network.getNetworkConfig().getLayerConnectivity()[layer].neuronOverlap[1];
            if(overlap_y!=0)
            {
                y_minus= (actual_y) *(neursizes_y-overlap_y);
            }
            else
            {
                y_minus=(actual_y)*neursizes_y;
            }
            
            y_plus = y_minus + m_network.getNetworkConfig().getLayerConnectivity()[layer].neuronSizes[0][1];
            to_return.x_minus = x_minus;
            to_return.x_plus = x_plus;
            to_return.y_minus = y_minus;
            to_return.y_plus = y_plus;
        }
    }
}

/**
 * Generates sequences of an increasing number of bars of the same length.
 * @param ev 
 * @param yPositionStart 
 * @param lengthsOfBars 
 * @param speed 
 * @param num_disparities 
 * @param nbPass 
 * @param pathToImgs 
 * @param thickness 
 * @param angle 
 */
void SurroundSuppression::multiBars1Length(events_sequences &ev, const std::vector<int> &yPositionStart, const std::vector<int> &lengthsOfBars, int speed, int num_disparities, int nbPass, const std::string &pathToImgs, int thickness, int angle) {
    std::vector<int> disparities;
    for(int i=0; i<num_disparities; i++) {
        fs::create_directory(pathToImgs + std::to_string(i));
        int frame = 0;
        int framerate = 1000;
        int shift = 0;
        int once = 0;
        bool ok = true;
        if(speed >0 ) {
            while(ok) {
                if(once==0 || (once<=i && (shift + disparities.at(once-1))>=10)) {
                    if(once==0) {
                        disparities.push_back(0);
                    }
                    else {
                        disparities.push_back(disparities.at(once-1)-10);
                    }
                    once+=1;
                }

                image_matrix img(height_, width_, CV_8UC3,cv::Scalar(128, 128, 128));
                shift = int(frame * float(float(speed)/framerate));
                for(int j=0; j<disparities.size(); j++) {
                    cv::Point p1(disparities.at(j)+shift, yPositionStart.at(0));
                    cv::Point p2(disparities.at(j)+shift, yPositionStart.at(0)+lengthsOfBars.at(0));
                    cv::line(img, p1, p2, cv::Scalar(255, 255, 255),thickness, cv::LINE_8);
                }
                if(once-1==i) {
                    if(shift + disparities.at(i) > width_ + 5 ) {
                        ok=false;
                    }
                }
                cv::Point center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
                image_matrix rot = getRotationMatrix2D(center, angle, 1.0); 
                image_matrix newimg;
                cv::warpAffine(img, newimg, rot, cv::Size(img.cols, img.rows), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT,cv::Scalar(0,0,0));
                std::ostringstream path;
                path << pathToImgs << i << "/img_" << frame << ".png";
                cv::imwrite(path.str(), newimg);
                frame+=1;
            }
            disparities.clear();
        }
        std::string new_path = pathToImgs + std::to_string(i) + "/";
        std::cout << new_path << std::endl;
        events_list temp_ev;
        createEvents(new_path, temp_ev, nbPass);
        ev.emplace_back(temp_ev);
    }
}

/**
 * Generates sequences of one bar with an increasing length.
 * @param ev 
 * @param lengthsOfBars 
 * @param yPositionStart 
 * @param angle 
 * @param speed 
 * @param nbPass 
 * @param pathToImgs 
 * @param thickness 
 */
void SurroundSuppression::oneBarMultiLengths(events_sequences &ev, const std::vector<int> &lengthsOfBars, const std::vector<int> &yPositionStart, int angle, int speed, int nbPass, const std::string &pathToImgs, int thickness, std::string mode) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution_color(0,2);
    int color_counter_white = 0;
    int color_counter_black = 0;
    int sign = speed/abs(speed);
    int frame_ref = 0;
    for(int i=0; i<lengthsOfBars.size(); i++)
    {
        fs::create_directory(pathToImgs + std::to_string(i));
        int frame = 0;
        int framerate = 1000;
        int shift = 0;
        cv::Scalar color(255, 255, 255);
        cv::Scalar black(0, 0, 0);
        cv::Scalar white(255, 255, 255);
        if(sign > 0 )
        {
            if (mode == "eval") { 
                color_counter_white = 0;
                color_counter_black = 0;
                for(int trial = 0; trial < 3; trial++) {
                    shift = 0;
                    frame_ref = 0;
                    if(trial == 0) {
                        speed = sign * 100;
                    }
                    else if(trial == 1) {
                        speed = sign * 299;
                    }
                    else {
                        speed = sign * 499;
                    }
                
                    if(distribution_color(generator)==0) {
                        if(color_counter_black!=2) {
                            color = black;
                            color_counter_black+=1;
                        }
                        else {
                            color = white;
                        }
                    }
                    else {
                        if(color_counter_white !=2) {
                            color = white;
                            color_counter_white+=1;
                        }
                        else {
                            color = black;
                        }
                    }
                    int x = 0;
                    while(shift < width_ + 5 ) {
                        image_matrix img(height_, width_, CV_8UC3,cv::Scalar(128, 128, 128));
                        shift = int(frame_ref * float(float(speed)/framerate));
                        cv::Point p1(x+shift, yPositionStart.at(i));
                        cv::Point p2(x+shift, yPositionStart.at(i)+lengthsOfBars.at(i));
                        cv::line(img, p1, p2, color, thickness, cv::LINE_8);
                        cv::Point center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
                        image_matrix rot = getRotationMatrix2D(center, angle, 1.0);
                        image_matrix newimg;
                        cv::warpAffine(img, newimg, rot, cv::Size(img.cols, img.rows), cv::INTER_LINEAR, 
                        cv::BORDER_TRANSPARENT,cv::Scalar(0,0,0));
                        std::ostringstream path;
                        path << pathToImgs << i << "/img_" << frame << ".png";
                        cv::imwrite(path.str(), newimg);
                        frame+=1;
                        frame_ref+=1;
                    }
                }
            }
            else {
                int x = 0;
                while(shift < width_ + 5 ) {
                    image_matrix img(height_, width_, CV_8UC3,cv::Scalar(128, 128, 128));
                    shift = int(frame * float(float(speed)/framerate));
                    cv::Point p1(x+shift, yPositionStart.at(i));
                    cv::Point p2(x+shift, yPositionStart.at(i)+lengthsOfBars.at(i));
                    cv::line(img, p1, p2, color, thickness, cv::LINE_8);
                    cv::Point center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
                    image_matrix rot = getRotationMatrix2D(center, angle, 1.0);
                    image_matrix newimg;
                    cv::warpAffine(img, newimg, rot, cv::Size(img.cols, img.rows), cv::INTER_LINEAR, 
                    cv::BORDER_TRANSPARENT,cv::Scalar(0,0,0));
                    std::ostringstream path;
                    path << pathToImgs << i << "/img_" << frame << ".png";
                    cv::imwrite(path.str(), newimg);
                    frame+=1;
                }
            }
        }
        else if (sign < 0)
        {
            if(mode=="eval") {
                for(int trial = 0; trial < 3; trial++) {
                    shift = 0;
                    frame_ref = 0;
                    if(trial == 0) {
                        speed = sign * 100;
                    }
                    else if(trial == 1) {
                        speed = sign * 299;
                    }
                    else {
                        speed = sign * 499;
                    }

                    if(distribution_color(generator)==0) {
                        if(color_counter_black!=2) {
                            color = black;
                            color_counter_black+=1;
                        }
                        else {
                            color = white;
                        }
                    }
                    else {
                        if(color_counter_white !=2) {
                            color = white;
                            color_counter_white+=1;
                        }
                        else {
                            color = black;
                        }
                    }
                    int x = width_-1;
                    while(shift > -(width_+5))  {
                        image_matrix img(height_, width_, CV_8UC3,cv::Scalar(128, 128, 128));
                        shift = int(frame_ref * float(float(speed)/framerate));
                        cv::Point p1(x+shift, yPositionStart.at(i));
                        cv::Point p2(x+shift, yPositionStart.at(i)+lengthsOfBars.at(i));
                        cv::line(img, p1, p2, color, thickness, cv::LINE_8);
                        cv::Point center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
                        image_matrix rot = getRotationMatrix2D(center, angle, 1.0);      
                        image_matrix newimg;
                        cv::warpAffine(img, newimg, rot, cv::Size(img.cols, img.rows), cv::INTER_LINEAR, 
                        cv::BORDER_TRANSPARENT,cv::Scalar(0,0,0));
                        std::ostringstream path;
                        path << pathToImgs << i << "/img_" << frame << ".png";
                        cv::imwrite(path.str(), newimg);
                        frame+=1;
                        frame_ref+=1;
                    }
                }
            }
            else {
                int x = width_-1;
                while(shift > -(width_+5))  {
                    image_matrix img(height_, width_, CV_8UC3,cv::Scalar(128, 128, 128));
                    shift = int(frame * float(float(speed)/framerate));
                    cv::Point p1(x+shift, yPositionStart.at(i));
                    cv::Point p2(x+shift, yPositionStart.at(i)+lengthsOfBars.at(i));
                    cv::line(img, p1, p2, color, thickness, cv::LINE_8);
                    cv::Point center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
                    image_matrix rot = getRotationMatrix2D(center, angle, 1.0);
                    image_matrix newimg;
                    cv::warpAffine(img, newimg, rot, cv::Size(img.cols, img.rows), cv::INTER_LINEAR, 
                    cv::BORDER_TRANSPARENT,cv::Scalar(0,0,0));
                    std::ostringstream path;
                    path << pathToImgs << i << "/img_" << frame << ".png";
                    cv::imwrite(path.str(), newimg);
                    frame+=1;
                 }
            }
        }
        std::string new_path = pathToImgs + std::to_string(i) + "/";
        std::cout << new_path << std::endl;
        events_list temp_ev;
        createEvents(new_path, temp_ev, nbPass);
        ev.emplace_back(temp_ev);
    }
}

/**
 * Generate sequences of bars.
 * @param ev 
 * @param numberOfTypesOfBars 
 * @param lengthsOfBars 
 * @param yPositionStart 
 * @param angle 
 * @param speed 
 * @param nbPass 
 */
void SurroundSuppression::generateBars(events_sequences &ev,  int numberOfTypesOfBars, const std::vector<int> &lengthsOfBars, const std::vector<int> &yPositionStart, int angle, int speed, int nbPass, int thickness, std::string mode) {
        if(std::all_of(yPositionStart.cbegin(), yPositionStart.cend(), [](int i){ return i>=0; }) && std::all_of(yPositionStart.cbegin(), yPositionStart.cend(), [this](int i){ return i<height_; })){
            int num_disparities = numberOfTypesOfBars;
            std::string choice;
            static bool entered = false;
            if(!entered) {
                std::cout << "Do you want to enter the mode to generate multiple bars of the same length in the same frame?" << std::endl;
                std::cin >> choice;
                entered = true;
            }
            std::string path = m_network.getNetworkConfig().getNetworkPath() + "generateSequences/";
            fs::file_status s = fs::file_status{};
            if(fs::status_known(s) ? fs::exists(s) : fs::exists(path)) {
                fs::remove_all(path);
            }
            fs::create_directory(path);
            if(choice=="yes" || choice=="y") {
                int fac;
                fac = 34;
                num_disparities = fac;
                multiBars1Length(ev, yPositionStart, lengthsOfBars, speed, num_disparities, nbPass, path, thickness, angle);
            }
            else {
                oneBarMultiLengths(ev, lengthsOfBars, yPositionStart, angle, speed, nbPass, path, thickness, mode);
            }
        }
}

/**
 * Finds the position where to start the bar to keep it centered on the tracked neuron regardless of the length.
 * @param middle 
 * @param value 
 * @param positionStart 
 * @param listOfLengths 
 * @param numberOfBars 
 */
void SurroundSuppression::findCenteredBarsPosition(int middle, int value, std::vector<int> &positionStart, std::vector<int> &listOfLengths, int numberOfBars) {
    int start;
    int offset;
    int n_ = numberOfBars;
    int border;
    bool checkLimit = true;
    int stop;
    int to_add = 3;
    int iterator = 1;
    for(int i=0; i<n_; i++) { 
        offset = value * (i+1); 
        if(offset > 45) {
            if(checkLimit) {
                stop = value * (i-1);
                checkLimit = false;
            }
            offset = stop + to_add * iterator;
            iterator += 1;
        }
        border = middle-int(offset/2)+ m_network.getNetworkConfig().getLayerConnectivity()[0].patches[1][0];
        if(i>=0 && border>=0) {
            start = border;  
        }      
        else {
            start = 0;
        }
        if(start>=0 && start+offset<height_) {
            listOfLengths.emplace_back(offset);
            positionStart.emplace_back(start);
        }
    }
}

/**
 * Finds the center of the bar according to its orientation.
 * @param middle 
 * @param angle 
 */
void SurroundSuppression::findMiddlePointOrdinate(int &middle, int angle) {
    int real_track_coord = m_network.getSimpleNeuronConfig().POTENTIAL_TRACK[0]*m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[1] *
            m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[2] + m_network.getSimpleNeuronConfig().POTENTIAL_TRACK[1]*m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[2];
    int to_track_x = ((m_neuronsReceptiveField[0][real_track_coord]).x_plus+(m_neuronsReceptiveField[0][real_track_coord]).x_minus)/2 +
            m_network.getNetworkConfig().getLayerConnectivity()[0].patches[0][0];
    int to_track_y = ((m_neuronsReceptiveField[0][real_track_coord]).y_plus+(m_neuronsReceptiveField[0][real_track_coord]).y_minus)/2 +
            m_network.getNetworkConfig().getLayerConnectivity()[0].patches[1][0];
    double sin_rad = sin(angle*M_PI/180);
    double cos_rad = cos(angle*M_PI/180);
    double center_x = width_/2;
    double center_y = height_/2;
    double num = to_track_y + (sin_rad/cos_rad)*to_track_x - (sin_rad/cos_rad)*(1-cos_rad)*center_x + (sin_rad/cos_rad)*(sin_rad*center_y)-(sin_rad*center_x)-(1-cos_rad)*center_y;
    double den = sin_rad*sin_rad/cos_rad + cos_rad;
    middle = int(floor(num/den)) - m_network.getNetworkConfig().getLayerConnectivity()[0].patches[1][0];
}

/**
 * Verifies if the parameters to launch the evaluation of Surround Suppression are safe.
 * @return true 
 * @return false 
 */
bool SurroundSuppression::verifyIfSafe() {
    int numberOfMax_nx = m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[0];
    int numberOfMax_ny = m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[1];
    if(numberOfMax_nx <= m_network.getSimpleNeuronConfig().POTENTIAL_TRACK[0] || numberOfMax_ny <= m_network.getSimpleNeuronConfig().POTENTIAL_TRACK[1]) {
        std::cout << "The coordinates in \"POTENTIAL_TRACK\" are not right. Program aborted." << std::endl;
        return false;
    }
    int n = m_network.getSimpleNeuronConfig().POTENTIAL_TRACK[0] *m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[1]*m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[2] + m_network.getSimpleNeuronConfig().POTENTIAL_TRACK[1] *m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[2];
    int x_minus=(m_neuronsReceptiveField[0][n]).x_minus;
    int x_plus=(m_neuronsReceptiveField[0][n]).x_plus;
    int y_minus=(m_neuronsReceptiveField[0][n]).y_minus;
    int y_plus=(m_neuronsReceptiveField[0][n]).y_plus;
    std::cout << "The neurons with indexes " << n << "-" << n + m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[2] -1 << " (last one included) should have their stats data being saved." << std::endl;
    std::cout << "The receptive field of this neuron is : " << x_minus << " - " << x_plus << " X " << y_minus << " - " << y_plus << std::endl << std::endl;
    if(m_network.getSimpleNeuronConfig().STDP_LEARNING!="excitatory" && m_network.getSimpleNeuronConfig().STDP_LEARNING!="inhibitory" && m_network.getSimpleNeuronConfig().STDP_LEARNING!="all") {
        return true;
    }
    else {
        std::cout << "Change the parameter \"STDP_LEARNING\" for the program to work." << std::endl;
        return false;
    }
}

bool SurroundSuppression::verifyIfSafe(const std::string &path, int n_thickness, int n_angles, int n_directions, int n_bars, int n_simulations, bool separate_speed, int n_speeds) {
    if(verifyIfSafe()) {
        int cter_thickness = 0;
        int cter_angles = 0;
        int cter_directions = 0;
        int cter_bars = 0;
        int cter_simulations = 0;
        for (const auto & thickness : fs::directory_iterator{path}) {
            cter_thickness +=1;
            std::string thickness_path = thickness.path().string();
            for(const auto & angles : fs::directory_iterator(thickness_path)) {
                cter_angles+=1;
                std::string angles_path = angles.path().string();
                for(const auto & directions : fs::directory_iterator(angles_path)) {
                    cter_directions+=1;
                    std::string directions_path = directions.path().string();
                    for(const auto &bars : fs::directory_iterator(directions_path)) {
                        cter_bars += 1;
                        std::string bars_path = bars.path().string();
                        if(!separate_speed) {
                            for(const auto & simulations : fs::directory_iterator(bars_path)) {
                                cter_simulations +=1;
                            }
                            if(cter_simulations < n_simulations) {
                                
                                std::cout << "Number of simulation selected doesn't correlate with number of files in the testing dataset." << std::endl;
                                return false;
                            }
                            else {
                                cter_simulations = 0;
                            }
                        }
                        else {
                            int cter_speeds = 0;
                            for(const auto & speeds : fs::directory_iterator(bars_path)) {
                                cter_speeds +=1;
                                std::string speeds_path = speeds.path().string();
                                for(const auto & simulations : fs::directory_iterator(speeds_path)) {
                                    cter_simulations +=1;
                                }
                                if(cter_simulations < n_simulations) {
                                    std::cout << "Number of simulation selected doesn't correlate with number of files in the testing dataset." << std::endl;
                                    return false;
                                }
                                else {
                                    cter_simulations = 0;
                                }
                            }
                            if(cter_speeds < n_speeds) {
                                std::cout << "Number of speeds selected doesn't correlate with number of files in the testing dataset." << std::endl;
                                return false;
                            }
                            else {
                                cter_speeds = 0;
                            }
                        }
                    }
                    if(cter_bars < n_bars) {
                        std::cout << "Number of bars selected doesn't correlate with number of bars folder in the testing dataset." << std::endl;
                        return false;
                    }
                    else {
                        cter_bars = 0;
                    }
                }
                if(cter_directions < n_directions) {
                    std::cout << "Number of directions selected doesn't correlate with number of directions folder in the testing dataset." << std::endl;
                    return false;
                }
                else {
                    cter_directions = 0;
                }
            }
            if(cter_angles < n_angles) {
                std::cout << "Number of angles selected doesn't correlate with number of angles folder in the testing dataset." << std::endl;
                return false;
            }
            else {
                cter_angles = 0;
            }
        }
        if(cter_thickness < n_thickness) {
            std::cout << "Number of thicknesses selected doesn't correlated with number of thicknesses folder in the testing dataset." << std::endl;
            return false;
        }
        else {
            std::cout << "All parameters are correct. The evaluation on the testing dataset can start." << std::endl;
            return true;
        }
    }
    else {
        return false;
    }
}

bool SurroundSuppression::verifyIfSafeMultipleOrientations(const std::string &path, int n_angles, int n_simulations) {
    if(verifyIfSafe()) {
        int cter_main_angles = 0;
        int cter_surrounding_angles = 0;
        int cter_simulations = 0;
        for (const auto & main_angle : fs::directory_iterator{path}) {
            cter_main_angles +=1;
            std::string main_angle_path = main_angle.path().string();
            for(const auto & surrounding_angle : fs::directory_iterator(main_angle_path)) {
                cter_surrounding_angles +=1;
                std::string surrounding_angle_path = surrounding_angle.path().string();
                for(const auto & simulation : fs::directory_iterator(surrounding_angle_path)) {
                    cter_simulations+=1;
                    std::string simulation_path = simulation.path().string();
                }
                if(cter_simulations < n_simulations) { 
                    std::cout << "Number of simulation selected doesn't correlate with number of files in the testing dataset." << std::endl;
                    return false;
                }
                else {
                    cter_simulations = 0;
                }
            }
            if(cter_surrounding_angles < 2) {
                std::cout << "Number of surrounding angles selected doesn't correlate with number of files in the testing dataset." << std::endl;
                return false;
            }
            else {
                cter_surrounding_angles = 0;
            }
        }
        if(cter_main_angles < n_angles) {
            std::cout << cter_main_angles << std::endl;
            std::cout << "Number of main angles selected doesn't correlate with number of files in the testing dataset." << std::endl;
            return false;
        }
        else {
            std::cout << "All parameters are correct. The evaluation on the testing dataset can start." << std::endl;
            return true;
        }
    }
    else {
        return false;
    }
}

/**
 * Specifies the simulation choices for the evaluation of Surround Suppression.
 * @param choice 
 * @param simulation 
 * @param n_ 
 */
void SurroundSuppression::simulationChoices(std::string &choice, int &simulation, int &n_) {
    std::string str_n;
    int max_n_ = 150;
    std::cout << "Do you want to evaluate the suppression?" << std::endl;
    std::cin >> choice;
    if(choice!="yes" && choice !="y" && choice !="eval") {
        std::cout << "The program will find the preferred orientation of each cells, then." << std::endl;
        str_n = "1";
    }
    else {
        std::cout << "Type the number of bars that will be generated." << std::endl;
        std::cin >> str_n;
    }
    
    while(std::stoi(str_n)<=0 || std::stoi(str_n)>=max_n_) {
        std::cout << "Please, type a number of bar that is comprised between 1 and " << max_n_ << "." << std::endl;
        std::cin >> str_n;
    }
    n_ = std::stoi(str_n);
    
    std::cout << "How many simulations do you want to run ?" << std::endl;
    std::cin >> simulation;
}

/**
 * Evaluates the responses of the network to different gratings stimuli.
 */
void SurroundSuppression::evaluateResponsesOnStimuli() {
    bool ok = verifyIfSafe();
    if(ok) {
        int nbPass = 1;
        int speedValue = 499;
        int n_;
        int middle;
        std::string choice;
        int simulation;
        simulationChoices(choice, simulation, n_);
        int value;
        int thickness = 3;
        if(choice=="yes" || choice=="y") {
            value = 1;
            int angle =0; // {0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338};
            findMiddlePointOrdinate(middle, angle);
            std::vector<int> listOfLengths;
            std::vector<int>positionStart;
            findCenteredBarsPosition(middle, value, positionStart, listOfLengths, n_);
            for(int sim=0; sim < simulation; sim++) {
                std::cout << "Running simulation " << sim+1 << "/" << simulation << "." << std::endl;
                events_sequences ev;
                generateBars(ev, n_, listOfLengths, positionStart, angle, speedValue, nbPass, thickness);
                for(int i=0; i< ev.size(); i++)
                {
                    if(listOfLengths.size() != 1 ) {
                        std::cout << "Evaluating bar number " << i << " that has a length of " << listOfLengths.at(i) << " and goes from ordinate " << positionStart.at(i) << " to " << positionStart.at(i) + listOfLengths.at(i) << std::endl;
                    }
                    else {
                        std::cout << "Evaluating bar number " << i << " that has a length of " << listOfLengths.at(0) << " and goes from ordinate " << positionStart.at(0) << " to " << positionStart.at(0) + listOfLengths.at(0) << std::endl;
                    }
                    m_network.feedEvents(ev.at(i)); 
                    m_network.saveStatistics(sim, i, std::to_string(3) + "/");
                }
            }
        }
        else if (choice=="eval") {

            //Thicknesses folder ( [1, ... , n_thickness] ) must be created manually before launching the code, for now.

            value = 1;
            int num_angles = 8;
            int angles[num_angles] = {0, 23, 45, 68, 90, 113, 135, 158};
            int angle;
            for(thickness = 0; thickness < 3; thickness++) {
                std::cout << "thickness = " << thickness+1 << std::endl;
                for(int val=0; val<num_angles*2; val++) {
                    if(val%2 == 0) {
                        angle = angles[val/2];
                        speedValue = abs(speedValue);
                        std::cout << "Angle = " << angle << std::endl;
                    }
                    else {
                        angle = angles[int(val/2)];
                        speedValue = -abs(speedValue);
                        std::cout << "Opposite direction !" << std::endl;
                    }
                    findMiddlePointOrdinate(middle, angle);
                    std::vector<int> listOfLengths;
                    std::vector<int> positionStart;
                    findCenteredBarsPosition(middle, value, positionStart, listOfLengths, n_);
                    for(int sim=0; sim < simulation; sim++) {
                        std::cout << "Running simulation " << sim+1 << "/" << simulation << "." << std::endl;
                        events_sequences ev;
                        generateBars(ev,n_,listOfLengths,positionStart,angle, speedValue, nbPass, thickness+1, choice);
                        for(int i=0; i< ev.size(); i++) {
                            std::cout << "Evaluating bar number " << i << " that has a length of " << listOfLengths.at(i) << " and goes from ordinate " << positionStart.at(i) << " to " << positionStart.at(i) + listOfLengths.at(i) << std::endl;
                            m_network.feedEvents(ev.at(i)); 
                            if(speedValue > 0 ) {
                                m_network.saveStatistics(sim, i, std::to_string(thickness+1) + "/" + std::to_string(angle) + "/");
                            }
                            else {
                                m_network.saveStatistics(sim, i, std::to_string(thickness+1) + "/-" + std::to_string(angle) + "/");
                            }
                        }
                    }
                }
            }
        }
        else {
            value = 10;
            int index_neur = m_network.getSimpleNeuronConfig().POTENTIAL_TRACK[0] * m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[1] * m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[2] + m_network.getSimpleNeuronConfig().POTENTIAL_TRACK[1] * m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[2];
            int depth = m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[2];
            int complex_number = m_network.getNetworkConfig().getLayerConnectivity()[1].patches[0].size() * m_network.getNetworkConfig().getLayerConnectivity()[1].sizes[0] * m_network.getNetworkConfig().getLayerConnectivity()[1].patches[1].size() * m_network.getNetworkConfig().getLayerConnectivity()[1].sizes[1] * m_network.getNetworkConfig().getLayerConnectivity()[1].patches[2].size() * m_network.getNetworkConfig().getLayerConnectivity()[1].sizes[2];
            int num_angles = 8;
            speedValue = 100;
            std::vector<std::vector<std::vector<float>>> vec(depth, std::vector<std::vector<float>> (2,std::vector<float>(num_angles*2,0)));
            std::vector<std::vector<std::vector<float>>> vec_complex(complex_number, std::vector<std::vector<float>> (1,std::vector<float>(num_angles*2,0)));
            std::vector<std::vector<std::vector<float>>> vec_mean_spikes(depth, std::vector<std::vector<float>> (1,std::vector<float>(num_angles*2,0)));
            std::vector<std::vector<std::vector<float>>> vec_complex_mean_spikes(complex_number, std::vector<std::vector<float>> (1,std::vector<float>(num_angles*2,0)));

            int angles[num_angles] = {0, 23, 45, 68, 90, 113, 135, 158}; 
            int double_angles[num_angles*2] = {0, 180, 23, -23, 45, -45, 68, -68, 90, -90, 113, -113, 135, -135, 158, -158};
            int angle;
            for(int sim=0; sim < simulation; sim++) {
                for(thickness = 0; thickness < 3; thickness++) {
                    for(int val=0; val<num_angles*2; val++) {
                        if(val%2 == 0) {
                            angle = angles[val/2];
                            speedValue = abs(speedValue);
                            std::cout << "Angle = " << angle << std::endl;
                        }
                        else {
                            angle = angles[int(val/2)];
                            speedValue = -abs(speedValue);
                            std::cout << "Opposite direction !" << std::endl;
                        }
                        findMiddlePointOrdinate(middle, angle);
                        std::vector<int> listOfLengths;
                        std::vector<int> positionStart;
                        findCenteredBarsPosition(middle, value, positionStart, listOfLengths, n_);
                        std::cout << "Running simulation " << sim+1 << "/" << simulation << "." << std::endl;
                        events_sequences ev;
                        generateBars(ev,n_,listOfLengths,positionStart,angle, speedValue, nbPass, thickness+1, "eval");
                        for(int i=0; i< ev.size(); i++) {
                            std::cout << "Evaluating bar number " << i << " that has a length of " << listOfLengths.at(i) << " and goes from ordinate " << positionStart.at(i) << " to " << positionStart.at(i) + listOfLengths.at(i) << std::endl;
                            m_network.feedEvents(ev.at(i)); 
                            m_network.saveStatistics(sim, -1, "");
                        }
                        for(int neur = 0; neur < depth; neur++) {
                            vec.at(neur).at(0).at(val)+=m_network.getNeuron(index_neur+neur,0).get().getTrackingSpikeTrain().size();
                            int den = 1;
                            if((m_network.getNeuron(index_neur+neur,0).get().getTrackingSpikeTrain().size()) != 0) {
                                den = (m_network.getNeuron(index_neur+neur,0).get().getTrackingSpikeTrain().size());
                            }
                            vec_mean_spikes.at(neur).at(0).at(val) += std::accumulate(m_network.getNeuron(index_neur+neur,0).get().getTrackingSpikeTrain().begin(), m_network.getNeuron(index_neur+neur,0).get().getTrackingSpikeTrain().end(), 0.0) / den;
                        }
                        for(int neur = 0; neur < complex_number; neur++) {
                            vec_complex.at(neur).at(0).at(val)+=m_network.getNeuron(neur,1).get().getTrackingSpikeTrain().size();
                            int den = 1;
                            if((m_network.getNeuron(neur,1).get().getTrackingSpikeTrain().size()) != 0) {
                                den = (m_network.getNeuron(neur,1).get().getTrackingSpikeTrain().size());
                            }
                            vec_complex_mean_spikes.at(neur).at(0).at(val)+=std::accumulate(m_network.getNeuron(neur,1).get().getTrackingSpikeTrain().begin(), m_network.getNeuron(neur,1).get().getTrackingSpikeTrain().end(), 0.0) /den;
                        }
                        m_network.saveStatistics(0, -1, "", true);
                    }
                }
            }
            for(int neur = 0; neur < depth; neur++) {
                        float spike_value = *std::max_element(vec.at(neur).at(0).begin(),vec.at(neur).at(0).end());
                        if(spike_value==0) {
                            m_network.assignOrientation(neur, -1, 0);
                        }
                        else {
                            auto it = vec.at(neur).at(0).begin();
                            std::vector<int> angles_index; 
                            while ((it = std::find_if(it, vec.at(neur).at(0).end(), [&spike_value](float x){return x == spike_value; })) != vec.at(neur).at(0).end()) {   
                                int index = std::distance(vec.at(neur).at(0).begin(), it);
                                angles_index.push_back(index);
                                it++;
                            }
                            if(angles_index.size()> 1) {
                                float max_elem = -1;
                                int max_elem_index =0;
                                for(int i; i<angles_index.size(); i++) {
                                    if(vec_mean_spikes.at(neur).at(0).at(angles_index.at(i)) > max_elem) {
                                        max_elem = vec_mean_spikes.at(neur).at(0).at(angles_index.at(i));
                                        max_elem_index = angles_index.at(i);
                                    }
                                }
                                m_network.assignOrientation(neur, double_angles[max_elem_index], 0);
                            }
                            else if(angles_index.size() ==1) {
                                m_network.assignOrientation(neur, double_angles[angles_index.at(0)], 0);
                            }
                        }
                    }
                    for(int neur = 0; neur < complex_number; neur++) {
                        float spike_value = *std::max_element(vec_complex.at(neur).at(0).begin(),vec_complex.at(neur).at(0).end());
                        if(spike_value==0) {
                            m_network.assignComplexOrientation(neur, -1, 0);
                        }
                        else {
                            auto it = vec_complex.at(neur).at(0).begin();
                            std::vector<int> angles_index; 
                            while ((it = std::find_if(it, vec_complex.at(neur).at(0).end(), [&spike_value](float x){return x == spike_value; })) != vec_complex.at(neur).at(0).end()) {   
                                int index = std::distance(vec_complex.at(neur).at(0).begin(), it);
                                angles_index.push_back(index);
                                it++;
                            }
                            if(angles_index.size()>1) {
                                float max_elem = -1;
                                int max_elem_index;
                                for(int i; i<angles_index.size(); i++) {
                                    if(vec_complex_mean_spikes.at(neur).at(0).at(angles_index.at(i)) > max_elem) {
                                        max_elem = vec_complex_mean_spikes.at(neur).at(0).at(angles_index.at(i));
                                        max_elem_index = angles_index.at(i);
                                    }
                                }
                                m_network.assignComplexOrientation(neur, double_angles[max_elem_index], 0);
                            }
                            else if(angles_index.size()==1){
                                m_network.assignComplexOrientation(neur, double_angles[angles_index.at(0)], 0);
                            }
                        }
                    }
                    m_network.saveStatistics(simulation, -1, "", true);
        } 
    }
}

/**
 * Evaluates neurons with already existing testing dataset.
 * @param eventsPath 
 */
void SurroundSuppression::evaluateResponsesOnStimuli(const std::string &eventsPath) {
    //Thicknesses folder ( [1, ... , n_thickness] ) must be created manually before launching the code, for now.
    //Same for speeds folder ([0, ..., n_speed-1]), they must be created manually.

    int num_thicknesses = 3; 
    int num_angles = 8; 
    int num_directions = 2;
    int num_bars = 55;
    int num_simulations = 2; 
    int num_speeds = 2;
    int angles[num_angles] = {0, 23, 45, 68, 90, 113, 135, 158};
    int direction;
    int angle;
    bool sep_speed = false;
    int y_pos = 4;
    int lim_sigmoid = 50000;
    bool ok = verifyIfSafe(eventsPath, num_thicknesses, num_angles, num_directions, num_bars, num_simulations, sep_speed, num_speeds);
    if(ok) {
        for(int thickness = 3; thickness < num_thicknesses+1; thickness++) {
            std::cout << "Thickness = " << thickness << std::endl;
            for(int val = 0; val < num_angles *2; val++) {
                
                if(val%2 == 0) {
                    angle = int(angles[val/2]);
                    direction = 0;
                    std::cout << "Angle = " << angle << std::endl;
                }
                else {
                    angle = angles[int(val/2)];
                    direction = 1;
                    std::cout << "Opposite direction !" << std::endl;
                }
                if(angle !=158) {
                     continue;
                }
                for(int sim=0; sim < num_simulations; sim++) {
                    std::cout << "Running simulation " << sim+1 << "/" << num_simulations << "." << std::endl;
                    for(int i=0; i< num_bars; i++) {
                        std::cout << "Evaluating bar number " << i << " ! " << std::endl;
                        if(!sep_speed) {
                            m_network.setEventPath(eventsPath + std::to_string(thickness) + "/" + std::to_string(angle) + "/" + std::to_string(direction) + "/" + std::to_string(i) + "/" + std::to_string(sim) + ".npz");
                            std::cout << eventsPath + std::to_string(thickness) + "/" + std::to_string(angle) + "/" + std::to_string(direction) + "/" + std::to_string(i) + "/" + std::to_string(sim) + ".npz" << std::endl;
                            Events events;
                            size_t numberOfTimes = 1;
                            int length = i+1;
                            if(i+1>45) {
                                length = 45 + 3*(i+1-45);
                            }
                            while (m_network.loadEvents(events, numberOfTimes)) {
                                m_network.feedEvents(events);
                            }
                            events.clear();
                            if(direction == 0 ) {
                                m_network.saveStatistics(sim, i, std::to_string(thickness) + "/" + std::to_string(angle) + "/");
                            }
                            else {
                                m_network.saveStatistics(sim, i, std::to_string(thickness) + "/-" + std::to_string(angle) + "/");
                            }
                        }
                        else {
                            for (int j=0; j< num_speeds; j++) {
                                if(num_speeds==3) {
                                    if(j==0) {
                                        std::cout << "Evaluating low speed." << std::endl;
                                    }
                                    else if(j==1) {
                                        std::cout << "Evaluating medium speed." << std::endl;
                                    }
                                    else if(j==2) {
                                        std::cout << "Evaluating high speed." << std::endl;
                                    }
                                }
                                else {
                                    std::cout << "Evaluating speed number " << j << std::endl;
                                }
                                m_network.setEventPath(eventsPath + std::to_string(thickness) + "/" + std::to_string(angle) + "/" + std::to_string(direction) + "/" + std::to_string(i) + "/" + std::to_string(j) + "/" + std::to_string(sim) + ".npz");
                                Events events;
                                size_t numberOfTimes = 1;
                                while (m_network.loadEvents(events, numberOfTimes)) {
                                    m_network.feedEvents(events);
                                }
                                events.clear();
                                if(direction == 0 ) {
                                    m_network.saveStatistics(sim, i, std::to_string(thickness) + "/speeds/" + std::to_string(j) + "/" + std::to_string(angle) + "/", false, sep_speed, num_speeds);
                                }
                                else {
                                    m_network.saveStatistics(sim, i, std::to_string(thickness) + "/speeds/" + std::to_string(j) + "/-" + std::to_string(angle) + "/", false, sep_speed, num_speeds);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void SurroundSuppression::evaluateResponsesOnStimuliMultipleOrientations(const std::string &eventsPath) {
    int n_angles = 8;
    int n_simulations = 2; //3;
    int angles[n_angles] = {0, 23, 45, 68, 90, 113, 135, 158};
    // int surrounding_angles[n_angles][2] = { {0,90}, {23, 113}, {45,135}, {68,158}, {90,0}, {113,23}, {135,45}, {158,68} };
    bool ok = verifyIfSafeMultipleOrientations(eventsPath, n_angles, n_simulations);
    int lat_values[1] = {0};
    int rval = 0;
    bool dynamic = true;
    if(ok) {
        for(auto val: lat_values) {
            if(val==-1) {
                dynamic = false;
            }
            else {
                dynamic = true;
            }
            m_network.deactivateDynamicInhib(dynamic);
            rval = val;
            for(const auto& main_angle: angles) {
                if(main_angle!=158) { 
                    continue;
                }
                for(const auto& surrounding_angle: angles) {
                    for(int sim=0; sim<n_simulations; sim++) {
                        std::cout << "Running simulation " << sim+1 << "/" << n_simulations << ". Lat_value = " << rval << "." << std::endl;
                        m_network.setEventPath(eventsPath + std::to_string(main_angle) + "/" + std::to_string(surrounding_angle) + "/" + std::to_string(sim) + ".npz");
                        std::cout << eventsPath + std::to_string(main_angle) + "/" + std::to_string(surrounding_angle) + "/" + std::to_string(sim) + ".npz" << std::endl;
                        Events events;
                        size_t numberOfTimes = 1;
                        while (m_network.loadEvents(events, numberOfTimes)) {
                            m_network.feedEvents(events);
                        }
                        events.clear();
                        m_network.saveStatistics(sim, 1, std::to_string(rval) + "/" + std::to_string(main_angle) + "/" + std::to_string(surrounding_angle) + "/");
                    }
                }
            }
        }
    }
}

void SurroundSuppression::evaluateResponsesSuppressionPatch(const std::string &eventsPath) {
    int n_angles = 8;
    int n_simulations = 2; //3;
    int angles[n_angles] = {0, 23, 45, 68, 90, 113, 135, 158};
    bool ok = verifyIfSafeMultipleOrientations(eventsPath, n_angles, n_simulations);
    int lat_values[1] = {0};
    double patches[16] = {0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2, 2.25, 2.5, 3};
    int rval = 3;
    bool dynamic = true;
    if(ok) {
        for(auto val: lat_values) {
            if(val==-1) {
                dynamic = false;
            }
            else {
                dynamic = true;
            }
            m_network.deactivateDynamicInhib(dynamic);
            for(const auto& main_angle: angles) {
                if(main_angle!=158) { 
                    continue;
                }
                for(const auto& surrounding_angle: angles) {
                    if(surrounding_angle == main_angle) {
                        int p = 0;
                        for(double& patch: patches) {
                            std::cout << "Running patch " << patch << "." << std::endl;
                            for(int sim=0; sim<n_simulations; sim++) {
                                m_network.setEventPath(eventsPath + std::to_string(main_angle) + "/" + std::to_string(surrounding_angle) + "/" + std::to_string(sim) + ".npz");
                                m_network.assignPatchSize(patch);
                                std::cout << eventsPath + std::to_string(main_angle) + "/" + std::to_string(surrounding_angle) + "/" + std::to_string(sim) + ".npz" << std::endl;
                                Events events;
                                size_t numberOfTimes = 1;
                                while (m_network.loadEvents(events, numberOfTimes)) {
                                    m_network.feedEvents(events);
                                }
                                events.clear();
                                m_network.saveStatistics(sim, 1, std::to_string(rval) + "/" + std::to_string(main_angle) + "/" + std::to_string(p) + "/");
                            }
                            p+=1;
                        }
                    }
                }
            }
        }
    }
}

void SurroundSuppression::evaluateNaturalEvents(const std::string &eventsPath) {
    std::vector<std::string> vectorOfPaths;
    for (const auto & frame : fs::directory_iterator{eventsPath}) {
            vectorOfPaths.emplace_back(frame.path().string());
        }
    size_t numberOfTimes = 1;
    double sigma = 5000;
    for(int i=0; i<vectorOfPaths.size();i++) {
        Events events;
        
        m_network.setEventPath(vectorOfPaths[i]);
        std::cout << "Event number : " << i+1 << " !" << std::endl;
        while (m_network.loadEvents(events, numberOfTimes)) {
            parameters_in_events params = getParameters(events, 0, 0, 1, 0, 0, 0);
            m_network.setSequenceParameters(params);
            m_network.feedEvents(events);
        }
        events.clear();
        m_network.saveStatistics(0, 1, "none");
    }
}

void SurroundSuppression::tuningCurvesData(const std::string &eventsPath) {
    int n_angles = 8;
    int n_fqcy = 7;
    int n_speeds = 3;
    int n_simulations = 3; 
    int angles[n_angles] = {0, 23, 45, 68, 90, 113, 135, 158}; 
    int fqcy[n_fqcy] = {1, 2, 3, 4, 5, 6, 7};
    for(const auto& frequence: fqcy) {
        for(const auto& main_angle: angles) {
            for(const auto& surrounding_angle: angles) {
                if(surrounding_angle != main_angle) {
                    continue;
                }
                for(int sim=0; sim<n_simulations; sim++) {
                        std::cout << "Running simulation " << sim+1 << "/" << n_simulations << "." << std::endl;
                        m_network.setEventPath(eventsPath + std::to_string(frequence) + "/" + std::to_string(main_angle) + "/" + std::to_string(surrounding_angle) + "/" + std::to_string(sim) + ".npz");
                        std::cout << eventsPath + std::to_string(frequence) + "/" + std::to_string(main_angle) + "/" + std::to_string(surrounding_angle) + "/" + std::to_string(sim) + ".npz" << std::endl;
                        Events events;
                        size_t numberOfTimes = 1;
                        while (m_network.loadEvents(events, numberOfTimes)) {
                            m_network.feedEvents(events);
                        }
                        events.clear();
                        m_network.saveStatistics(sim, 1, std::to_string(frequence) + "/" + std::to_string(main_angle) + "/" + std::to_string(surrounding_angle) + "/");
                }
            }
        }
    }
}

void SurroundSuppression::tuningCurvesCounterPhase(const std::string &eventsPath) {
    int n_angles = 8;
    int n_fqcy = 18;
    int n_phases = 11;
    int n_simulations = 2; 
    int angles[n_angles] = {0, 23, 45, 68, 90, 113, 135, 158}; 
    int fqcy[n_fqcy] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    int phases[n_phases] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    for(const auto& main_angle: angles) {
        if(main_angle!=113) {
            continue;
        }
        for(const auto& frequence: fqcy) {
            std::cout << "frequence = " << frequence << std::endl;
            for(const auto& phase: phases) {
                std::cout << "phase = " << phase << std::endl;
                for(int sim=0; sim<n_simulations; sim++) {
                        std::cout << "Running simulation " << sim+1 << "/" << n_simulations << "." << std::endl;
                        m_network.setEventPath(eventsPath + std::to_string(main_angle) + "/" + std::to_string(frequence) + "/" + std::to_string(phase) + "/" + std::to_string(sim) + ".npz");
                        std::cout << eventsPath + std::to_string(main_angle) + "/" + std::to_string(frequence) + "/" + std::to_string(phase) + "/" + std::to_string(sim) + ".npz" << std::endl;
                        Events events;
                        size_t numberOfTimes = 1;
                        while (m_network.loadEvents(events, numberOfTimes)) {
                            m_network.feedEvents(events);
                        }
                        events.clear();
                        m_network.saveStatistics(sim, 1, std::to_string(main_angle) + "/" + std::to_string(frequence) + "/" + std::to_string(phase) + "/");
                }
            }
        }
    }
}

void SurroundSuppression::classificationDescriptor(const std::vector<std::string> &vectorOfPath, const std::vector<int> &vectorOfLabels) {

    int num_labels = 11; 
    bool ok = true; 
    int sigma = 5000; 
    int id;
    int n_sim = 3500;
    std::vector<size_t> act_label;
    m_network.normalizeL1Weights();
    if(ok) {
        for(const auto & label : vectorOfLabels) {
            std::cout << "Label = " << label << std::endl;
            id = 0;
            act_label.push_back(label-1);
            parameters_in_events params; 
            params.push_back(act_label);
            int sim = 0;
            for (const auto & frame : std::filesystem::directory_iterator{vectorOfPath[label-1]}) { // label -1 for gestures
                if(sim==n_sim) {
                    break;
                }
                    std::cout << "File number " << id << " for label " << label-1 << "." << std::endl;
                    std::cout << frame.path().string() << std::endl;
                    m_network.setEventPath(frame.path().string());
                    Events events;
                    size_t numberOfTimes = 1;
                    while (m_network.loadEvents(events, numberOfTimes)) {
                        m_network.setSequenceParameters(params);
                        m_network.feedEvents(events);
                    }
                    events.clear();
                    m_network.saveStatistics(sim, 1, std::to_string(3) + "/" + std::to_string(label-1) + "/");
                    id+=1;
                    sim+=1;
            }
            act_label.clear();            
        }
    }
}

void SurroundSuppression::recordSpikes(const std::string &vectorOfPath) {

    bool ok = true; 
    int n_sim = 3500;
    if(ok) {
        int sim = 0;
        for (const auto & frame : std::filesystem::directory_iterator{vectorOfPath}) { 
            if(sim==n_sim) {
                break;
            }
            std::cout << frame.path().string() << std::endl;
            m_network.setEventPath(frame.path().string());
            Events events;
            size_t numberOfTimes = 1;
            while (m_network.loadEvents(events, numberOfTimes)) {
                m_network.feedEvents(events);
            }
            events.clear();
            m_network.saveStatistics(sim, 1, std::to_string(3) + "no/");
            sim+=1;
        }
    }
}

/**
 * Calculates the logarithmic value of the frame to simulate events.(from the ESIM simulator)
 * @param frame 
 * @param new_frame 
 */
void SurroundSuppression::convertFrame(image_matrix frame, image_matrix& new_frame) {
    image_matrix channels[3];
    cv::split(frame,channels);    
    for(int i=0; i<width_; i++) {
        for(int j=0; j<height_; j++) {
            new_frame.at<double>(cv::Point(i, j)) = (int)channels[0].at<uchar>(cv::Point(i, j))*0.299 + (int)channels[1].at<uchar>(cv::Point(i, j))*0.587 + (int)channels[2].at<uchar>(cv::Point(i, j))*0.114;
            if(new_frame.at<double>(cv::Point(i, j)) > m_log_threshold) {
                new_frame.at<double>(cv::Point(i, j)) = log(new_frame.at<double>(cv::Point(i, j)));
            }
        }
    }
}

/**
 * Generates events.
 * @param events 
 * @param delta_b 
 * @param thresh 
 * @param frame_id 
 * @param x 
 * @param y 
 * @param polarity 
 */
void SurroundSuppression::writeEvents(events_list &events, float delta_b, float thresh, float frame_id, int x, int y, int polarity) {
    std::random_device rd;
    std::default_random_engine generator(rd());

    std::uniform_int_distribution<int> distribution(floor(-m_timestamp_noise/2),floor(m_timestamp_noise/2)+1);
    int nb_event;
    int moddiff = int(delta_b/thresh);
    if(moddiff > m_n_max || moddiff <0) {
        nb_event = m_n_max;
    }
    else {
        nb_event = moddiff;
    }
    for(int e=0; e<nb_event; e++) {
        int random_val = distribution(generator);
        int timestamp = ( ( (m_time_gap * (e+1) * thresh) /delta_b) + m_time_gap * frame_id + random_val );
        if(timestamp < 0) {
            timestamp = 0;
        }
        events.emplace_back(Event(timestamp,x,y,polarity,false));
        Event ev(timestamp,x,y,polarity,false);
    }
}

/**
 * Converts successive frames to events if a brightness threshold is locally reached.
 * @param frame_id 
 * @param frame 
 * @param reference 
 * @param threshold_map 
 * @param events 
 */
void SurroundSuppression::frameToEvents(int frame_id, image_matrix frame, image_matrix reference, image_matrix &threshold_map, events_list &events) {
    image_matrix delta = frame - reference;
    for(int i=0; i<width_; i++) {
        for(int j=0; j<height_; j++) {
            if(delta.at<double>(cv::Point(i,j)) > threshold_map.at<double>(cv::Point(i,j))) {
                writeEvents(events, delta.at<double>(cv::Point(i,j)), threshold_map.at<double>(cv::Point(i,j)), frame_id, i, j, 1);
                threshold_map.at<double>(cv::Point(i,j)) *= (1+m_adapt_thresh_coef_shift);
            }
            else if(delta.at<double>(cv::Point(i,j)) < -threshold_map.at<double>(cv::Point(i,j))) {
                writeEvents(events, -delta.at<double>(cv::Point(i,j)), threshold_map.at<double>(cv::Point(i,j)), frame_id, i, j, 0);
                threshold_map.at<double>(cv::Point(i,j)) *= (1+m_adapt_thresh_coef_shift);
            }
            else if(delta.at<double>(cv::Point(i,j)) <= threshold_map.at<double>(cv::Point(i,j)) || delta.at<double>(cv::Point(i,j)) >= -threshold_map.at<double>(cv::Point(i,j))) {
                threshold_map.at<double>(cv::Point(i,j)) *= (1-m_adapt_thresh_coef_shift);
            }
        }
    }
}

/**
 * Creates events sequence from a folder of frames of a moving object.
 * @param pathToFrames 
 * @param events 
 * @param nbPass 
 */
void SurroundSuppression::createEvents(const std::string &pathToFrames, events_list &events, int nbPass) {
    image_matrix threshold_map(cv::Size(width_, height_), CV_64FC1, cv::Scalar(m_map_threshold));
    paths_container frames;
    for (const auto & frame : fs::directory_iterator{pathToFrames}) {
        frames.emplace_back(frame.path().filename().stem().string());
    }
    std::sort(frames.begin(), frames.end(), compareImgName);
    std::string extension = ".png";
    image_matrix init = cv::imread(pathToFrames + frames[0] + extension);
    image_matrix reference(cv::Size(width_, height_), CV_64FC1);
    convertFrame(init,reference);
    int sze = frames.size();
    int i =1;
    while(i<frames.size()) {
        std::string newpath = pathToFrames + frames[i] + extension;
        init = cv::imread(pathToFrames + frames[i] + extension);
        if (!init.data) {
            std::cout << "Could not open or find" << " the image" << std::endl;
        }
        image_matrix frame(cv::Size(width_, height_), CV_64FC1, cv::Scalar(0));
        convertFrame(init,frame);
        
        frameToEvents(i,frame,reference,threshold_map,events);
        for(int v=0; v<width_; v++) {
            for(int w = 0; w<height_; w++) {
                reference.at<double>(cv::Point(v,w)) = frame.at<double>(cv::Point(v,w));
            }
        }
        i++;
    }
    std::sort(events.begin(),events.end());
    int size = events.size();
    long firstTimestamp = static_cast<long>((events.at(0)).timestamp());
    long actualLast = static_cast<long>((events.at(size-1)).timestamp());
    long value = actualLast - firstTimestamp;
    
    for(i=0; i<nbPass; i++)
    {
        for (int j=0; j<size; j++)
        {
            if(i>0)
            {
                Event ev( (events.at(j)).timestamp() + static_cast<long>(i) * (value), (events.at(j)).x(), (events.at(j)).y(), (events.at(j)).polarity(), (events.at(j)).camera());
                events.push_back(ev);
            }
            else
            {
                break;
            }
        }
    }
}

/**
 * Re-initializes randomly the lateral inhibition weights.
 */
void SurroundSuppression::randomLateralInit(int norm_factor) {
    m_network.lateralRandom(norm_factor);
}

/**
 * Shuffles one or all inhibition weights.
 * @param c 
 */
void SurroundSuppression::shuffleInhibition(int c) {
    m_network.inhibitionShuffle(c);
}

/**
 * Orders correctly the png files generated by the class.
 * @param file_1 
 * @param file_2 
 * @return true 
 * @return false 
 */
bool SurroundSuppression::compareImgName(std::string file_1, std::string file_2) {
    if(file_1.at(0)=='i' && file_1.at(1)=='m' && file_1.at(2) == 'g' && file_1.at(3)== '_' && file_2.at(0)=='i' && file_2.at(1)=='m' && file_2.at(2) == 'g' && file_2.at(3)== '_') {
        file_1.erase(0,4);
        file_2.erase(0,4);
        return(std::stoi(file_1) < std::stoi(file_2));
    }
    else {
        return false;
    }
}

parameters_in_events SurroundSuppression::getParameters(events_list evts, int y, int theta, int thickness, int length, int dir, int id) {
    size_t window = 1000; //5000;
    size_t start = evts[0].timestamp();
    parameters_in_events allPos = findAllPossiblePositions(theta, y);
    std::vector<std::vector<int>> counter(width_, std::vector<int>(height_, 0));
    parameters_in_events to_return;
    int max_value = 0;
    int max_index_x = -1;
    int max_index_y = -1;
    size_t first_timestamp = 0;
    size_t timestamp = start;
    int memo_index = 0;
    bool ok = false;
    for(auto ev : evts) {
        //std::cout << ev.timestamp() << std::endl;
        if(ev.timestamp() - timestamp >= window) {
             if(max_index_x >=150 && max_index_x < 216 && max_index_y >= 100 && max_index_y < 166) {
                if(first_timestamp==0) {
                    first_timestamp = timestamp;
                }
                std::vector<size_t> params = {max_index_x, max_index_y, ev.timestamp(), theta, thickness, length, first_timestamp, dir, id};
                if(to_return.size()==0 && max_value!=0) {
                    to_return.push_back(params);
                }
                else if(to_return.size()!=0) {
                        first_timestamp = timestamp;
                        params[6] = first_timestamp;
                        to_return.push_back(params);
                        memo_index+=1;
                }
                
             }
            
            for(int i=0; i<width_; i++) {
                std::fill(counter[i].begin(), counter[i].end(), 0);
            }
            
            max_value = 0;
            max_index_x = -1;
            max_index_y = -1;
            timestamp = ev.timestamp();
            first_timestamp = 0;
            start = ev.timestamp();
        }

        for(auto elem : allPos) {
            if( elem[0]==ev.x() && elem[1]==ev.y() ) {
                ok = true;
                break;
            }
        }

        if(ok) {
            counter[ev.x()][ev.y()]+=1;
            if(counter[ev.x()][ev.y()] > max_value) {
                max_value = counter[ev.x()][ev.y()];
                max_index_x = ev.x();
                max_index_y = ev.y(); 
                timestamp = ev.timestamp();
            }
        }
    }

    return to_return;
}

parameters_in_events SurroundSuppression::findAllPossiblePositions(int theta, int center) {

    int real_track_coord = center*m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[1] *
            m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[2] + 
            center*m_network.getNetworkConfig().getLayerConnectivity()[0].sizes[2];
    int to_track_x = ((m_neuronsReceptiveField[0][real_track_coord]).x_plus+(m_neuronsReceptiveField[0][real_track_coord]).x_minus)/2 +
            m_network.getNetworkConfig().getLayerConnectivity()[0].patches[0][0];
    int to_track_y = ((m_neuronsReceptiveField[0][real_track_coord]).y_plus+(m_neuronsReceptiveField[0][real_track_coord]).y_minus)/2 +
            m_network.getNetworkConfig().getLayerConnectivity()[0].patches[1][0];
    double sin_rad = sin(theta*M_PI/180);
    double cos_rad = cos(theta*M_PI/180);
    double center_x = width_/2;
    double center_y = height_/2;
    double num = to_track_y + (sin_rad/cos_rad)*to_track_x - (sin_rad/cos_rad)*(1-cos_rad)*center_x + (sin_rad/cos_rad)*(sin_rad*center_y)-(sin_rad*center_x)-(1-cos_rad)*center_y;
    double den = sin_rad*sin_rad/cos_rad + cos_rad;
    int y = int(floor(num/den));
    
    std::vector<int> all_X;
    parameters_in_events to_return;
    for(int i=0; i<346; i++) {
        all_X.push_back(i);
    }

    for(auto x: all_X) {
            int new_x = round(cos_rad * x + sin_rad * y + (1 - cos_rad) * center_x - sin_rad * center_y);
            int new_y = round(-sin_rad * x + cos_rad * y + sin_rad * center_x + (1 - cos_rad) * center_y);
            std::vector<size_t> params;
            if((new_x >=0 && new_x <=345) && (new_y >=0 && new_y <= 259)) {
                params.push_back(new_x);
                params.push_back(new_y);
                to_return.push_back(params);
            }
    }
    return to_return;
}