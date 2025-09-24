#include <iostream>
#include <cmath>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <random>
using namespace cv;


static inline double sigmoid(double z){ return 1.0 / (1.0 + std::exp(-z));}
static inline double sigmoid_derivative(double a){ return a * (1 - a); }


const int IMG_SIZE = 32;
const int INPUT_SIZE = IMG_SIZE * IMG_SIZE;
const int HIDDEN_SIZE = 64;
const int NUM_CLASSES = 1;
const double LEARNING_RATE = 0.01;
const int EPOCHS = 500;



void loadDataset(const std::string &path, std::vector<std::vector<double>> &X, std::vector<std::vector<double>> &Y){
    
    std::vector<std::string> categories = {"class_a", "class_b"}; 

    for(size_t class_index = 0; class_index < categories.size(); class_index++){
        std::string folder = path + "/" + categories[class_index];

        if(!std::filesystem::exists(folder)) {
            std::cerr << "Warning: folder \"" << folder << "\" does not exist!" << std::endl;
            continue;
        }

        int count = 0; 

        for(const auto &img_file : std::filesystem::directory_iterator(folder)){
            cv::Mat img = cv::imread(img_file.path().string());
            if(img.empty()) continue;

            cv::Mat gray, resized;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::resize(gray, resized, cv::Size(IMG_SIZE, IMG_SIZE));

            std::vector<double> input(INPUT_SIZE);
            for(int r = 0; r < IMG_SIZE; r++)
                for(int c = 0; c < IMG_SIZE; c++)
                    input[r*IMG_SIZE + c] = resized.at<uchar>(r,c) / 255.0;

            X.push_back(input);

            std::vector<double> output(NUM_CLASSES, class_index == 0 ? 1.0 : 0.0);
            Y.push_back(output);

            count++; 
        }

        if(count > 0) {
            std::cout << "Folder \"" << folder << "\" loaded with class_index " 
                      << class_index << " (" << count << " images)" << std::endl;
            std::cout.flush(); 
        }
    }
}







struct NeuralNetwork{

    std::vector<std::vector<double>> w_input_hidden;
    std::vector<double> b_hidden;
    std::vector<std::vector<double>> w_hidden_output;
    std::vector<double> b_output;

    NeuralNetwork(){
        w_input_hidden.resize(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
        b_hidden.resize(HIDDEN_SIZE, 0.0);
        w_hidden_output.resize(HIDDEN_SIZE, std::vector<double>(NUM_CLASSES));
        b_output.resize(NUM_CLASSES, 0.0);
		
		static std::random_device rd;
		static std::mt19937 gen(rd());
		double limit  = std::sqrt(6.0 / (INPUT_SIZE+HIDDEN_SIZE));
		double limit1 = std::sqrt(6.0 / (HIDDEN_SIZE+NUM_CLASSES));
		double limit2 = std::sqrt(6.0 / (HIDDEN_SIZE));
		double limit3 = std::sqrt(6.0 / (NUM_CLASSES));
		std::uniform_real_distribution<double> dist(-limit, limit);
		std::uniform_real_distribution<double> dist1(-limit1, limit1);
		std::uniform_real_distribution<double> dist2(-limit2, limit2);
		std::uniform_real_distribution<double> dist3(-limit3, limit3);
		
        for(int i = 0; i < INPUT_SIZE; i++)
            for(int j = 0; j < HIDDEN_SIZE; j++)
                w_input_hidden[i][j] = dist(gen);


        for(int i = 0; i < HIDDEN_SIZE; i++)
            for(int j = 0; j < NUM_CLASSES; j++)
                w_hidden_output[i][j] = dist1(gen);
		

		for(int i = 0; i < HIDDEN_SIZE; i++)
			b_hidden[i] = dist2(gen);

		for(int i = 0; i < NUM_CLASSES; i++)
			b_output[i] = dist3(gen);

    }

    void forward(const std::vector<double> &input, std::vector<double> &hidden, std::vector<double> &output){
        for(int j = 0; j < HIDDEN_SIZE; j++){
            double sum = b_hidden[j];
            for(int i = 0; i < INPUT_SIZE; i++)
                sum += input[i] * w_input_hidden[i][j];
            hidden[j] = sigmoid(sum);
        }

        for(int k = 0; k < NUM_CLASSES; k++){
            double sum = b_output[k];
            for(int j = 0; j < HIDDEN_SIZE; j++)
                sum += hidden[j] * w_hidden_output[j][k];
            output[k] = sigmoid(sum);
        }
    }


    double loss(const std::vector<double> &output, const std::vector<double> &target){
        double sum = 0.0;
        for(int i = 0; i < output.size(); i++)
            sum += (output[i] - target[i]) * (output[i] - target[i]);

        return sum / output.size();
    }


    double trainSample(const std::vector<double> &input, const std::vector<double> &target){
        std::vector<double> hidden(HIDDEN_SIZE), output(NUM_CLASSES);
        forward(input, hidden, output);

        double L = loss(output, target);

        std::vector<double> delta_output(NUM_CLASSES);
        for(int k = 0; k < NUM_CLASSES; k++)
            delta_output[k] = (output[k] - target[k]) * sigmoid_derivative(output[k]);

        std::vector<double> delta_hidden(HIDDEN_SIZE);
        for(int j = 0; j < HIDDEN_SIZE; j++){
            double sum = 0.0;
            for(int k = 0; k < NUM_CLASSES; k++)
                sum += delta_output[k] * w_hidden_output[j][k];
            delta_hidden[j] = sum * sigmoid_derivative(hidden[j]);
        }


        for(int j = 0; j < HIDDEN_SIZE; j++)
            for(int k = 0; k < NUM_CLASSES; k++)
                w_hidden_output[j][k] -= LEARNING_RATE*delta_output[k]*hidden[j];

        for(int j = 0; j < NUM_CLASSES; j++)
            b_output[j] -= LEARNING_RATE * delta_output[j];

        for(int i = 0; i < INPUT_SIZE; i++)
            for(int j = 0; j < HIDDEN_SIZE; j++)
                w_input_hidden[i][j] -= LEARNING_RATE * delta_hidden[j]*input[i];

        for(int j = 0; j < HIDDEN_SIZE; j++)
            b_hidden[j] -= LEARNING_RATE * delta_hidden[j];

    return L;
    }


};


void saveNN(const std::string filename, const NeuralNetwork &nn){

	std::ofstream myfile(filename);

	for(size_t i = 0; i < nn.w_input_hidden.size(); i++)
		for(size_t j = 0; j < nn.w_input_hidden[i].size(); j++)
			myfile << nn.w_input_hidden[i][j]<<"\n";
	
	for(size_t i = 0; i < nn.b_hidden.size(); i++)
		myfile << nn.b_hidden[i]<<"\n";

	for(size_t i = 0; i < nn.w_hidden_output.size(); i++)
		for(size_t j = 0; j < nn.w_hidden_output[i].size(); j++)
			myfile << nn.w_hidden_output[i][j]<<"\n";
	

	for(size_t i = 0; i < nn.b_output.size(); i ++)
		myfile << nn.b_output[i]<<"\n";

		std::cout<<"Network save complete."<<std::endl;

	}



int main(int argc, char* argv[]){

    if(argc < 2){std::cerr<< "Usage: " << argv[0] <<" <dataset_path>" <<std::endl; return 1;}

    std::string dataset_path = argv[1];

    std::vector<std::vector<double>> X, Y;
    loadDataset(dataset_path,X,Y);

    std::cout << "Loaded " << X.size() << " samples." << std::endl;

    NeuralNetwork nn;

    for(int epoch = 0; epoch < EPOCHS; epoch++){
        double total_loss = 0.0;
        for(size_t i = 0; i < X.size(); i++){
            total_loss += nn.trainSample(X[i], Y[i]);
        }

        if(epoch % 50 == 0){ 

        std::cout<<"Epoch: "<<epoch<<" || Loss: "<<total_loss / X.size()<<std::endl;

        }

    }
	

	for(size_t i = 0; i < X.size(); i++){
		std::vector<double> hidden(HIDDEN_SIZE);
		std::vector<double> output(NUM_CLASSES);
		nn.forward(X[i], hidden, output);
		std::cout<< "Sample " << i << " targer= "<<Y[i][0]<<" pred="<< output[0]<<std::endl;
	}

	saveNN("Network.csv", nn);


    return 0;

}
