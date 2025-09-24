#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
using namespace cv;

static inline double sigmoid(double z){return 1.0 / (1.0 + std::exp(-z));}

const int IMG_SIZE = 32;
const int INPUT_SIZE = IMG_SIZE * IMG_SIZE;
const int HIDDEN_SIZE = 64;
const int NUM_CLASSES = 1;





void loadIMG(const std::string &imgname, std::vector<double> &X){
	Mat img = imread(imgname);
	if(img.empty()) {std::cout<<"Cannont find the img: "<<imgname<<std::endl; exit(1);}
	
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat resized;
	resize(gray, resized, Size(IMG_SIZE, IMG_SIZE));
	

	for(int r = 0; r < IMG_SIZE; r++)
		for(int c = 0; c < IMG_SIZE; c++)
			X[r*IMG_SIZE +c] = resized.at<uchar>(r,c) / 255.0;

}




struct NeuralNetwork{
	std::vector<std::vector<double>> w_input_hidden;
	std::vector<double> b_hidden;
	std::vector<std::vector<double>> w_hidden_output;
	std::vector<double> b_output;
	
	NeuralNetwork(){
		w_input_hidden.resize(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
		b_hidden.resize(HIDDEN_SIZE,0.0);
		w_hidden_output.resize(HIDDEN_SIZE, std::vector<double>(NUM_CLASSES));
		b_output.resize(NUM_CLASSES,0.0);
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
};


void loadNN(const std::string& filename, NeuralNetwork &nn){
	
	std::ifstream myfile(filename);
	if(!myfile.is_open()) {std::cout<<"Cannont open file: "<<filename<<std::endl; return;}
	
	for(int i = 0; i < nn.w_input_hidden.size(); i++)
		for(int j = 0; j < nn.w_input_hidden[i].size(); j++)
			myfile >> nn.w_input_hidden[i][j];

	for(int i = 0; i < nn.b_hidden.size(); i++)
		myfile >> nn.b_hidden[i];

	for(int i = 0; i < nn.w_hidden_output.size(); i++)
		for(int j = 0; j < nn.w_hidden_output[i].size(); j++)
			myfile >> nn.w_hidden_output[i][j];

	for(int i = 0; i < nn.b_output.size(); i++)
		myfile >> nn.b_output[i];
	
	std::cout<<"Network loaded successfully\n";
	myfile.close();
}





int main(int argc, const char* argv[]){
	std::string image_file;
	std::string network_file;

	for(int i = 1; i < argc; i++){
		std::string arg = argv[i];

		if(arg == "-n" && i + 1 < argc){
			network_file = argv[i + 1];
			i++;
		} 
		else if(arg == "-i" && i + 1 < argc){
			image_file = argv[i + 1];
			i++;
		}
	}
	
	

	if(network_file.empty() || image_file.empty()){
		std::cerr<<"Usage: "<< argv[0] << " -n network.csv -i img.png"<<std::endl;
		return 1;
	}


	std::vector<double> X(INPUT_SIZE);
	loadIMG(image_file, X);


	NeuralNetwork nn;
	loadNN(network_file, nn);
	
	std::vector<double> hidden(HIDDEN_SIZE);
	std::vector<double> output(NUM_CLASSES);

	nn.forward(X, hidden, output);	

	std::cout<<"Predecit: "<<output[0]<<std::endl;
	
	return 0;

}

