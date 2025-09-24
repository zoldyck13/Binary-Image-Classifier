#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <random>
using namespace cv;


const int IMG_SIZE = 32;
const int INPUT_SIZE = IMG_SIZE * IMG_SIZE;
const int HIDDEN_SIZE = 64;
const int NUM_CLASSES = 1;
const double LEARNING_RATE = 0.01;
const int EPOCHS = 500;


// ===================== CUDA DEVICE FUNCTIONS =====================
__device__ double sigmoid(double z){
    return 1.0 / (1.0 + exp(-z));
}

__device__ double sigmoid_derivative(double a){
    return a * (1 - a);
}

// ===================== CUDA KERNELS =====================
// Forward hidden layer
__global__ void forwardHiddenKernel(const double* d_input, double* d_hidden, const double* d_w_input_hidden, const double* d_b_hidden){
    int j = threadIdx.x;
    if(j < HIDDEN_SIZE){
        double sum = d_b_hidden[j];
        for(int i = 0; i < INPUT_SIZE; i++)
            sum += d_input[i] * d_w_input_hidden[i*HIDDEN_SIZE + j];
        d_hidden[j] = sigmoid(sum);
    }
}

// Forward output layer
__global__ void forwardOutputKernel(const double* d_hidden, double* d_output, const double* d_w_hidden_output, const double* d_b_output){
    int k = threadIdx.x;
    if(k < NUM_CLASSES){
        double sum = d_b_output[k];
        for(int j = 0; j < HIDDEN_SIZE; j++)
            sum += d_hidden[j] * d_w_hidden_output[j*NUM_CLASSES + k];
        d_output[k] = sigmoid(sum);
    }
}

// Compute delta output
__global__ void computeDeltaOutputKernel(const double* d_output, const double* d_target, double* d_delta_output){
    int k = threadIdx.x;
    if(k < NUM_CLASSES)
        d_delta_output[k] = (d_output[k] - d_target[k]) * sigmoid_derivative(d_output[k]);
}

// Compute delta hidden
__global__ void computeDeltaHiddenKernel(const double* d_w_hidden_output, const double* d_delta_output, const double* d_hidden, double* d_delta_hidden){
    int j = threadIdx.x;
    if(j < HIDDEN_SIZE){
        double sum = 0.0;
        for(int k = 0; k < NUM_CLASSES; k++)
            sum += d_delta_output[k] * d_w_hidden_output[j*NUM_CLASSES + k];
        d_delta_hidden[j] = sum * sigmoid_derivative(d_hidden[j]);
    }
}

// Update weights input->hidden
__global__ void updateWeightsInputHidden(double* d_w_input_hidden, const double* d_input, const double* d_delta_hidden){
    int i = threadIdx.x;
    int j = threadIdx.y;
    if(i < INPUT_SIZE && j < HIDDEN_SIZE)
        d_w_input_hidden[i*HIDDEN_SIZE + j] -= LEARNING_RATE * d_delta_hidden[j] * d_input[i];
}

// Update weights hidden->output
__global__ void updateWeightsHiddenOutput(double* d_w_hidden_output, const double* d_hidden, const double* d_delta_output){
    int j = threadIdx.x;
    int k = threadIdx.y;
    if(j < HIDDEN_SIZE && k < NUM_CLASSES)
        d_w_hidden_output[j*NUM_CLASSES + k] -= LEARNING_RATE * d_delta_output[k] * d_hidden[j];
}

// Update biases
__global__ void updateBiases(double* d_b_hidden, const double* d_delta_hidden,
                             double* d_b_output, const double* d_delta_output){
    int idx = threadIdx.x;
    if(idx < HIDDEN_SIZE) d_b_hidden[idx] -= LEARNING_RATE * d_delta_hidden[idx];
    if(idx < NUM_CLASSES) d_b_output[idx] -= LEARNING_RATE * d_delta_output[idx];
}



// ===================== DATASET LOADING =====================
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


void saveNN(const std::string &filename,
            const std::vector<double> &w_input_hidden,
            const std::vector<double> &b_hidden,
            const std::vector<double> &w_hidden_output,
            const std::vector<double> &b_output) 
{
    std::ofstream myfile(filename);

    for(size_t i = 0; i < w_input_hidden.size(); i++)
        myfile << w_input_hidden[i] << "\n";

    for(size_t i = 0; i < b_hidden.size(); i++)
        myfile << b_hidden[i] << "\n";

    for(size_t i = 0; i < w_hidden_output.size(); i++)
        myfile << w_hidden_output[i] << "\n";

    for(size_t i = 0; i < b_output.size(); i++)
        myfile << b_output[i] << "\n";

    std::cout << "Network saved to " << filename << std::endl;
}



int main(){
    std::vector<std::vector<double>> X, Y;
    loadDataset("dataset", X, Y);
    std::cout << "Loaded " << X.size() << " samples." << std::endl;

    // ===================== INITIALIZE NETWORK =====================
    std::vector<double> w_input_hidden(INPUT_SIZE*HIDDEN_SIZE);
    std::vector<double> b_hidden(HIDDEN_SIZE,0.0);
    std::vector<double> w_hidden_output(HIDDEN_SIZE*NUM_CLASSES);
    std::vector<double> b_output(NUM_CLASSES,0.0);

	static std::random_device rd;
	static std::mt19937 gen(rd());
	double limit = std::sqrt(6.0 / (INPUT_SIZE+HIDDEN_SIZE));
	double limit1 = std::sqrt(6.0 / (HIDDEN_SIZE+NUM_CLASSES));
	double limit2 = std::sqrt(6.0 / (HIDDEN_SIZE));
	double limit3 = std::sqrt(6.0 / (NUM_CLASSES));
	std::uniform_real_distribution<double> dist(-limit, limit);
	std::uniform_real_distribution<double> dist1(-limit1, limit1);
	std::uniform_real_distribution<double> dist2(-limit2, limit2);
	std::uniform_real_distribution<double> dist3(-limit3, limit3);


    for(auto &w : w_input_hidden) w = dist(gen);
    for(auto &w : w_hidden_output) w = dist1(gen);
	for(auto &b : b_hidden) b = dist2(gen);
	for(auto &b : b_output) b = dist3(gen);

    // ===================== ALLOCATE GPU MEMORY =====================
    double *d_input, *d_hidden, *d_output;
    double *d_w_input_hidden, *d_b_hidden, *d_w_hidden_output, *d_b_output;
    double *d_delta_hidden, *d_delta_output;
    double *d_target;

    cudaMalloc(&d_input, INPUT_SIZE*sizeof(double));
    cudaMalloc(&d_hidden, HIDDEN_SIZE*sizeof(double));
    cudaMalloc(&d_output, NUM_CLASSES*sizeof(double));
    cudaMalloc(&d_w_input_hidden, INPUT_SIZE*HIDDEN_SIZE*sizeof(double));
    cudaMalloc(&d_b_hidden, HIDDEN_SIZE*sizeof(double));
    cudaMalloc(&d_w_hidden_output, HIDDEN_SIZE*NUM_CLASSES*sizeof(double));
    cudaMalloc(&d_b_output, NUM_CLASSES*sizeof(double));
    cudaMalloc(&d_delta_hidden, HIDDEN_SIZE*sizeof(double));
    cudaMalloc(&d_delta_output, NUM_CLASSES*sizeof(double));
    cudaMalloc(&d_target, NUM_CLASSES*sizeof(double));

    cudaMemcpy(d_w_input_hidden, w_input_hidden.data(), INPUT_SIZE*HIDDEN_SIZE*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_hidden, b_hidden.data(), HIDDEN_SIZE*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_hidden_output, w_hidden_output.data(), HIDDEN_SIZE*NUM_CLASSES*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_output, b_output.data(), NUM_CLASSES*sizeof(double), cudaMemcpyHostToDevice);

    // ===================== TRAINING =====================
    for(int epoch=0; epoch<EPOCHS; epoch++){
        double total_loss = 0.0;

        for(size_t sample=0; sample<X.size(); sample++){
            
            cudaMemcpy(d_input, X[sample].data(), INPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, Y[sample].data(), NUM_CLASSES*sizeof(double), cudaMemcpyHostToDevice);

            // Forward pass
            forwardHiddenKernel<<<1,HIDDEN_SIZE>>>(d_input,d_hidden,d_w_input_hidden,d_b_hidden);
            cudaDeviceSynchronize();
            forwardOutputKernel<<<1,NUM_CLASSES>>>(d_hidden,d_output,d_w_hidden_output,d_b_output);
            cudaDeviceSynchronize();

            
            computeDeltaOutputKernel<<<1,NUM_CLASSES>>>(d_output,d_target,d_delta_output);
            cudaDeviceSynchronize();
            computeDeltaHiddenKernel<<<1,HIDDEN_SIZE>>>(d_w_hidden_output,d_delta_output,d_hidden,d_delta_hidden);
            cudaDeviceSynchronize();

           
            dim3 threadsIH(INPUT_SIZE,HIDDEN_SIZE);
            updateWeightsInputHidden<<<1,threadsIH>>>(d_w_input_hidden,d_input,d_delta_hidden);
            cudaDeviceSynchronize();

            dim3 threadsHO(HIDDEN_SIZE,NUM_CLASSES);
            updateWeightsHiddenOutput<<<1,threadsHO>>>(d_w_hidden_output,d_hidden,d_delta_output);
            cudaDeviceSynchronize();

            updateBiases<<<1,HIDDEN_SIZE>>>(d_b_hidden,d_delta_hidden,d_b_output,d_delta_output);
            cudaDeviceSynchronize();
        }

        if(epoch%50==0) std::cout<<"Epoch "<<epoch<<" done."<<std::endl;
    }

    std::cout<<"Training complete."<<std::endl;

    // ===================== COPY WEIGHTS BACK TO HOST =====================
    cudaMemcpy(w_input_hidden.data(),d_w_input_hidden,INPUT_SIZE*HIDDEN_SIZE*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(b_hidden.data(),d_b_hidden,HIDDEN_SIZE*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(w_hidden_output.data(),d_w_hidden_output,HIDDEN_SIZE*NUM_CLASSES*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(b_output.data(),d_b_output,NUM_CLASSES*sizeof(double),cudaMemcpyDeviceToHost);
	

    // ===================== SAVE THE NETWORK ================
    saveNN("Network.csv", w_input_hidden, b_hidden, w_hidden_output, b_output);


    // ===================== CLEANUP =====================
    cudaFree(d_input); cudaFree(d_hidden); cudaFree(d_output);
    cudaFree(d_w_input_hidden); cudaFree(d_b_hidden);
    cudaFree(d_w_hidden_output); cudaFree(d_b_output);
    cudaFree(d_delta_hidden); cudaFree(d_delta_output); cudaFree(d_target);

    return 0;
}
