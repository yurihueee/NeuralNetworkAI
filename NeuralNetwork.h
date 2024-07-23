#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include <vector>
#include <fstream>


using namespace std;


struct TrainingExample {
    vector<double> input; // input value for training
    vector<bool> output; // expected output values for training
};

struct ConfusionMatrix {
    double accuracy;
    double precision;
    double recall;
    double f1;
    ConfusionMatrix() : accuracy(0), precision(0), recall(0), f1(0) {}

    // method to handle division by zero (NaN cases)
    void handleNaN() {
        if (accuracy != accuracy) accuracy = 0;   // Check for NaN in accuracy
        if (precision != precision) precision = 0; // Check for NaN in precision
        if (recall != recall) recall = 0;         // Check for NaN in recall
        if (f1 != f1) f1 = 0;                     // Check for NaN in F1 score
    }
};



class NeuralNetwork{
public:
    // initialize network
    NeuralNetwork(ifstream &initial);

    // method for training network with given dataset
    int training(ifstream &training, double learnRate, int epochs);

    // method for testing network and writing to output 
    int test(ifstream &test, ofstream &output);

    // method for saving trained network's weight to an output file
    void save(ostream &output);


    double sigmoid(double x){
        return (1.0 / (1.0 + exp(-x)));
    }
    double sigmoidPrime(double x) {
        return (sigmoid(x) * (1 - sigmoid(x)));
    }

private:
    class Neuron;

    // inner class representing a link between neurons
    class Link{
    public:
        double weight; // weight of connection
        Neuron *connected; // pointer to connected neuron
    };

    // class for neuron
    class Neuron{
    public:
        double inputValue; // input value for neuron
        double activation; // activation (output) value of neuron 
        double error; // the error value used in training
        vector<Link> inLink; // vector of incoming links
        vector<Link> outLink; // vector of outgoing links
    };

    int numLayer; // number of layers in the network
    vector<int> layerSize; // sizes of each layer
    vector<vector<Neuron> > layer; // 2d vector for activation function
};

#endif /* network_hpp */