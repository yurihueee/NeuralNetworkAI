
#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>
#include <cmath>
#include <iomanip>

using namespace std;

double A, B, C, D;
double overall, precision, recall, f1;
double avOverall, avPrecision, avRecall, avF1;

NeuralNetwork::NeuralNetwork(ifstream &initial)
{
    numLayer = 3; // only one hidden layer, so three layers total

    // resize vectors to hold the layers and their sizes
    layerSize.resize(numLayer);
    layer.resize(numLayer);

    for (int i = 0; i < numLayer; i++)
    {
        initial >> layerSize[i];
        layerSize[i]++;
        layer[i].resize(layerSize[i]);
    }
    // for w0 it is set to -1
    for (int i = 0; i < numLayer; i++)
    {

        layer[i][0].activation = -1;
    }

    for (int layerIndex = 0; layerIndex < numLayer - 1; ++layerIndex) {
        for (int neuronIndex = 1; neuronIndex < layerSize[layerIndex + 1]; ++neuronIndex) {
            for (int prevLayerNeuronIndex = 0; prevLayerNeuronIndex < layerSize[layerIndex]; ++prevLayerNeuronIndex) {
                double weight;
                initial >> weight;

                // Create a link from the current neuron in the previous layer to the neuron in the current layer
                Link forwardLink;
                forwardLink.weight = weight;
                forwardLink.connected = &layer[layerIndex + 1][neuronIndex];
                layer[layerIndex][prevLayerNeuronIndex].outLink.push_back(forwardLink);

                // Create a link back from the neuron in the current layer to the current neuron in the previous layer
                Link backwardLink;
                backwardLink.weight = weight;
                backwardLink.connected = &layer[layerIndex][prevLayerNeuronIndex];
                layer[layerIndex + 1][neuronIndex].inLink.push_back(backwardLink);
            }
        }
    }

}


// method for training network with given dataset
// back prop
int NeuralNetwork::training(ifstream &training, double learnRate, int epoch)
{


    int inputN, outputN, setN;
    training >> setN >> inputN >> outputN;

    // vector to hold all training examples
    vector<TrainingExample> trainingSet(setN);

    // Read each training example
    for (auto &example : trainingSet) {
        // Resize input and output vectors within each training example
        example.input.resize(inputN);
        example.output.resize(outputN);

        // Read inputs
        for (double &input : example.input) {
            training >> input;
        }

        // Read outputs and convert to bool
        for (int k = 0; k < outputN; ++k) {
            int outputValue;
            training >> outputValue;
            example.output[k] = static_cast<bool>(outputValue);
        }
    }

    int outputI = numLayer - 1;

    for (int i = 0; i < epoch; i++)
    {
        for (int c = 0; c < setN; c++)
        {
            for (int j = 0; j < inputN; j++)
            {

                // copy training examples to input node of network
                layer[0][j + 1].activation = trainingSet[c].input[j];

            }
            for (int l = 1; l < numLayer; l++)
            {
                for (int n = 1; n < layerSize[l]; n++)
                {
                    layer[l][n].inputValue = 0;
                    vector<Link>::iterator it;
                    for (it = layer[l][n].inLink.begin(); it != layer[l][n].inLink.end(); it++)
                    {
                        layer[l][n].inputValue += it->weight * it->connected->activation;
                    }
                    layer[l][n].activation = sigmoid(layer[l][n].inputValue);
                }
            }

            for (int n = 1; n < layerSize[outputI]; n++)
            {
                layer[outputI][n].error = sigmoidPrime(layer[outputI][n].inputValue) * (static_cast<double>(trainingSet[c].output[n - 1]) - layer[outputI][n].activation);
            }
            for (int l = outputI - 1; l > 0; l--)
            {
                for (int ii = 1; ii < layerSize[l]; ii++)
                {
                    double sum = 0;
                    vector<Link>::iterator it;
                    for (it = layer[l][ii].outLink.begin(); it != layer[l][ii].outLink.end(); it++)
                    {
                        sum += it->weight * it->connected->error;
                    }
                    layer[l][ii].error = sigmoidPrime(layer[l][ii].inputValue) * sum;
                }
            }
            for (int l = 1; l < numLayer; l++)
            {
                for (int j = 1; j < layerSize[l]; j++)
                {
                    vector<Link>::iterator it;
                    for (it = layer[l][j].inLink.begin(); it != layer[l][j].inLink.end(); it++)
                    {
                        it->weight = it->weight + learnRate * it->connected->activation * layer[l][j].error;
                        it->connected->outLink[j - 1].weight = it->weight;
                    }
                }
            }
        }
    }

    
    return 0;
}



int NeuralNetwork::test(ifstream &test, ofstream &output)
{
    int setN, inputN, outputN;
    vector<TrainingExample> example;
    vector<vector<double> > result;
    test >> setN >> inputN >> outputN;
    example.resize(setN);
    result.resize(outputN);

    for (int i = 0; i < setN; i++)
    {
        example[i].input.resize(inputN);
        example[i].output.resize(outputN);
        for (int j = 0; j < inputN; j++)
        {
            test >> example[i].input[j];
        }
        for (int k = 0; k < outputN; k++)
        {
            int outputValue;
            test >> outputValue;
            example[i].output[k] = static_cast<bool>(outputValue);
            if (i == 0)
            {
                result[k].resize(4);
                for (int m = 0; m < 4; m++)
                {
                    result[k][m] = 0;
                }
            }
        }
    }

    int outputI = numLayer - 1;
    for (int c = 0; c < setN; c++)
    {
        for (int i = 0; i < inputN; i++)
        {
            layer[0][i + 1].activation = example[c].input[i];
        }
        for (int l = 1; l < numLayer; l++)
        {
            for (int j = 1; j < layerSize[l]; j++)
            {
                layer[l][j].inputValue = 0;
                vector<Link>::iterator it;
                for (it = layer[l][j].inLink.begin(); it != layer[l][j].inLink.end(); it++)
                {
                    layer[l][j].inputValue += it->weight * (it->connected->activation);
                }
                layer[l][j].activation = sigmoid(layer[l][j].inputValue);
            }
        }
        for (int n = 1; n < layerSize[outputI]; n++)
        {
            if (layer[outputI][n].activation >= 0.5)
            {
                if (example[c].output[n - 1])
                {
                    result[n - 1][0]++;
                }
                else
                {
                    result[n - 1][1]++;
                }
            }
            else
            {
                if (example[c].output[n - 1])
                {
                    result[n - 1][2]++;
                }
                else
                {
                    result[n - 1][3]++;
                }
            }
        }
    }


    // set precision to three decimal places for floating point
    output << setprecision(3) << fixed;

    // create a vector of confusion matrix, one for each class
    vector<ConfusionMatrix> matrices(outputN);

    // create confusion matrix instances for macro and micro averaging
    ConfusionMatrix macroAvg, microAvg; // For averaging

    // initialize total counts for micro averaging
    double totalA = 0, totalB = 0, totalC = 0, totalD = 0;

    // interate over each class to process results
    for (int i = 0; i < outputN; i++) {
        // extract individual counts from the result matrix
        double A = result[i][0]; // true positives
        double B = result[i][1]; // false positives
        double C = result[i][2]; // false negatives
        double D = result[i][3]; // true negatives

        // update totals for micro averaging
        totalA += A;
        totalB += B;
        totalC += C;
        totalD += D;

        // calculate accuracy, precision, recall, and f1 score for each class
        matrices[i].accuracy = (A + D) / (A + B + C + D);
        matrices[i].precision = A / (A + B);
        matrices[i].recall = A / (A + C);
        matrices[i].f1 = (2 * matrices[i].precision * matrices[i].recall) / (matrices[i].precision + matrices[i].recall);

        matrices[i].handleNaN(); // handle any division by zero cases (NaN)

        // output the confusion matrix counts and calculated metrics for each class/category
        output << (int)A << " " << (int)B << " " << (int)C << " " << (int)D << " ";
        output << matrices[i].accuracy << " " << matrices[i].precision << " " << matrices[i].recall << " " << matrices[i].f1 << endl;

        // accumulate metrics for macro averaging
        macroAvg.accuracy += matrices[i].accuracy;
        macroAvg.precision += matrices[i].precision;
        macroAvg.recall += matrices[i].recall;
        macroAvg.f1 += matrices[i].f1;
    }

    // perform macro averaging
    int numClasses = outputN;
    macroAvg.accuracy /= numClasses;
    macroAvg.precision /= numClasses;
    macroAvg.recall /= numClasses;
    macroAvg.f1 = (2 * macroAvg.precision * macroAvg.recall) / (macroAvg.precision + macroAvg.recall); // avg of individual f1 scores

    macroAvg.handleNaN(); // handle any division by zero cases (NaN)
    

    // calculate and output micro-averaged metrics
    microAvg.accuracy = (totalA + totalD) / (totalA + totalB + totalC + totalD);
    microAvg.precision = totalA / (totalA + totalB);
    microAvg.recall = totalA / (totalA + totalC);
    microAvg.f1 = (2 * microAvg.precision * microAvg.recall) / (microAvg.precision + microAvg.recall);

    microAvg.handleNaN(); // handle any division by zero cases (NaN)

    // Output the macro-averaged and micro-averaged metrics
    output << microAvg.accuracy << " " << microAvg.precision << " " << microAvg.recall << " " << microAvg.f1 << endl;
    output << macroAvg.accuracy << " " << macroAvg.precision << " " << macroAvg.recall << " " << macroAvg.f1 << endl;

    return 0;

}

void NeuralNetwork::save(ostream &output) {
    output << setprecision(3) << fixed;

    // Output layer sizes
    for (int l = 0; l < numLayer; ++l) {
        if (l != 0) output << " ";
        output << layerSize[l] - 1;
    }
    output << endl;

    // Output weights of the links
    for (int i = 1; i < numLayer; ++i) {
        for (int j = 1; j < layerSize[i]; ++j) {
            auto &inLinks = layer[i][j].inLink;
            for (size_t k = 0; k < inLinks.size(); ++k) {
                if (k != 0) output << " ";
                output << inLinks[k].weight;
            }
            output << endl;
        }
    }
}
