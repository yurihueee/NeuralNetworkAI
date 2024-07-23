#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"
#include <string>
using namespace std;

int main(int argc, const char * argv[]) {
    // input variables
    string networkFile, trainFile, testFile, outputFile;
    double learningRate;
    int epochs;
    int choice;

    // corresponds to input variables above
    ifstream initialFile,trainingFile, trainedFile, testingFile;
    ofstream outFile;

    // ask user input for training or testing
    cout << "What would you like to do? " << endl;
    cout << "\t [1] train neural network" << endl;
    cout << "\t [2] test neural network" << endl;
    cin >> choice;
    cout << endl;


    if (choice == 1) {

        // ask for initial file
        cout << "Please enter initial neural network file " << endl;
        cout << "Network File: ";
        cin >> networkFile;
        cout << endl;
        initialFile.open(networkFile);

        // handle error if unable to network file
        while (!initialFile.is_open()){
            cout << "Unable to open neural network file" << endl;
            cout << "Network File: ";
            cin >> networkFile;
            initialFile.open(networkFile);
            cout << endl;

        }

        // ask for training file
        cout << "Please enter training file" << endl;
        cout << "Training file: ";
        cin >> trainFile;
        cout << endl;
        cout << endl;
        trainingFile.open(trainFile);
        
        // handle error if unable to open training file
        while (!trainingFile.is_open()){
            cout << "Unable to open training  file" << endl;
            cout << "Training File: ";
            cin >> trainFile;
            trainingFile.open(trainFile);
            cout << endl;

        }

        // ask for epochs
        cout << "Please enter the number of epochs" << endl;
        cout << "Number of epochs: ";
        cin >> epochs;
        cout << endl;
        cout << endl;


        // ask for learning rate
        cout << "Please enter the learning rate" << endl;
        cout << "Learning rate: ";
        cin >> learningRate;
        cout << endl;
        cout << endl;



        // ask for output file
        cout << "Please enter the name of your output file" << endl;
        cout << "Output file: ";
        cin >> outputFile;
        cout << endl;
        cout << endl;
        outFile.open(outputFile);
        


        NeuralNetwork *test = new NeuralNetwork(initialFile);
        test -> training(trainingFile, learningRate, epochs);
        test -> save(outFile);
    }

    if (choice == 2){
        // ask for initial file
        cout << "Please enter trained neural network file " << endl;
        cout << "Network File: ";
        cin >> networkFile;
        cout << endl;
        trainedFile.open(networkFile);

        // handle error if unable to network file
        while (!trainedFile.is_open()){
            cout << "Unable to open neural network file" << endl;
            cout << "Network File: ";
            cin >> networkFile;
            trainedFile.open(networkFile);
            cout << endl;

        }

        // ask for training file
        cout << "Please enter testing file" << endl;
        cout << "Testing file: ";
        cin >> testFile;
        cout << endl;
        cout << endl;
        testingFile.open(testFile);
        
        // handle error if unable to open training file
        while (!testingFile.is_open()){
            cout << "Unable to open training  file" << endl;
            cout << "Training File: ";
            cin >> testFile;
            testingFile.open(testFile);
            cout << endl;

        }
        
        // ask for output file
        cout << "Please enter the name of your output file" << endl;
        cout << "Output file: ";
        cin >> outputFile;
        cout << endl;
        outFile.open(outputFile);


        NeuralNetwork *test = new NeuralNetwork(trainedFile);
        test -> test(testingFile, outFile);

    }
    return 0;
}