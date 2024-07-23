CXXFLAGS = -std=c++11

NeuralNetwork.exe: main.o NeuralNetwork.o
	g++ $(CXXFLAGS) -o NeuralNetwork.exe main.o NeuralNetwork.o

main.o: main.cpp
	g++ $(CXXFLAGS) -c main.cpp

NeuralNetwork.o: NeuralNetwork.cpp
	g++ $(CXXFLAGS) -c NeuralNetwork.cpp
clean:
	rm *.exe*.o*.stackdump *~