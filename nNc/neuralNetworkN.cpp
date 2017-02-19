#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace std;

//constants
const int NUMINPUTNODES = 2;
const int NUMHIDDENNODES = 2;
const int NOMOUTPUTNODES = 1;
const int NUMNODES = NUMINPUTNODES + NUMHIDDENNODES + NOMOUTPUTNODES;

const int ARRAYSIZE = NUMNODES + 1; //1 - offset to match node 1 node 2 etc
const int MAXITERATIONS = 131072;

const double E = 2.71828;
const double LEARNINGRATE = 0.2;

//function prototypes
void initialise(double[][ARRAYSIZE], double[], double[], double[]);
void connectNodes(double[][ARRAYSIZE], double[]);
void trainingExample(double[], double[]);
void activateNetwork(double[][ARRAYSIZE], double[], double[]);
double updateWeights(double[][ARRAYSIZE], double[], double[], double[]);
void displayNetwork(double[], double);

int main() {

	std::cout<<"Neural Network Program"<<endl;

	double weights[ARRAYSIZE][ARRAYSIZE];
	double values[ARRAYSIZE];
	double expectedValues[ARRAYSIZE];
	double thresholds[ARRAYSIZE];

	initialise(weights, values, expectedValues, thresholds);
	connectNodes(weights, thresholds);
	int counter = 0;

	while (counter < MAXITERATIONS) {
		trainingExample(values, expectedValues);
		activateNetwork(weights, values, thresholds);
		double sumOfSquaredErrors = updateWeights(weights, values, expectedValues, thresholds);
		displayNetwork(values, sumOfSquaredErrors);
		counter++;
	}
	return 0;
}

void initialise(double weights[][ARRAYSIZE], double values[], double expectedValues[], double thresholds[]) {
	for (size_t x = 0; x < NUMNODES; x++) {
		values[x] = 0.0;
		expectedValues[x] = 0.0;
		thresholds[x] = 0.0;
		for (size_t y = 0; y < NUMNODES; y++) {
			weights[x][y] = 0.0;
		}
	}
}

void connectNodes(double weights[][ARRAYSIZE], double thresholds[]) {
	for (size_t x = 1; x < NUMNODES + 1; x++) {
		for (size_t y = 1; y < NUMNODES + 1; y++) {
			weights[x][y] = (rand() % 200) / 200.0 - 0.5;
		}
	}

	thresholds[3] = rand() / (double)rand();
	thresholds[4] = rand() / (double)rand();
	thresholds[5] = rand() / (double)rand();

	cout<< weights[1][3] <<" "<< weights[1][4] << " " << weights[2][3]<<\
		" " << weights[2][4] << " " << weights[3][5] << " " << weights[4][5] << endl\
		<< thresholds[3] << " " << thresholds[4] << " " << thresholds[5]<<endl;

}

void trainingExample(double values[], double expectedValues[]) {
	static int counter = 0;

	switch (counter % 4) {
	case 0:
		values[1] = 1.0;
		values[2] = 1.0;
		expectedValues[5] = 0.0;
		break;
	case 1:
		values[1] = 0.0;
		values[2] = 1.0;
		expectedValues[5] = 1.0;
		break;
	case 2:
		values[1] = 1.0;
		values[2] = 0.0;
		expectedValues[5] = 1.0;
		break;
	case 3:
		values[1] = 0.0;
		values[2] = 0.0;
		expectedValues[5] = 0.0;
		break;
	}
	counter++;
}

void activateNetwork(double weights[][ARRAYSIZE], double values[], double thresholds[]) {
	//double weightedInput = 0.0;
	for (size_t h = 1 + NUMINPUTNODES; h < 1 + NUMINPUTNODES + NUMHIDDENNODES; h++) {
		double weightedInput = 0.0;
		//add up the weihted input
		for (size_t i = 1; i <1 + NUMINPUTNODES; i++) {
			weightedInput += weights[i][h] * values[i];
		}
		weightedInput += (-1.0 * thresholds[h]);

		double tempValue = 1.0 / (1.0 + pow(E, -weightedInput));
		values[h] = tempValue;// 1.0 / (1.0 + pow(E, -weightedInput));
	}
	for (size_t out = 1 + NUMINPUTNODES + NUMHIDDENNODES; out < 1 + NUMNODES; out++) {
		double weightedInput = 0.0;
		for (size_t h = 1 + NUMINPUTNODES; h < 1 + NUMINPUTNODES + NUMHIDDENNODES; h++) {
			weightedInput += weights[h][out] * values[h];
		}
		//handle the thresholds
		weightedInput += (-1.0 * thresholds[out]);
		double tempValue = 1.0 / (1.0 + pow(E, -weightedInput));
		values[out] = tempValue; // 1.0 / (1.0 + pow(E, -weightedInput));
	}
}

double updateWeights(double weights[][ARRAYSIZE], double values[], double expectedValues[], double thresholds[]) {
	double sumOfSquaredErrors = 0.0;

	for (size_t o = 1 + NUMINPUTNODES + NUMHIDDENNODES; o < 1 + NUMNODES; o++) {
		double absoulteError = expectedValues[o] - values[o];
		sumOfSquaredErrors += pow(absoulteError, 2);
		double outputErrorGradient = values[o] * (1.0 - values[o])* absoulteError;

		//update each weigting from the hidden layer
		for (size_t h = 1 + NUMINPUTNODES; h < 1 + NUMINPUTNODES + NUMHIDDENNODES; h++) {
			double delta = LEARNINGRATE * values[h] * outputErrorGradient;
			weights[h][o] += delta;
			double hiddenErrorGradient = values[h] * (1.0 - values[h]) * outputErrorGradient * weights[h][o];

			for (size_t i = 1; i < 1 + NUMINPUTNODES; i++) {
				double delta = LEARNINGRATE * values[i] * hiddenErrorGradient;
				weights[i][h] += delta;
			}
			double thresholDelta = LEARNINGRATE * -1.0 * hiddenErrorGradient;
			thresholds[h] += thresholDelta;
		}
		double delta = LEARNINGRATE * -1.0 * outputErrorGradient;
		thresholds[o] += delta;
	}
	return sumOfSquaredErrors;
}

void displayNetwork(double values[], double sumOfSquaredErrors) {
	static int counter = 0;
	if ((counter % 4) == 0) cout << setfill('-')<< setw(40) << "-" << endl;
	cout << setfill(' ')<< fixed << setw(8) << setprecision(4)  << values[1] \
		<< "|" << setprecision(4) << values[2] << "|" << setprecision(4) \
		<< values[5] << "|" << " error: " <<setprecision(5)	<< sumOfSquaredErrors << endl;
	counter++;
}
