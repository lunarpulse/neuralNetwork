#include <stdio.h>
#include <stdlib.h>
#include <cmath>

//constants
const int NUMINPUTNODES = 2;
const int NUMHIDDENNODES = 2;
const int NOMOUTPUTNODES = 1;
const int NUMNODES = NOMOUTPUTNODES + NUMHIDDENNODES + NUMINPUTNODES;

const int ARRAYSIZE = NUMNODES + 1; //1 - offset to match node 1 node 2 etc
const int MAXITERATIONS = 13107;//131072;

const double E = 2.71828;
const double LEARNINGRATE = 0.1;

//function prototypes
void initialise(double[][ARRAYSIZE], double[], double[], double[]);
void connectNode(double[][ARRAYSIZE], double[]);
void trainingExample(double[], double[]);
void activateNetwork(double[][ARRAYSIZE], double[], double[]);
double updateWeights(double[][ARRAYSIZE], double[], double[], double[]);
void displayNetwork(double[], double);

int main() {
	printf("Neural Network Program\n");

	double weights[ARRAYSIZE][ARRAYSIZE];
	double values[ARRAYSIZE];
	double expectedValues[ARRAYSIZE];
	double thresholds[ARRAYSIZE];

	initialise(weights, values, expectedValues, thresholds);
	connectNode(weights, thresholds);
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

void connectNode(double weights[][ARRAYSIZE], double thresholds[]) {
	for (size_t x = 0; x < NUMNODES; x++) {
		for (size_t y = 0; y < NUMNODES; y++) {
			weights[x][y] = (rand() % 200) / 100.0;
		}
	}

	thresholds[3] = rand() / (double)rand();
	thresholds[4] = rand() / (double)rand();
	thresholds[5] = rand() / (double)rand();

	printf("%f%f%f%f%f%f\n%f%f%f\n", weights[1][3], weights[1][4], weights[2][3], weights[2][4], weights[3][5], weights[4][5], thresholds[3], thresholds[4], thresholds[5]);

}

void trainingExample(double values[], double expectedValues[]) {
	static int counter = 0;

	switch (counter % 4) {
	case 0:
		values[1] = 1;
		values[2] = 1;
		expectedValues[5] = 0;
		break;
	case 1:
		values[1] = 0;
		values[2] = 1;
		expectedValues[5] = 1;
		break;
	case 2:
		values[1] = 1;
		values[2] = 0;
		expectedValues[5] = 1;
		break;
	case 3:
		values[1] = 0;
		values[2] = 0;
		expectedValues[5] = 0;
		break;
	}
	counter++;
}

void activateNetwork(double weights[][ARRAYSIZE], double values[], double thresholds[]) {
	//double weightedInput = 0.0;
	for (size_t h = 1 + NUMINPUTNODES; h < 1 + NUMINPUTNODES + NUMHIDDENNODES; h++) {
		double weightedInput = 0.0;
		//add up the weihted input
		for (size_t i = 0; i < NUMINPUTNODES; i++) {
			weightedInput += weights[i][h] * values[i];
		}
		weightedInput += (-1 * thresholds[h]);

		values[h] = 1.0 / (1.0 + pow(E, weightedInput));
	}
	for (size_t out = 1 + NUMINPUTNODES; out < 1 + NUMNODES; out++) {
		double weightedInput = 0.0;
		for (size_t h = 0; h < 1 + NUMINPUTNODES + NUMHIDDENNODES; h++) {
			weightedInput += weights[h][out] * values[h];
		}
		//handle the thresholds
		weightedInput += (-1 * thresholds[out]);

		values[out] = 1.0 / (1.0 + pow(E, -weightedInput));
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
			double delta = LEARNINGRATE* values[h] * outputErrorGradient;
			weights[h][o] += delta;
			double hiddenErrorGradient = values[h] * (1 - values[h]) * outputErrorGradient * weights[h][o];

			for (size_t i = 0; i < 1 + NUMINPUTNODES; i++) {
				double delta = LEARNINGRATE * values[i] * hiddenErrorGradient;
				weights[i][h] += delta;
			}
			double thresholDelta = LEARNINGRATE * -1 * hiddenErrorGradient;
			thresholds[h] += thresholDelta;
		}
		double delta = LEARNINGRATE * -1 * outputErrorGradient;
		thresholds[o] += delta;
	}
	return sumOfSquaredErrors;
}

void displayNetwork(double values[], double sumOfSquaredErrors) {
	static int counter = 0;
	if ((counter % 4) == 0) printf("---------------------------------------------------\n");
	printf("%8.4f|", values[1]);
	printf("%8.4f|", values[2]);
	printf("%8.4f|", values[5]);
	printf(" error:%8.5f\n", sumOfSquaredErrors);
	counter++;
}
