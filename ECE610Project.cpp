/*

ECE610 Project - Queue Simulation

Filza Mazahir 
20295951
fmazahir@uwaterloo.ca

Compile ->  g++ -o project.out ECE610Project.cpp
Run -> ./project.out

*/


#include <iostream>
#include <math.h>
using namespace std; //that means don't need to do std::cout


// Generate Random Values between 0 and 1 using Uniform Distribution
double getRandomValueUniformDist() {  
    double uniformRandomValue = (double) rand() / (RAND_MAX);
    return uniformRandomValue;
}


// Generate Random Values with Exponential Distribution given a parameter mu
double getRandomValueExpDist(double parameter) {
    double uniformRV = getRandomValueUniformDist();
    double expRandomValue = -(log(uniformRV))/parameter;  // using Cumulative Distributive Function
    return expRandomValue;

}

// Mean of an array
double getMeanOfArray(double *arr, int size){
    double sum=0;
    int i=0;
    while (i++ < size) {
        sum += arr[i];
    }
    double mean = sum/size;
    return mean;
}

// Variance of an array
double getVarianceOfArray(double *arr, int size){
    double var=0;
    int i = 0;
    double mean = getMeanOfArray(arr, size);
    while (i++ < size) {
        var += (arr[i] - mean) * (arr[i] - mean);
    
    }
    var = var/size;
    return var;
}

// Testing Uniform Random Number Generator
void testUniformRandomValueGenerator(){
    srand(132452);
    int numValues = 1000; // Number of random values to generate 

    // Generate list of 1000 uniformly distributed random values
    double uniformRandomValues[numValues];  
    int i = 0;
    while (i++ < numValues) {
        uniformRandomValues[i] = getRandomValueUniformDist();  
    }

    double mean = getMeanOfArray(uniformRandomValues, numValues);
    double var = getVarianceOfArray(uniformRandomValues, numValues);

    cout << "Testing Random Value Generator" << endl;
    cout << "Mean of the 1000 uniform random values: " << mean << endl;  // Expected value 10
    cout << "Variance of the 1000 uniform random values: " << var << endl << endl;  // Expected value 100 (1/mu * 1/mu)
    
    return;
}

// Question 1
void generate1000ExpRandomValues (){
    srand(132452);
    int numValues = 1000; // Number of random values to generate 
    double parameter = 0.1;  // 1/mu = 10 seconds

    // Generate list of 1000 exponentially distributed random values
    double expRandomValues[numValues]; 
    int i = 0;
    while (i++ < numValues) {
        expRandomValues[i] = getRandomValueExpDist(parameter);  
    }

    double mean = getMeanOfArray(expRandomValues, numValues);
    double var = getVarianceOfArray(expRandomValues, numValues);

    cout << "Question 1" << endl;
    cout << "Mean of the 1000 exponential random values: " << mean << endl;  // Expected value 10
    cout << "Variance of the 1000 exponential random values: " << var << endl << endl;  // Expected value 100 (1/mu * 1/mu)
    
    return;
}

// Set up Discrete Event Simulation

// Event Class
class Event {
    public:
        char type; // Type of event - O -> observation, A -> arrival, D -> departure
        double t;  //Time of the event
        Event(char eventType, double eventTime) {
            type = eventType;
            t = eventTime;
        }
};
// // Event Class Constructor
// Event::Event(char eventType, double eventTime) {
    
// }

// double generateExpRandomValueLessT(){

// }

void setupDiscreteEventSimulation(){
    int T = 100;
    Event eventsList[10];
    int i=0;
    double randomExpValue;
    double parameter = 0.1; // 1/mu = 10 secones

    // Generate Random Observation Times
    double observationTime = 0;

    while (observationTime < T) {
        randomExpValue = getRandomValueExpDist(parameter);
        observationTime += randomExpValue;

        Event observationEvent('O', observationTime);

        eventsList[i] = observationEvent;
        i++;
    }


}


    // DISCRETE EVENT SIMULATION

    // take the first exp random value, thats the first time. then second one is second+first, third is third+second+first
    //  keep doing until the values (sum) are less than T
    // store in array -> observation time

    // do these twice again- once for packet arrival, one for packet length
    // calculate departure times = arrival + wait+ L/C


    // class of EVENT
    // members- Type (A, O, D), Time (shouldn't need packet)
    // add all previous arrays into a ES list of Event objects - sorted by time

    //read events one at a time from the ES list, then depending on type, run a specific function to update variables


// MAIN FUNCTION
int main() {

    // Test that the uniform random value generator is good
    testUniformRandomValueGenerator();

    //Question 1
    generate1000ExpRandomValues();

    //srand(time(NULL));  
    srand(132452);


    cout << endl;
    return 0;
}






