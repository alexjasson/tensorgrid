#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensorgrid.h"

#define NUM_CLASSES  10
#define IMAGE_SIZE   784
#define HIDDEN_SIZE  256
#define BATCH_SIZE   64
#define EPOCHS       10
#define LOG          100
#define LR           0.01f
#define MOMENTUM     0.9f

int main(void)
{
    // Read data into Tensors from .npy files
    Tensor xTrain = TensorRead("data/mnist/xTrain.npy"); // [60000, 28, 28]
    Tensor yTrain = TensorRead("data/mnist/yTrain.npy"); // [60000]
    Tensor xTest  = TensorRead("data/mnist/xTest.npy");  // [10000, 28, 28]
    Tensor yTest  = TensorRead("data/mnist/yTest.npy");  // [10000]

    // Compute number of training/test samples and number of batches
    int nTrain = TensorShape(xTrain, 0);
    int nTest  = TensorShape(xTest, 0);
    int trainBatches = nTrain / BATCH_SIZE;
    int testBatches  = nTest  / BATCH_SIZE;
    printf("Train: %d  Test: %d\n\n", nTrain, nTest);

    // Reshape data Tensors from [..., 28, 28] to [..., 784] and scale entries from [0, 255] to [0, 1]
    xTrain = TensorReshape(xTrain, 2, (int[]){nTrain, IMAGE_SIZE}); // [60000, 784]
    xTest  = TensorReshape(xTest,  2, (int[]){nTest,  IMAGE_SIZE}); // [10000, 784]
    TensorScale(xTrain, 1.0f / 255.0f);
    TensorScale(xTest,  1.0f / 255.0f);

    // Create and initialize weight matrices of model
    Tensor W1 = TensorNew(2, (int[]){IMAGE_SIZE,  HIDDEN_SIZE}, 1); // [784, 256]
    Tensor W2 = TensorNew(2, (int[]){HIDDEN_SIZE, NUM_CLASSES}, 1); // [256, 10]
    TensorXavier(W1);
    TensorXavier(W2);

    // One-hot encoding of yTrain needed to compute cross entropy loss
    Tensor oneHot = TensorOneHot(yTrain, NUM_CLASSES); // [60000, 10]

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        // Shuffle training data
        unsigned int seed = (unsigned int)time(NULL);
        Tensor xShuffled = TensorShuffle(xTrain, seed); // [60000, 784]
        Tensor yShuffled = TensorShuffle(oneHot, seed); // [60000, 10]

        for (int b = 1; b <= trainBatches; b++) {
            Tensor X = TensorSlice(xShuffled, 0, (b - 1) * BATCH_SIZE, BATCH_SIZE); // [64, 784]
            Tensor Y = TensorSlice(yShuffled, 0, (b - 1) * BATCH_SIZE, BATCH_SIZE); // [64, 10]

            Tensor Z1   = TensorMultiply(X, W1);          // [64, 256]
            Tensor A1   = TensorMap(RELU, Z1);            // [64, 256]
            Tensor Z2   = TensorMultiply(A1, W2);         // [64, 10]
            Tensor P    = TensorSoftmax(Z2, 1.0f);        // [64, 10]
            Tensor CE   = TensorCrossEntropyLoss(Y, P);   // [64, 1]
            
            TensorBackward(CE);
            TensorSGD(W1, LR, MOMENTUM);
            TensorSGD(W2, LR, MOMENTUM);
            
            if ((b % LOG == 0) || (b == trainBatches)) {
                printf("Epoch %2d | Batch %3d/%d | Loss: ", epoch, b, trainBatches);
                Tensor loss = TensorReduce(ADD, 0.0f, CE, 0); // [1, 1]
                TensorScale(loss, 1.0f / BATCH_SIZE);
                TensorPrint(loss);
                TensorFree(loss);
            }

            TensorFree(Z1); TensorFree(A1); TensorFree(Z2);
            TensorFree(P);  TensorFree(CE);
        }
        TensorFree(xShuffled); TensorFree(yShuffled);

        int totalSamples = 0, totalErrors = 0;
        for (int b = 1; b <= testBatches; b++) {
            Tensor X = TensorSlice(xTest, 0, (b - 1) * BATCH_SIZE, BATCH_SIZE); // [64, 784]
            Tensor Y = TensorSlice(yTest, 0, (b - 1) * BATCH_SIZE, BATCH_SIZE); // [64]

            Tensor Z1          = TensorMultiply(X, W1);   // [64, 256]
            Tensor A1          = TensorMap(RELU, Z1);     // [64, 256]
            Tensor Z2          = TensorMultiply(A1, W2);  // [64, 10]
            Tensor predictions = TensorSample(Z2, 0);     // [64]

            totalErrors  += TensorHammingDistance(predictions, Y);
            totalSamples += TensorSize(predictions);

            TensorFree(Z1); TensorFree(A1); TensorFree(Z2);
            TensorFree(predictions);
        }
        printf("Epoch %2d | Test Accuracy: %.2f%%\n", epoch, 100.0f * (totalSamples - totalErrors) / totalSamples);
    }

    TensorFree(W1); TensorFree(W2);
    TensorFree(oneHot);
    TensorFree(xTrain); TensorFree(yTrain);
    TensorFree(xTest);  TensorFree(yTest);
    return 0;
}