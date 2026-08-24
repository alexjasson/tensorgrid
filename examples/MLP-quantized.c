#include <stdio.h>
#include "tensorgrid.h"

#define NUM_CLASSES 10
#define IMAGE_SIZE  784
#define HIDDEN_SIZE 128
#define BATCH_SIZE  64
#define LOG         100
#define LR          0.01f
#define MOMENTUM    0.9f

/*
 * MLP with a single hidden layer and INT8 quantized inference.
 *
 * h = relu(X @ W_h + b_h)
 * Z = h @ W_y + b_y
 */
typedef struct {
    Tensor W_h, b_h; // [IMAGE_SIZE, HIDDEN_SIZE],  [HIDDEN_SIZE]
    Tensor W_y, b_y; // [HIDDEN_SIZE, NUM_CLASSES], [NUM_CLASSES]
} Model;

static Tensor forward(Tensor X, Model *m)
{
    Tensor h = TensorLinearLayer(X, m->W_h, m->b_h, RELU);     // [N, HIDDEN_SIZE]
    Tensor Z = TensorLinearLayer(h, m->W_y, m->b_y, IDENTITY); // [N, NUM_CLASSES]

    TensorFree(h);
    return Z;
}

int main(void)
{
    // Read data into Tensors
    Tensor xTrain = TensorReadData("examples/data/mnist/train-images-idx3-ubyte", 2, (int[]){60000, IMAGE_SIZE}, 16, UINT8); // [60000, 784]
    Tensor yTrain = TensorReadData("examples/data/mnist/train-labels-idx1-ubyte", 1, (int[]){60000}, 8, UINT8);              // [60000]
    Tensor xTest  = TensorReadData("examples/data/mnist/t10k-images-idx3-ubyte", 2, (int[]){10000, IMAGE_SIZE}, 16, UINT8);  // [10000, 784]
    Tensor yTest  = TensorReadData("examples/data/mnist/t10k-labels-idx1-ubyte", 1, (int[]){10000}, 8, UINT8);               // [10000]

    // Compute number of training/test samples and number of batches
    int trainSamples = TensorShape(xTrain)[0];
    int testSamples  = TensorShape(xTest)[0];
    int trainBatches = (trainSamples + BATCH_SIZE - 1) / BATCH_SIZE;
    int testBatches  = (testSamples  + BATCH_SIZE - 1) / BATCH_SIZE;

    // Scale entries from [0, 255] to [0, 1]
    xTrain = TensorScalarMultiply(xTrain, 1.0f / 255);
    xTest  = TensorScalarMultiply(xTest,  1.0f / 255);

    // Create and initialize hidden and output layer weights/biases
    TensorArray parameters = TensorArrayNew();
    Model m;
    m.W_h = TensorArrayAdd(parameters, 2, (int[]){IMAGE_SIZE,  HIDDEN_SIZE}, FLOAT32, TRUE); // [784, 128]
    m.b_h = TensorArrayAdd(parameters, 1, (int[]){HIDDEN_SIZE}, FLOAT32, TRUE);              // [128]
    m.W_y = TensorArrayAdd(parameters, 2, (int[]){HIDDEN_SIZE, NUM_CLASSES}, FLOAT32, TRUE); // [128, 10]
    m.b_y = TensorArrayAdd(parameters, 1, (int[]){NUM_CLASSES}, FLOAT32, TRUE);              // [10]
    TensorXavier(m.W_h); TensorXavier(m.W_y);

    printf("Train Samples: %d\n", trainSamples);
    printf("Test Samples:  %d\n", testSamples);
    printf("Parameters:    %d\n\n", TensorArraySize(parameters));

    // Train for one epoch
    Tensor xShuffled = TensorShuffle(xTrain, 1); // [60000, 784]
    Tensor yShuffled = TensorShuffle(yTrain, 1); // [60000]
    for (int batch = 0; batch < trainBatches; batch++) {
        // Cap final batch size so we don't slice past the end of dataset
        int remaining = trainSamples - batch * BATCH_SIZE;
        int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

        Tensor X = TensorSlice(xShuffled, batch * BATCH_SIZE, N); // [N, 784]
        Tensor Y = TensorSlice(yShuffled, batch * BATCH_SIZE, N); // [N]
        Tensor Z    = forward(X, &m);                             // [N, 10]
        Tensor loss = TensorCrossEntropyLoss(Y, Z, 1.0f);         // [] (Scalar)

        TensorBackward(loss);
        TensorArraySGD(parameters, LR, MOMENTUM);

        if (((batch + 1) % LOG == 0) || (batch + 1 == trainBatches)) {
            printf("Batch %3d/%d | Loss: ", batch + 1, trainBatches);
            TensorPrint(loss);
        }

        TensorFree(X); TensorFree(Y);
        TensorFree(Z); TensorFree(loss);
    }
    TensorFree(xShuffled); TensorFree(yShuffled);

    // Compute test accuracy with the regular weights
    int totalErrors = 0;
    for (int batch = 0; batch < testBatches; batch++) {
        // Cap final batch size so we don't slice past the end of dataset
        int remaining = testSamples - batch * BATCH_SIZE;
        int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

        Tensor X = TensorSlice(xTest, batch * BATCH_SIZE, N); // [N, 784]
        Tensor Y = TensorSlice(yTest, batch * BATCH_SIZE, N); // [N]
        Tensor Z           = forward(X, &m);                  // [N, 10]
        Tensor predictions = TensorSample(Z, 0);              // [N]

        totalErrors += TensorHammingDistance(predictions, Y);
        TensorFree(X); TensorFree(Y);
        TensorFree(Z); TensorFree(predictions);
    }
    printf("\nRegular Test Accuracy:   %.2f%%\n", 100.0f * (testSamples - totalErrors) / testSamples);

    // Quantize the parameters to INT8 and compute test accuracy again
    TensorArrayToType(parameters, INT8);
    totalErrors = 0;
    for (int batch = 0; batch < testBatches; batch++) {
        // Cap final batch size so we don't slice past the end of dataset
        int remaining = testSamples - batch * BATCH_SIZE;
        int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

        Tensor X = TensorSlice(xTest, batch * BATCH_SIZE, N); // [N, 784]
        Tensor Y = TensorSlice(yTest, batch * BATCH_SIZE, N); // [N]
        Tensor Z           = forward(X, &m);                  // [N, 10]
        Tensor predictions = TensorSample(Z, 0);              // [N]

        totalErrors += TensorHammingDistance(predictions, Y);
        TensorFree(X); TensorFree(Y);
        TensorFree(Z); TensorFree(predictions);
    }
    printf("Quantized Test Accuracy: %.2f%%\n", 100.0f * (testSamples - totalErrors) / testSamples);

    TensorArrayFree(parameters);
    TensorFree(xTrain); TensorFree(yTrain);
    TensorFree(xTest);  TensorFree(yTest);
    return 0;
}
