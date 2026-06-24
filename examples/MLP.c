#include <stdio.h>
#include "tensorgrid.h"

#define NUM_CLASSES 10
#define IMAGE_SIZE  784
#define HIDDEN_SIZE 128
#define BATCH_SIZE  64
#define EPOCHS      10
#define LOG         100
#define LR          0.01f
#define MOMENTUM    0.9f

/*
 * MLP with a single hidden layer. Let N = batch size.
 *
 * h = relu(X @ W_h + b_h)
 * Z = h @ W_y + b_y
 */
typedef struct {
    Tensor W_h, b_h; // [IMAGE_SIZE, HIDDEN_SIZE],  [HIDDEN_SIZE]
    Tensor W_y, b_y; // [HIDDEN_SIZE, NUM_CLASSES], [NUM_CLASSES]
} Model;

// X: [N, IMAGE_SIZE]
static Tensor forward(Tensor X, Model *m)
{
    Tensor h = TensorLinearLayer(X, m->W_h, m->b_h, RELU);     // [N, HIDDEN_SIZE]
    Tensor Z = TensorLinearLayer(h, m->W_y, m->b_y, IDENTITY); // [N, NUM_CLASSES]

    TensorFree(h);
    return Z;
}

int main(void)
{
    // Read data into Tensors from .npy files
    Tensor xTrain = TensorRead("data/mnist/xTrain.npy"); // [60000, 28, 28]
    Tensor yTrain = TensorRead("data/mnist/yTrain.npy"); // [60000]
    Tensor xTest  = TensorRead("data/mnist/xTest.npy");  // [10000, 28, 28]
    Tensor yTest  = TensorRead("data/mnist/yTest.npy");  // [10000]

    // Compute number of training/test samples and number of batches
    int trainSamples = TensorShape(xTrain, 0);
    int testSamples  = TensorShape(xTest, 0);
    int trainBatches = (trainSamples + BATCH_SIZE - 1) / BATCH_SIZE;
    int testBatches  = (testSamples  + BATCH_SIZE - 1) / BATCH_SIZE;

    // Reshape data Tensors from [..., 28, 28] to [..., 784] and scale entries from [0, 255] to [0, 1]
    xTrain = TensorReshape(xTrain, 2, (int[]){trainSamples, IMAGE_SIZE}); // [60000, 784]
    xTest  = TensorReshape(xTest,  2, (int[]){testSamples,  IMAGE_SIZE}); // [10000, 784]
    TensorScale(xTrain, 1.0f / 255.0f);
    TensorScale(xTest,  1.0f / 255.0f);

    // Create and initialize hidden and output layer weights/biases
    TensorArray parameters = TensorArrayNew();
    Model m;
    m.W_h = TensorArrayAdd(parameters, 2, (int[]){IMAGE_SIZE,  HIDDEN_SIZE}); // [784, 128]
    m.b_h = TensorArrayAdd(parameters, 1, (int[]){HIDDEN_SIZE});              // [128]
    m.W_y = TensorArrayAdd(parameters, 2, (int[]){HIDDEN_SIZE, NUM_CLASSES}); // [128, 10]
    m.b_y = TensorArrayAdd(parameters, 1, (int[]){NUM_CLASSES});              // [10]
    TensorXavier(m.W_h); TensorXavier(m.W_y);
    
    printf("Train Samples: %d\n", trainSamples);
    printf("Test Samples:  %d\n", testSamples);
    printf("Parameters:    %d\n\n", TensorArraySize(parameters));

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        // Shuffle training data using epoch as seed
        Tensor xShuffled = TensorShuffle(xTrain, epoch); // [60000, 784]
        Tensor yShuffled = TensorShuffle(yTrain, epoch); // [60000]

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
                printf("Epoch %2d | Batch %3d/%d | Loss: ", epoch, batch + 1, trainBatches);
                TensorPrint(loss);
            }

            TensorFree(Z); TensorFree(loss);
        }
        TensorFree(xShuffled); TensorFree(yShuffled);

        int totalSamples = 0, totalErrors = 0;
        for (int batch = 0; batch < testBatches; batch++) {
            int remaining = testSamples - batch * BATCH_SIZE;
            int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

            Tensor X = TensorSlice(xTest, batch * BATCH_SIZE, N); // [N, 784]
            Tensor Y = TensorSlice(yTest, batch * BATCH_SIZE, N); // [N]
            Tensor Z           = forward(X, &m);                  // [N, 10]
            Tensor predictions = TensorSample(Z, 0);              // [N]

            totalErrors  += TensorHammingDistance(predictions, Y);
            totalSamples += TensorSize(predictions);

            TensorFree(Z); TensorFree(predictions);
        }
        printf("Epoch %2d | Test Accuracy: %.2f%%\n", epoch, 100.0f * (totalSamples - totalErrors) / totalSamples);
    }

    TensorArrayFree(parameters);
    TensorFree(xTrain); TensorFree(yTrain);
    TensorFree(xTest);  TensorFree(yTest);
    return 0;
}
