#include <stdio.h>
#include <stdlib.h>
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
 * MLP with a single hidden layer. Let N = batch size
 */
typedef struct {
    Tensor W1, b1, W2, b2; // Hidden/output layer weights and biases
} Model;

// X: [N, 784]
static Tensor forward(Tensor X, Model *m)
{
    Tensor XW1  = TensorMultiply(X,  m->W1);   // [N, 128]
    Tensor Z1   = TensorZip(ADD, XW1, m->b1);  // [N, 128]
    Tensor A1   = TensorMap(RELU, Z1);         // [N, 128]
    Tensor A1W2 = TensorMultiply(A1, m->W2);   // [N, 10]
    Tensor Z2   = TensorZip(ADD, A1W2, m->b2); // [N, 10]

    TensorFree(XW1); TensorFree(A1W2);
    TensorFree(Z1); TensorFree(A1);
    return Z2;
}

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
    int trainBatches = (nTrain + BATCH_SIZE - 1) / BATCH_SIZE;
    int testBatches  = (nTest  + BATCH_SIZE - 1) / BATCH_SIZE;
    printf("Train: %d  Test: %d\n\n", nTrain, nTest);

    // Reshape data Tensors from [..., 28, 28] to [..., 784] and scale entries from [0, 255] to [0, 1]
    xTrain = TensorReshape(xTrain, 2, (int[]){nTrain, IMAGE_SIZE}); // [60000, 784]
    xTest  = TensorReshape(xTest,  2, (int[]){nTest,  IMAGE_SIZE}); // [10000, 784]
    TensorScale(xTrain, 1.0f / 255.0f);
    TensorScale(xTest,  1.0f / 255.0f);

    // Create and initialize hidden and output layer weights/biases
    Model m;
    m.W1 = TensorNew(2, (int[]){IMAGE_SIZE,  HIDDEN_SIZE}, TRUE); // [784, 128]
    m.b1 = TensorNew(2, (int[]){1, HIDDEN_SIZE}, TRUE);           // [1, 128]
    m.W2 = TensorNew(2, (int[]){HIDDEN_SIZE, NUM_CLASSES}, TRUE); // [128, 10]
    m.b2 = TensorNew(2, (int[]){1, NUM_CLASSES}, TRUE);           // [1, 10]
    TensorXavier(m.W1); TensorXavier(m.W2);

    // One-hot encoding of yTrain needed to compute cross entropy loss
    Tensor oneHot = TensorOneHot(yTrain, NUM_CLASSES); // [60000, 10]

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        // Shuffle training data using epoch as seed
        Tensor xShuffled = TensorShuffle(xTrain, epoch); // [60000, 784]
        Tensor yShuffled = TensorShuffle(oneHot, epoch); // [60000, 10]

        for (int batch = 0; batch < trainBatches; batch++) {
            // Cap final batch size so we don't slice past the end of dataset
            int N = nTrain - batch * BATCH_SIZE;
            if (N > BATCH_SIZE) N = BATCH_SIZE;
            Tensor X = TensorSlice(xShuffled, 0, batch * BATCH_SIZE, N); // [N, 784]
            Tensor Y = TensorSlice(yShuffled, 0, batch * BATCH_SIZE, N); // [N, 10]

            // Cross entropy loss implemented as NLL(y, logSoftmax(z))
            Tensor Z    = forward(X, &m);            // [N, 10]
            Tensor logP = TensorLogSoftmax(Z, 1.0f); // [N, 10]
            Tensor loss = TensorNLL(Y, logP);        // [] (Scalar)

            TensorBackward(loss);
            TensorSGD(m.W1, LR, MOMENTUM);
            TensorSGD(m.b1, LR, MOMENTUM);
            TensorSGD(m.W2, LR, MOMENTUM);
            TensorSGD(m.b2, LR, MOMENTUM);
            
            if (((batch + 1) % LOG == 0) || (batch + 1 == trainBatches)) {
                printf("Epoch %2d | Batch %3d/%d | Loss: ", epoch, batch + 1, trainBatches);
                TensorPrint(loss);
            }

            TensorFree(X); TensorFree(Y);
            TensorFree(Z); TensorFree(logP); TensorFree(loss);
        }
        TensorFree(xShuffled); TensorFree(yShuffled);

        int totalSamples = 0, totalErrors = 0;
        for (int batch = 0; batch < testBatches; batch++) {
            int N = nTest - batch * BATCH_SIZE;
            if (N > BATCH_SIZE) N = BATCH_SIZE;
            Tensor X = TensorSlice(xTest, 0, batch * BATCH_SIZE, N); // [N, 784]
            Tensor Y = TensorSlice(yTest, 0, batch * BATCH_SIZE, N); // [N]

            Tensor Z           = forward(X, &m);     // [N, 10]
            Tensor predictions = TensorSample(Z, 0); // [N]

            totalErrors  += TensorHammingDistance(predictions, Y);
            totalSamples += TensorSize(predictions);

            TensorFree(X); TensorFree(Y);
            TensorFree(Z); TensorFree(predictions);
        }
        printf("Epoch %2d | Test Accuracy: %.2f%%\n", epoch, 100.0f * (totalSamples - totalErrors) / totalSamples);
    }

    TensorFree(m.W1); TensorFree(m.b1); TensorFree(m.W2); TensorFree(m.b2);
    TensorFree(oneHot);
    TensorFree(xTrain); TensorFree(yTrain);
    TensorFree(xTest);  TensorFree(yTest);
    return 0;
}
