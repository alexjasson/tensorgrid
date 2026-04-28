#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensorgrid.h"

#define NUM_CLASSES  10
#define BATCH_SIZE   64
#define C1_CHANNELS  16
#define C2_CHANNELS  32
#define FEATURES     14*14*C2_CHANNELS
#define EPOCHS       10
#define LOG          100
#define LR           0.01f
#define MOMENTUM     0.9f

/*
 * CNN with two convolution layers and no pooling. Let N = batch size
 * [N, 1, 28, 28]
 * -> Conv [16, 1, 3, 3], stride 1, padding 1
 * -> ReLU
 * -> [N, 16, 28, 28]
 * -> Conv [32, 16, 3, 3], stride 2, padding 1
 * -> ReLU
 * -> [N, 32, 14, 14]
 * -> Reshape -> [N, 32*14*14]
 * -> Linear  [32*14*14, 10]
 * -> [N, 10]
 */
typedef struct {
    Tensor F1, F2; // Convolution filters
    Tensor W;      // Output layer weights
} Model;

static Tensor forward(Tensor X, Model *m)
{
    Tensor C1 = TensorConvolution(X,  m->F1, 1, 1);                    // [N, 16, 28, 28]
    Tensor A1 = TensorMap(RELU, C1);                                   // [N, 16, 28, 28]
    Tensor C2 = TensorConvolution(A1, m->F2, 2, 1);                    // [N, 32, 14, 14]
    Tensor A2 = TensorMap(RELU, C2);                                   // [N, 32, 14, 14]

    Tensor flat = TensorReshape(A2, 2, (int[]){BATCH_SIZE, FEATURES}); // [N, FEATURES]
    Tensor Z    = TensorMultiply(flat, m->W);                          // [N, 10]

    TensorFree(C1); TensorFree(A1);
    TensorFree(C2); TensorFree(A2);
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
    int nTrain = TensorShape(xTrain, 0);
    int nTest  = TensorShape(xTest, 0);
    int trainBatches = nTrain / BATCH_SIZE;
    int testBatches  = nTest  / BATCH_SIZE;
    printf("Train: %d  Test: %d\n\n", nTrain, nTest);

    // Reshape data Tensors from [..., 28, 28] to [..., 1, 28, 28] and scale entries from [0, 255] to [0, 1]
    xTrain = TensorReshape(xTrain, 4, (int[]){nTrain, 1, 28, 28}); // [60000, 1, 28, 28]
    xTest  = TensorReshape(xTest,  4, (int[]){nTest,  1, 28, 28}); // [10000, 1, 28, 28]
    TensorScale(xTrain, 1.0f / 255.0f);
    TensorScale(xTest,  1.0f / 255.0f);

    // Create and initialize convolution filters and output layer weights
    Model m;
    m.F1 = TensorNew(4, (int[]){C1_CHANNELS,  1, 3, 3}, 1);          // [16, 1, 3, 3]
    m.F2 = TensorNew(4, (int[]){C2_CHANNELS, C1_CHANNELS, 3, 3}, 1); // [32, 16, 3, 3]
    m.W  = TensorNew(2, (int[]){FEATURES, NUM_CLASSES}, 1);          // [32*14*14, 10]
    TensorXavier(m.F1); TensorXavier(m.F2); TensorXavier(m.W);

    // One-hot encoding of yTrain needed to compute cross entropy loss
    Tensor oneHot = TensorOneHot(yTrain, NUM_CLASSES); // [60000, 10]

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        // Shuffle training data
        unsigned int seed = (unsigned int)time(NULL);
        Tensor xShuffled = TensorShuffle(xTrain, seed); // [60000, 1, 28, 28]
        Tensor yShuffled = TensorShuffle(oneHot, seed); // [60000, 10]

        for (int b = 1; b <= trainBatches; b++) {
            Tensor X = TensorSlice(xShuffled, 0, (b - 1) * BATCH_SIZE, BATCH_SIZE); // [64, 1, 28, 28]
            Tensor Y = TensorSlice(yShuffled, 0, (b - 1) * BATCH_SIZE, BATCH_SIZE); // [64, 10]

            // Computes cross-entropy loss, NLL(y, LogSoftmax(z)) ≡ CrossEntropy(y, softmax(z))
            Tensor Z    = forward(X, &m);             // [64, 10]
            Tensor logP = TensorLogSoftmax(Z, 1.0f);  // [64, 10]
            Tensor loss = TensorNLL(Y, logP);         // scalar

            TensorBackward(loss);
            TensorSGD(m.F1, LR, MOMENTUM);
            TensorSGD(m.F2, LR, MOMENTUM);
            TensorSGD(m.W,  LR, MOMENTUM);

            if ((b % LOG == 0) || (b == trainBatches)) {
                printf("Epoch %2d | Batch %3d/%d | Loss: ", epoch, b, trainBatches);
                TensorPrint(loss);
            }

            TensorFree(Z); TensorFree(logP); TensorFree(loss);
        }
        TensorFree(xShuffled); TensorFree(yShuffled);

        int totalSamples = 0, totalErrors = 0;
        for (int b = 1; b <= testBatches; b++) {
            Tensor X = TensorSlice(xTest, 0, (b - 1) * BATCH_SIZE, BATCH_SIZE); // [64, 1, 28, 28]
            Tensor Y = TensorSlice(yTest, 0, (b - 1) * BATCH_SIZE, BATCH_SIZE); // [64]

            Tensor Z           = forward(X, &m);     // [64, 10]
            Tensor predictions = TensorSample(Z, 0); // [64]

            totalErrors  += TensorHammingDistance(predictions, Y);
            totalSamples += TensorSize(predictions);

            TensorFree(Z); TensorFree(predictions);
        }
        printf("Epoch %2d | Test Accuracy: %.2f%%\n", epoch, 100.0f * (totalSamples - totalErrors) / totalSamples);
    }

    TensorFree(m.F1); TensorFree(m.F2); TensorFree(m.W);
    TensorFree(oneHot);
    TensorFree(xTrain); TensorFree(yTrain);
    TensorFree(xTest);  TensorFree(yTest);
    return 0;
}
