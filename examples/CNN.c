#include <stdio.h>
#include <stdlib.h>
#include "tensorgrid.h"

#define NUM_CLASSES 10
#define BATCH_SIZE  64
#define C1_CHANNELS 16
#define C2_CHANNELS 32
#define FEATURES    7*7*C2_CHANNELS
#define HIDDEN_SIZE 128
#define EPOCHS      10
#define LOG         100
#define LR          0.01f
#define MOMENTUM    0.9f

/*
 * CNN with two convolution layers with max-pooling, one
 * hidden layer and an output layer. Let N = batch size
 */
typedef struct {
    Tensor K1, b1, K2, b2; // Convolution filter weights and biases
    Tensor W1, b3, W2, b4; // Hidden/output layer weights and biases
} Model;

// X: [N, 1, 28, 28]
static Tensor forward(Tensor X, Model *m)
{
    Tensor XK1  = TensorConvolution(X,  m->K1, 1, 1);                  // [N, 16, 28, 28]
    Tensor C1   = TensorZip(ADD, XK1, m->b1);                          // [N, 16, 28, 28]
    Tensor A1   = TensorMap(RELU, C1);                                 // [N, 16, 28, 28]
    Tensor P1   = TensorPool(MAX, -INFINITE, A1, 2, 2, 0);             // [N, 16, 14, 14]
    Tensor P1K2 = TensorConvolution(P1, m->K2, 1, 1);                  // [N, 32, 14, 14]
    Tensor C2   = TensorZip(ADD, P1K2, m->b2);                         // [N, 32, 14, 14]
    Tensor A2   = TensorMap(RELU, C2);                                 // [N, 32, 14, 14]
    Tensor P2   = TensorPool(MAX, -INFINITE, A2, 2, 2, 0);             // [N, 32, 7, 7]

    Tensor F    = TensorReshape(P2, 2, (int[]){BATCH_SIZE, FEATURES}); // [N, FEATURES]
    Tensor FW1  = TensorMultiply(F, m->W1);                            // [N, 128]
    Tensor Z1   = TensorZip(ADD, FW1, m->b3);                          // [N, 128]
    Tensor H1   = TensorMap(RELU, Z1);                                 // [N, 128]
    Tensor H1W2 = TensorMultiply(H1, m->W2);                           // [N, 10]
    Tensor Z2   = TensorZip(ADD, H1W2, m->b4);                         // [N, 10]

    TensorFree(XK1); TensorFree(P1K2); TensorFree(H1W2);
    TensorFree(C1); TensorFree(A1); TensorFree(P1);
    TensorFree(C2); TensorFree(A2); TensorFree(P2);
    TensorFree(FW1); TensorFree(Z1); TensorFree(H1);
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
    int trainBatches = nTrain / BATCH_SIZE;
    int testBatches  = nTest  / BATCH_SIZE;
    printf("Train: %d  Test: %d\n\n", nTrain, nTest);

    // Reshape data Tensors from [..., 28, 28] to [..., 1, 28, 28] and scale entries from [0, 255] to [0, 1]
    xTrain = TensorReshape(xTrain, 4, (int[]){nTrain, 1, 28, 28}); // [60000, 1, 28, 28]
    xTest  = TensorReshape(xTest,  4, (int[]){nTest,  1, 28, 28}); // [10000, 1, 28, 28]
    TensorScale(xTrain, 1.0f / 255.0f);
    TensorScale(xTest,  1.0f / 255.0f);

    // Create and initialize convolution filters, weights and biases
    Model m;
    m.K1 = TensorNew(4, (int[]){C1_CHANNELS,  1, 3, 3}, TRUE);          // [16, 1, 3, 3]
    m.b1 = TensorNew(4, (int[]){1, C1_CHANNELS, 1, 1}, TRUE);           // [1, 16, 1, 1]
    m.K2 = TensorNew(4, (int[]){C2_CHANNELS, C1_CHANNELS, 3, 3}, TRUE); // [32, 16, 3, 3]
    m.b2 = TensorNew(4, (int[]){1, C2_CHANNELS, 1, 1}, TRUE);           // [1, 32, 1, 1]
    m.W1 = TensorNew(2, (int[]){FEATURES, HIDDEN_SIZE}, TRUE);          // [32*7*7, 128]
    m.b3 = TensorNew(2, (int[]){1, HIDDEN_SIZE}, TRUE);                 // [1, 128]
    m.W2 = TensorNew(2, (int[]){HIDDEN_SIZE, NUM_CLASSES}, TRUE);       // [128, 10]
    m.b4 = TensorNew(2, (int[]){1, NUM_CLASSES}, TRUE);                 // [1, 10]
    TensorXavier(m.K1); TensorXavier(m.K2);
    TensorXavier(m.W1); TensorXavier(m.W2);

    // One-hot encoding of yTrain needed to compute cross entropy loss
    Tensor oneHot = TensorOneHot(yTrain, NUM_CLASSES); // [60000, 10]

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        // Shuffle training data using epoch as seed
        Tensor xShuffled = TensorShuffle(xTrain, epoch); // [60000, 1, 28, 28]
        Tensor yShuffled = TensorShuffle(oneHot, epoch); // [60000, 10]

        for (int b = 0; b < trainBatches; b++) {
            Tensor X = TensorSlice(xShuffled, 0, b * BATCH_SIZE, BATCH_SIZE); // [64, 1, 28, 28]
            Tensor Y = TensorSlice(yShuffled, 0, b * BATCH_SIZE, BATCH_SIZE); // [64, 10]

            // Computes cross-entropy loss, NLL(y, LogSoftmax(z)) ≡ CrossEntropy(y, softmax(z))
            Tensor Z    = forward(X, &m);             // [64, 10]
            Tensor logP = TensorLogSoftmax(Z, 1.0f);  // [64, 10]
            Tensor loss = TensorNLL(Y, logP);         // [] (Scalar)

            TensorBackward(loss);
            TensorSGD(m.K1, LR, MOMENTUM);
            TensorSGD(m.b1, LR, MOMENTUM);
            TensorSGD(m.K2, LR, MOMENTUM);
            TensorSGD(m.b2, LR, MOMENTUM);
            TensorSGD(m.W1, LR, MOMENTUM);
            TensorSGD(m.b3, LR, MOMENTUM);
            TensorSGD(m.W2, LR, MOMENTUM);
            TensorSGD(m.b4, LR, MOMENTUM);

            int batchNumber = b + 1;
            if ((batchNumber % LOG == 0) || (batchNumber == trainBatches)) {
                printf("Epoch %2d | Batch %3d/%d | Loss: ", epoch, batchNumber, trainBatches);
                TensorPrint(loss);
            }

            TensorFree(Z); TensorFree(logP); TensorFree(loss);
        }
        TensorFree(xShuffled); TensorFree(yShuffled);

        int totalSamples = 0, totalErrors = 0;
        for (int b = 0; b < testBatches; b++) {
            Tensor X = TensorSlice(xTest, 0, b * BATCH_SIZE, BATCH_SIZE); // [64, 1, 28, 28]
            Tensor Y = TensorSlice(yTest, 0, b * BATCH_SIZE, BATCH_SIZE); // [64]

            Tensor Z           = forward(X, &m);     // [64, 10]
            Tensor predictions = TensorSample(Z, 0); // [64]

            totalErrors  += TensorHammingDistance(predictions, Y);
            totalSamples += TensorSize(predictions);

            TensorFree(Z); TensorFree(predictions);
        }
        printf("Epoch %2d | Test Accuracy: %.2f%%\n", epoch, 100.0f * (totalSamples - totalErrors) / totalSamples);
    }

    TensorFree(m.K1); TensorFree(m.b1); TensorFree(m.K2); TensorFree(m.b2);
    TensorFree(m.W1); TensorFree(m.b3); TensorFree(m.W2); TensorFree(m.b4);
    TensorFree(oneHot);
    TensorFree(xTrain); TensorFree(yTrain);
    TensorFree(xTest);  TensorFree(yTest);
    return 0;
}
