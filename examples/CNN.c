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
    int N     = TensorShape(X, 0);
    Tensor A1 = TensorConvolutionLayer(X,  m->K1, m->b1, 1, 1, RELU); // [N, 16, 28, 28]
    Tensor P1 = TensorMaxPool(A1, 2, 2, 0);                           // [N, 16, 14, 14]
    Tensor A2 = TensorConvolutionLayer(P1, m->K2, m->b2, 1, 1, RELU); // [N, 32, 14, 14]
    Tensor P2 = TensorMaxPool(A2, 2, 2, 0);                           // [N, 32, 7, 7]
    Tensor F  = TensorReshape(P2, 2, (int[]){N, FEATURES});           // [N, FEATURES]
    Tensor H  = TensorLinearLayer(F, m->W1, m->b3, RELU);             // [N, 128]
    Tensor Z  = TensorLinearLayer(H, m->W2, m->b4, IDENTITY);         // [N, 10]

    TensorFree(A1); TensorFree(P1);
    TensorFree(A2); TensorFree(P2);
    TensorFree(H);
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
    int trainBatches = (nTrain + BATCH_SIZE - 1) / BATCH_SIZE;
    int testBatches  = (nTest  + BATCH_SIZE - 1) / BATCH_SIZE;
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

        for (int batch = 0; batch < trainBatches; batch++) {
            // Cap final batch size so we don't slice past the end of dataset
            int N = nTrain - batch * BATCH_SIZE;
            if (N > BATCH_SIZE) N = BATCH_SIZE;
            Tensor X = TensorSlice(xShuffled, 0, batch * BATCH_SIZE, N); // [N, 1, 28, 28]
            Tensor Y = TensorSlice(yShuffled, 0, batch * BATCH_SIZE, N); // [N, 10]

            Tensor Z    = forward(X, &m);                     // [N, 10]
            Tensor loss = TensorCrossEntropyLoss(Y, Z, 1.0f); // [] (Scalar)

            TensorBackward(loss);
            TensorSGD(m.K1, LR, MOMENTUM);
            TensorSGD(m.b1, LR, MOMENTUM);
            TensorSGD(m.K2, LR, MOMENTUM);
            TensorSGD(m.b2, LR, MOMENTUM);
            TensorSGD(m.W1, LR, MOMENTUM);
            TensorSGD(m.b3, LR, MOMENTUM);
            TensorSGD(m.W2, LR, MOMENTUM);
            TensorSGD(m.b4, LR, MOMENTUM);

            if (((batch + 1) % LOG == 0) || (batch + 1 == trainBatches)) {
                printf("Epoch %2d | Batch %3d/%d | Loss: ", epoch, batch + 1, trainBatches);
                TensorPrint(loss);
            }

            TensorFree(Z); TensorFree(loss);
        }
        TensorFree(xShuffled); TensorFree(yShuffled);

        int totalSamples = 0, totalErrors = 0;
        for (int batch = 0; batch < testBatches; batch++) {
            int N = nTest - batch * BATCH_SIZE;
            if (N > BATCH_SIZE) N = BATCH_SIZE;
            Tensor X = TensorSlice(xTest, 0, batch * BATCH_SIZE, N); // [N, 1, 28, 28]
            Tensor Y = TensorSlice(yTest, 0, batch * BATCH_SIZE, N); // [N]

            Tensor Z           = forward(X, &m);     // [N, 10]
            Tensor predictions = TensorSample(Z, 0); // [N]

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
