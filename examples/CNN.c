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
 * CNN with two convolution layers with max-pooling, one hidden layer and
 * an output layer. Let N = batch size.
 *
 * p_1 = maxpool(relu(conv(X,   K_1) + b_1))
 * p_2 = maxpool(relu(conv(p_1, K_2) + b_2))
 * h   = relu(flatten(p_2) @ W_h + b_h)
 * Z   = h @ W_y + b_y
 */
typedef struct {
    Tensor K_1, b_1; // [C1_CHANNELS, 1, 3, 3],           [1, C1_CHANNELS, 1, 1]
    Tensor K_2, b_2; // [C2_CHANNELS, C1_CHANNELS, 3, 3], [1, C2_CHANNELS, 1, 1]
    Tensor W_h, b_h; // [FEATURES, HIDDEN_SIZE],          [1, HIDDEN_SIZE]
    Tensor W_y, b_y; // [HIDDEN_SIZE, NUM_CLASSES],       [1, NUM_CLASSES]
} Model;

// X: [N, 1, 28, 28]
static Tensor forward(Tensor X, Model *m)
{
    int N      = TensorShape(X, 0);
    Tensor c_1 = TensorConvolutionLayer(X,   m->K_1, m->b_1, 1, 1, RELU); // [N, 16, 28, 28]
    Tensor p_1 = TensorMaxPool(c_1, 2, 2, 0);                             // [N, 16, 14, 14]
    Tensor c_2 = TensorConvolutionLayer(p_1, m->K_2, m->b_2, 1, 1, RELU); // [N, 32, 14, 14]
    Tensor p_2 = TensorMaxPool(c_2, 2, 2, 0);                             // [N, 32, 7, 7]
    Tensor f   = TensorReshape(p_2, 2, (int[]){N, FEATURES});             // [N, FEATURES]
    Tensor h   = TensorLinearLayer(f, m->W_h, m->b_h, RELU);              // [N, HIDDEN_SIZE]
    Tensor Z   = TensorLinearLayer(h, m->W_y, m->b_y, IDENTITY);          // [N, NUM_CLASSES]

    TensorFree(c_1); TensorFree(p_1);
    TensorFree(c_2); TensorFree(p_2);
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
    printf("Train: %d  Test: %d\n\n", trainSamples, testSamples);

    // Reshape data Tensors from [..., 28, 28] to [..., 1, 28, 28] and scale entries from [0, 255] to [0, 1]
    xTrain = TensorReshape(xTrain, 4, (int[]){trainSamples, 1, 28, 28}); // [60000, 1, 28, 28]
    xTest  = TensorReshape(xTest,  4, (int[]){testSamples,  1, 28, 28}); // [10000, 1, 28, 28]
    TensorScale(xTrain, 1.0f / 255.0f);
    TensorScale(xTest,  1.0f / 255.0f);

    // Create and initialize convolution filters, weights and biases
    Model m;
    m.K_1 = TensorNew(4, (int[]){C1_CHANNELS,  1, 3, 3}, TRUE);          // [16, 1, 3, 3]
    m.b_1 = TensorNew(4, (int[]){1, C1_CHANNELS, 1, 1}, TRUE);           // [1, 16, 1, 1]
    m.K_2 = TensorNew(4, (int[]){C2_CHANNELS, C1_CHANNELS, 3, 3}, TRUE); // [32, 16, 3, 3]
    m.b_2 = TensorNew(4, (int[]){1, C2_CHANNELS, 1, 1}, TRUE);           // [1, 32, 1, 1]
    m.W_h = TensorNew(2, (int[]){FEATURES, HIDDEN_SIZE}, TRUE);          // [32*7*7, 128]
    m.b_h = TensorNew(2, (int[]){1, HIDDEN_SIZE}, TRUE);                 // [1, 128]
    m.W_y = TensorNew(2, (int[]){HIDDEN_SIZE, NUM_CLASSES}, TRUE);       // [128, 10]
    m.b_y = TensorNew(2, (int[]){1, NUM_CLASSES}, TRUE);                 // [1, 10]
    TensorXavier(m.K_1); TensorXavier(m.K_2);
    TensorXavier(m.W_h); TensorXavier(m.W_y);

    // One-hot encoding of yTrain needed to compute cross entropy loss
    Tensor oneHot = TensorOneHot(yTrain, NUM_CLASSES); // [60000, 10]

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        // Shuffle training data using epoch as seed
        Tensor xShuffled = TensorShuffle(xTrain, epoch); // [60000, 1, 28, 28]
        Tensor yShuffled = TensorShuffle(oneHot, epoch); // [60000, 10]

        for (int batch = 0; batch < trainBatches; batch++) {
            // Cap final batch size so we don't slice past the end of dataset
            int remaining = trainSamples - batch * BATCH_SIZE;
            int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

            Tensor X = TensorSlice(xShuffled, batch * BATCH_SIZE, N); // [N, 1, 28, 28]
            Tensor Y = TensorSlice(yShuffled, batch * BATCH_SIZE, N); // [N, 10]
            Tensor Z    = forward(X, &m);                             // [N, 10]
            Tensor loss = TensorCrossEntropyLoss(Y, Z, 1.0f);         // [] (Scalar)

            TensorBackward(loss);
            TensorSGD(m.K_1, LR, MOMENTUM);
            TensorSGD(m.b_1, LR, MOMENTUM);
            TensorSGD(m.K_2, LR, MOMENTUM);
            TensorSGD(m.b_2, LR, MOMENTUM);
            TensorSGD(m.W_h, LR, MOMENTUM);
            TensorSGD(m.b_h, LR, MOMENTUM);
            TensorSGD(m.W_y, LR, MOMENTUM);
            TensorSGD(m.b_y, LR, MOMENTUM);

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

            Tensor X = TensorSlice(xTest, batch * BATCH_SIZE, N); // [N, 1, 28, 28]
            Tensor Y = TensorSlice(yTest, batch * BATCH_SIZE, N); // [N]
            Tensor Z           = forward(X, &m);                  // [N, 10]
            Tensor predictions = TensorSample(Z, 0);              // [N]

            totalErrors  += TensorHammingDistance(predictions, Y);
            totalSamples += TensorSize(predictions);

            TensorFree(Z); TensorFree(predictions);
        }
        printf("Epoch %2d | Test Accuracy: %.2f%%\n", epoch, 100.0f * (totalSamples - totalErrors) / totalSamples);
    }

    TensorFree(m.K_1); TensorFree(m.b_1); TensorFree(m.K_2); TensorFree(m.b_2);
    TensorFree(m.W_h); TensorFree(m.b_h); TensorFree(m.W_y); TensorFree(m.b_y);
    TensorFree(oneHot);
    TensorFree(xTrain); TensorFree(yTrain);
    TensorFree(xTest);  TensorFree(yTest);
    return 0;
}
