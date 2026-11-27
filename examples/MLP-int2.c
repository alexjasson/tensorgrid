#include <stdio.h>
#include "tensorgrid.h"

#define NUM_CLASSES 10
#define IMAGE_SIZE  784
#define HIDDEN_SIZE 128
#define BATCH_SIZE  64
#define EPOCHS      10
#define LOG         300
#define LR          0.01f
#define MOMENTUM    0.9f

/*
 * The same as MLP.c but with all parameters stored as ternary INT2.
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
    Tensor xTrain = TensorReadData("examples/data/mnist/train-images-idx3-ubyte", 2, (int[]){60000, IMAGE_SIZE}, 16, UINT8);
    Tensor yTrain = TensorReadData("examples/data/mnist/train-labels-idx1-ubyte", 1, (int[]){60000}, 8, UINT8);
    Tensor xTest  = TensorReadData("examples/data/mnist/t10k-images-idx3-ubyte", 2, (int[]){10000, IMAGE_SIZE}, 16, UINT8);
    Tensor yTest  = TensorReadData("examples/data/mnist/t10k-labels-idx1-ubyte", 1, (int[]){10000}, 8, UINT8);

    int trainSamples = TensorShape(xTrain)[0];
    int testSamples  = TensorShape(xTest)[0];
    int trainBatches = (trainSamples + BATCH_SIZE - 1) / BATCH_SIZE;
    int testBatches  = (testSamples  + BATCH_SIZE - 1) / BATCH_SIZE;

    // Scale entries from [0, 255] to [0, 1]
    xTrain = TensorScalarMultiply(xTrain, 1.0f / 255);
    xTest  = TensorScalarMultiply(xTest,  1.0f / 255);

    TensorArray parameters = TensorArrayNew();
    Model m;
    m.W_h = TensorArrayAdd(parameters, 2, (int[]){IMAGE_SIZE,  HIDDEN_SIZE}, INT2, TRUE);
    m.b_h = TensorArrayAdd(parameters, 1, (int[]){HIDDEN_SIZE}, INT2, TRUE);
    m.W_y = TensorArrayAdd(parameters, 2, (int[]){HIDDEN_SIZE, NUM_CLASSES}, INT2, TRUE);
    m.b_y = TensorArrayAdd(parameters, 1, (int[]){NUM_CLASSES}, INT2, TRUE);
    TensorXavier(m.W_h); TensorXavier(m.W_y);

    printf("Train Samples: %d\n", trainSamples);
    printf("Test Samples:  %d\n", testSamples);
    printf("Parameters:    %d\n\n", TensorArraySize(parameters));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        Tensor xShuffled = TensorShuffle(xTrain, epoch + 1);
        Tensor yShuffled = TensorShuffle(yTrain, epoch + 1);
        for (int batch = 0; batch < trainBatches; batch++) {
            int remaining = trainSamples - batch * BATCH_SIZE;
            int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

            Tensor X = TensorSlice(xShuffled, batch * BATCH_SIZE, N);
            Tensor Y = TensorSlice(yShuffled, batch * BATCH_SIZE, N);
            Tensor Z    = forward(X, &m);
            Tensor loss = TensorCrossEntropyLoss(Y, Z, 1.0f);

            TensorBackward(loss);
            TensorArraySGD(parameters, LR, MOMENTUM);

            if ((batch + 1) % LOG == 0 || batch + 1 == trainBatches) {
                printf("Epoch %2d | Batch %3d/%d | Loss: ", epoch + 1, batch + 1, trainBatches);
                TensorPrint(loss);
            }

            TensorFree(X); TensorFree(Y);
            TensorFree(Z); TensorFree(loss);
        }
        TensorFree(xShuffled); TensorFree(yShuffled);

        int totalErrors = 0;
        for (int batch = 0; batch < testBatches; batch++) {
            int remaining = testSamples - batch * BATCH_SIZE;
            int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

            Tensor X = TensorSlice(xTest, batch * BATCH_SIZE, N);
            Tensor Y = TensorSlice(yTest, batch * BATCH_SIZE, N);
            Tensor Z           = forward(X, &m);
            Tensor predictions = TensorSample(Z, 0);

            totalErrors += TensorHammingDistance(predictions, Y);
            TensorFree(X); TensorFree(Y);
            TensorFree(Z); TensorFree(predictions);
        }
        printf("Epoch %2d | Test Accuracy: %.2f%%\n\n", epoch + 1,
               100.0f * (testSamples - totalErrors) / testSamples);
    }

    TensorArrayFree(parameters);
    TensorFree(xTrain); TensorFree(yTrain);
    TensorFree(xTest);  TensorFree(yTest);
    return 0;
}
