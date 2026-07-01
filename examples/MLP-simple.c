#include <stdio.h>
#include "tensorgrid.h"

#define IMAGE_SIZE  784
#define HIDDEN_SIZE 128
#define NUM_CLASSES 10
#define BATCH_SIZE  64
#define LOG         100
#define LR          0.01f

/*
 * A minimalist MLP with a single hidden layer
 *
 * h = relu(X @ W_h + b_h)
 * Z = h @ W_y + b_y
 */
int main(void)
{
    // Read training data, flatten [N, 28, 28] -> [N, 784] and scale [0, 255] -> [0, 1]
    Tensor xTrain = TensorReadData("examples/data/mnist/train-images-idx3-ubyte", 3, (int[]){60000, 28, 28}, 16, DataUint8); // [60000, 28, 28]
    Tensor yTrain = TensorReadData("examples/data/mnist/train-labels-idx1-ubyte", 1, (int[]){60000}, 8, DataUint8);          // [60000]
    int samples = TensorShape(xTrain, 0);
    xTrain = TensorReshape(xTrain, 2, (int[]){samples, IMAGE_SIZE}); // [60000, 784]
    TensorScale(xTrain, 1.0f / 255.0f);

    // Create and initialize hidden and output layer weights/biases
    TensorArray parameters = TensorArrayNew();
    Tensor W_h = TensorArrayAdd(parameters, 2, (int[]){IMAGE_SIZE,  HIDDEN_SIZE}, TRUE); // [784, 128]
    Tensor b_h = TensorArrayAdd(parameters, 1, (int[]){HIDDEN_SIZE}, TRUE);              // [128]
    Tensor W_y = TensorArrayAdd(parameters, 2, (int[]){HIDDEN_SIZE, NUM_CLASSES}, TRUE); // [128, 10]
    Tensor b_y = TensorArrayAdd(parameters, 1, (int[]){NUM_CLASSES}, TRUE);              // [10]
    TensorXavier(W_h); TensorXavier(W_y);

    // One epoch of training
    for (int batch = 0; batch < samples / BATCH_SIZE; batch++) {
        Tensor X = TensorSlice(xTrain, batch * BATCH_SIZE, BATCH_SIZE); // [64, 784]
        Tensor Y = TensorSlice(yTrain, batch * BATCH_SIZE, BATCH_SIZE); // [64]

        Tensor h    = TensorLinearLayer(X, W_h, b_h, RELU);      // [64, 128]
        Tensor Z    = TensorLinearLayer(h,  W_y, b_y, IDENTITY); // [64, 10]
        Tensor loss = TensorCrossEntropyLoss(Y, Z, 1.0f);        // [] (scalar)

        TensorBackward(loss);
        TensorArraySGD(parameters, LR, 0.0f);

        if ((batch + 1) % LOG == 0) {
            printf("Batch %3d | Loss: ", batch + 1);
            TensorPrint(loss);
        }

        TensorFree(h); TensorFree(Z); TensorFree(loss);
    }

    TensorArrayFree(parameters);
    TensorFree(xTrain); TensorFree(yTrain);
    return 0;
}
