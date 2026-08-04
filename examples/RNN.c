#include <stdio.h>
#include "tensorgrid.h"

#define NUM_CLASSES 128
#define SEQ_LENGTH  16
#define HIDDEN_SIZE 64
#define BATCH_SIZE  64
#define EPOCHS      5
#define LOG         50
#define LR          0.005f
#define BETA1       0.9f
#define BETA2       0.999f
#define EPSILON     1e-8f

/*
 * Single-layer character RNN with an embedding layer. Let N = batch size.
 *
 * e_t = E[x_t]
 * h_t = tanh(e_t + h_{t-1} @ U_h + b_h)
 * Z   = h_{SEQ_LENGTH-1} @ W_y + b_y
 */
typedef struct {
    Tensor E;          // [NUM_CLASSES, HIDDEN_SIZE]
    Tensor U_h, b_h;   // [HIDDEN_SIZE, HIDDEN_SIZE], [HIDDEN_SIZE]
    Tensor W_y, b_y;   // [HIDDEN_SIZE, NUM_CLASSES], [NUM_CLASSES]
} Model;

// X: [N, SEQ_LENGTH]
static Tensor forward(Tensor X, Model *m)
{
    int N = TensorShape(X, 0);

    Tensor X_T       = TensorTranspose(X);                           // [SEQ_LENGTH, N]
    Tensor embedding = TensorEmbedding(m->E, X_T);                   // [SEQ_LENGTH, N, HIDDEN_SIZE]
    Tensor h_prev    = TensorNew(2, (int[]){N, HIDDEN_SIZE}, FALSE); // [N, HIDDEN_SIZE]
    TensorFree(X_T);

    for (int t = 0; t < SEQ_LENGTH; t++) {
        Tensor e_t    = TensorIndex(embedding, t);                       // [N, HIDDEN_SIZE]
        Tensor e_bias = TensorAdd(e_t, m->b_h);                          // [N, HIDDEN_SIZE]
        Tensor h_curr = TensorLinearLayer(h_prev, m->U_h, e_bias, TANH); // [N, HIDDEN_SIZE]

        TensorFree(h_prev);
        TensorFree(e_bias);
        h_prev = h_curr;
    }
    TensorFree(embedding);

    Tensor Z = TensorLinearLayer(h_prev, m->W_y, m->b_y, IDENTITY); // [N, NUM_CLASSES]
    TensorFree(h_prev);
    return Z;
}

int main(void)
{
    // Read data into Tensors
    Tensor data = TensorReadData("examples/data/tinyShakespeare/input.txt", 1, (int[]){1115394}, 0, UINT8); // [1115394]
    Tensor train = TensorSlice(data, 0, TensorShape(data, 0) * 9 / 10);                                         // [1003854]
    Tensor test  = TensorSlice(data, TensorShape(train, 0), TensorShape(data, 0) - TensorShape(train, 0));      // [111540]

    // Compute number of training/test samples and number of batches
    int trainSamples = TensorShape(train, 0) - SEQ_LENGTH;
    int testSamples  = TensorShape(test,  0) - SEQ_LENGTH;
    int trainBatches = (trainSamples + BATCH_SIZE - 1) / BATCH_SIZE;
    int testBatches  = (testSamples  + BATCH_SIZE - 1) / BATCH_SIZE;

    Tensor xTrain = TensorRestride(train, 2, (int[]){trainSamples, SEQ_LENGTH}, (int[]){1, 1}, 0);
    Tensor xTest  = TensorRestride(test,  2, (int[]){testSamples,  SEQ_LENGTH}, (int[]){1, 1}, 0);
    Tensor yTrain = TensorSlice(train, SEQ_LENGTH, trainSamples);
    Tensor yTest  = TensorSlice(test,  SEQ_LENGTH, testSamples);

    TensorArray parameters = TensorArrayNew();
    Model m;
    m.E   = TensorArrayAdd(parameters, 2, (int[]){NUM_CLASSES, HIDDEN_SIZE}, TRUE);
    m.U_h = TensorArrayAdd(parameters, 2, (int[]){HIDDEN_SIZE, HIDDEN_SIZE}, TRUE);
    m.b_h = TensorArrayAdd(parameters, 1, (int[]){HIDDEN_SIZE}, TRUE);
    m.W_y = TensorArrayAdd(parameters, 2, (int[]){HIDDEN_SIZE, NUM_CLASSES}, TRUE);
    m.b_y = TensorArrayAdd(parameters, 1, (int[]){NUM_CLASSES}, TRUE);
    TensorXavier(m.E); TensorXavier(m.U_h); TensorXavier(m.W_y);

    printf("Train Samples: %d\n", trainSamples);
    printf("Test Samples:  %d\n", testSamples);
    printf("Parameters:    %d\n\n", TensorArraySize(parameters));

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        Tensor xShuffled = TensorShuffle(xTrain, epoch); // [trainSamples, SEQ_LENGTH]
        Tensor yShuffled = TensorShuffle(yTrain, epoch); // [trainSamples]

        for (int batch = 0; batch < trainBatches; batch++) {
            int remaining = trainSamples - batch * BATCH_SIZE;
            int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

            Tensor X    = TensorSlice(xShuffled, batch * BATCH_SIZE, N); // [N, SEQ_LENGTH]
            Tensor Y    = TensorSlice(yShuffled, batch * BATCH_SIZE, N); // [N]
            Tensor Z    = forward(X, &m);                                // [N, NUM_CLASSES]
            Tensor loss = TensorCrossEntropyLoss(Y, Z, 1.0f);            // [] (Scalar)

            TensorBackward(loss);
            TensorArrayAdam(parameters, LR, BETA1, BETA2, EPSILON, 0);

            if (((batch + 1) % LOG == 0) || (batch + 1 == trainBatches)) {
                printf("Epoch %2d | Batch %4d/%d | Loss: ", epoch, batch + 1, trainBatches);
                TensorPrint(loss);
            }

            TensorFree(Z); TensorFree(loss);
        }
        TensorFree(xShuffled); TensorFree(yShuffled);

        int totalSamples = 0, totalErrors = 0;
        for (int batch = 0; batch < testBatches; batch++) {
            int remaining = testSamples - batch * BATCH_SIZE;
            int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

            Tensor X = TensorSlice(xTest, batch * BATCH_SIZE, N); // [N, SEQ_LENGTH]
            Tensor Y = TensorSlice(yTest, batch * BATCH_SIZE, N); // [N]
            Tensor Z           = forward(X, &m);                  // [N, NUM_CLASSES]
            Tensor predictions = TensorSample(Z, 0);              // [N]

            totalErrors  += TensorHammingDistance(predictions, Y);
            totalSamples += TensorSize(predictions);

            TensorFree(Z); TensorFree(predictions);
        }
        printf("Epoch %2d | Test Accuracy: %.2f%%\n", epoch, 100.0f * (totalSamples - totalErrors) / totalSamples);
    }

    TensorArrayFree(parameters);
    TensorFree(xTrain); TensorFree(xTest);
    TensorFree(train); TensorFree(test); TensorFree(data);
    return 0;
}
