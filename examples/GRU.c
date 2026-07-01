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
 * Gated Recurrent Unit (GRU) with an embedding layer. Let N = batch size.
 *
 * e_t       = E[x_t]
 * z_t       = sigmoid(e_t + h_{t-1} @ U_z + b_z)
 * r_t       = sigmoid(e_t + h_{t-1} @ U_r + b_r)
 * h_tilde_t = tanh(e_t + (r_t * h_{t-1}) @ U_h + b_h)
 * h_t       = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
 * Z         = h_{SEQ_LENGTH-1} @ W_y + b_y
 */
typedef struct {
    Tensor E;        // [NUM_CLASSES, HIDDEN_SIZE]
    Tensor U_z, b_z; // [HIDDEN_SIZE, HIDDEN_SIZE], [HIDDEN_SIZE]
    Tensor U_r, b_r; // [HIDDEN_SIZE, HIDDEN_SIZE], [HIDDEN_SIZE]
    Tensor U_h, b_h; // [HIDDEN_SIZE, HIDDEN_SIZE], [HIDDEN_SIZE]
    Tensor W_y, b_y; // [HIDDEN_SIZE, NUM_CLASSES], [NUM_CLASSES]
} Model;

// X: [N, SEQ_LENGTH]
static Tensor forward(Tensor X, Model *m)
{
    int N = TensorShape(X, 0);

    Tensor X_T       = TensorTranspose(X);                           // [SEQ_LENGTH, N]
    Tensor embedding = TensorEmbedding(m->E, X_T);                   // [SEQ_LENGTH, N, HIDDEN_SIZE]
    Tensor h_prev    = TensorNew(2, (int[]){N, HIDDEN_SIZE}, FALSE); // [N, HIDDEN_SIZE]
    Tensor ones      = TensorNew(2, (int[]){N, HIDDEN_SIZE}, FALSE); // [N, HIDDEN_SIZE]
    TensorFill(ones, 1.0f);
    TensorFree(X_T);

    for (int t = 0; t < SEQ_LENGTH; t++) {
        Tensor e_t = TensorIndex(embedding, t);                                 // [N, HIDDEN_SIZE]

        Tensor e_bias_z = TensorAdd(e_t, m->b_z);                               // [N, HIDDEN_SIZE]
        Tensor z_t      = TensorLinearLayer(h_prev, m->U_z, e_bias_z, SIGMOID); // [N, HIDDEN_SIZE]
        TensorFree(e_bias_z);

        Tensor e_bias_r = TensorAdd(e_t, m->b_r);                               // [N, HIDDEN_SIZE]
        Tensor r_t      = TensorLinearLayer(h_prev, m->U_r, e_bias_r, SIGMOID); // [N, HIDDEN_SIZE]
        TensorFree(e_bias_r);

        Tensor r_h      = TensorHadamard(r_t, h_prev);                          // [N, HIDDEN_SIZE]
        Tensor e_bias_h = TensorAdd(e_t, m->b_h);                               // [N, HIDDEN_SIZE]
        Tensor h_tilde  = TensorLinearLayer(r_h, m->U_h, e_bias_h, TANH);       // [N, HIDDEN_SIZE]
        TensorFree(r_h);
        TensorFree(e_bias_h);

        Tensor one_minus_z = TensorSubtract(ones, z_t);                         // [N, HIDDEN_SIZE]
        Tensor left        = TensorHadamard(one_minus_z, h_prev);               // [N, HIDDEN_SIZE]
        Tensor right       = TensorHadamard(z_t, h_tilde);                      // [N, HIDDEN_SIZE]
        Tensor h_curr      = TensorAdd(left, right);                            // [N, HIDDEN_SIZE]

        TensorFree(z_t); TensorFree(r_t); TensorFree(h_tilde);
        TensorFree(one_minus_z); TensorFree(left); TensorFree(right);
        TensorFree(h_prev);
        h_prev = h_curr;
    }
    TensorFree(embedding);
    TensorFree(ones);

    Tensor Z = TensorLinearLayer(h_prev, m->W_y, m->b_y, IDENTITY); // [N, NUM_CLASSES]
    TensorFree(h_prev);
    return Z;
}

int main(void)
{
    // Read data into Tensors
    Tensor data = TensorReadData("examples/data/tinyShakespeare/input.txt", 1, (int[]){1115394}, 0, DataUint8); // [1115394]
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
    m.E   = TensorArrayAdd(parameters, 2, (int[]){NUM_CLASSES, HIDDEN_SIZE});
    m.U_z = TensorArrayAdd(parameters, 2, (int[]){HIDDEN_SIZE, HIDDEN_SIZE});
    m.b_z = TensorArrayAdd(parameters, 1, (int[]){HIDDEN_SIZE});
    m.U_r = TensorArrayAdd(parameters, 2, (int[]){HIDDEN_SIZE, HIDDEN_SIZE});
    m.b_r = TensorArrayAdd(parameters, 1, (int[]){HIDDEN_SIZE});
    m.U_h = TensorArrayAdd(parameters, 2, (int[]){HIDDEN_SIZE, HIDDEN_SIZE});
    m.b_h = TensorArrayAdd(parameters, 1, (int[]){HIDDEN_SIZE});
    m.W_y = TensorArrayAdd(parameters, 2, (int[]){HIDDEN_SIZE, NUM_CLASSES});
    m.b_y = TensorArrayAdd(parameters, 1, (int[]){NUM_CLASSES});
    TensorXavier(m.E);
    TensorXavier(m.U_z); TensorXavier(m.U_r); TensorXavier(m.U_h);
    TensorXavier(m.W_y);

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
