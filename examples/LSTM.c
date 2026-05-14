#include <stdio.h>
#include "tensorgrid.h"

#define NUM_CLASSES 65
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
 * Long Short-Term Memory (LSTM) with an embedding layer. Let N = batch size.
 *
 * e_t       = E[x_t]
 * f_t       = sigmoid(e_t + h_{t-1} @ U_f + b_f)
 * i_t       = sigmoid(e_t + h_{t-1} @ U_i + b_i)
 * o_t       = sigmoid(e_t + h_{t-1} @ U_o + b_o)
 * c_tilde_t = tanh(e_t + h_{t-1} @ U_c + b_c)
 * c_t       = f_t * c_{t-1} + i_t * c_tilde_t
 * h_t       = o_t * tanh(c_t)
 * Z         = h_{SEQ_LENGTH-1} @ W_y + b_y
 */
typedef struct {
    Tensor E;          // [NUM_CLASSES, HIDDEN_SIZE]
    Tensor U_f, b_f;   // [HIDDEN_SIZE, HIDDEN_SIZE], [1, HIDDEN_SIZE]
    Tensor U_i, b_i;   // [HIDDEN_SIZE, HIDDEN_SIZE], [1, HIDDEN_SIZE]
    Tensor U_o, b_o;   // [HIDDEN_SIZE, HIDDEN_SIZE], [1, HIDDEN_SIZE]
    Tensor U_c, b_c;   // [HIDDEN_SIZE, HIDDEN_SIZE], [1, HIDDEN_SIZE]
    Tensor W_y, b_y;   // [HIDDEN_SIZE, NUM_CLASSES], [1, NUM_CLASSES]
} Model;

// X: [N, SEQ_LENGTH]
static Tensor forward(Tensor X, Model *m)
{
    int N = TensorShape(X, 0);

    Tensor X_T       = TensorTranspose(X);                           // [SEQ_LENGTH, N]
    Tensor embedding = TensorEmbedding(m->E, X_T);                   // [SEQ_LENGTH, N, HIDDEN_SIZE]
    Tensor h_prev    = TensorNew(2, (int[]){N, HIDDEN_SIZE}, FALSE); // [N, HIDDEN_SIZE]
    Tensor c_prev    = TensorNew(2, (int[]){N, HIDDEN_SIZE}, FALSE); // [N, HIDDEN_SIZE]
    TensorFree(X_T);

    for (int t = 0; t < SEQ_LENGTH; t++) {
        Tensor e_t = TensorIndex(embedding, t);                                 // [N, HIDDEN_SIZE]

        Tensor e_bias_f = TensorAdd(e_t, m->b_f);                               // [N, HIDDEN_SIZE]
        Tensor f_t      = TensorLinearLayer(h_prev, m->U_f, e_bias_f, SIGMOID); // [N, HIDDEN_SIZE]
        TensorFree(e_bias_f);

        Tensor e_bias_i = TensorAdd(e_t, m->b_i);                               // [N, HIDDEN_SIZE]
        Tensor i_t      = TensorLinearLayer(h_prev, m->U_i, e_bias_i, SIGMOID); // [N, HIDDEN_SIZE]
        TensorFree(e_bias_i);

        Tensor e_bias_o = TensorAdd(e_t, m->b_o);                               // [N, HIDDEN_SIZE]
        Tensor o_t      = TensorLinearLayer(h_prev, m->U_o, e_bias_o, SIGMOID); // [N, HIDDEN_SIZE]
        TensorFree(e_bias_o);

        Tensor e_bias_c = TensorAdd(e_t, m->b_c);                               // [N, HIDDEN_SIZE]
        Tensor c_tilde  = TensorLinearLayer(h_prev, m->U_c, e_bias_c, TANH);    // [N, HIDDEN_SIZE]
        TensorFree(e_bias_c);

        Tensor left   = TensorHadamard(f_t, c_prev);                            // [N, HIDDEN_SIZE]
        Tensor right  = TensorHadamard(i_t, c_tilde);                           // [N, HIDDEN_SIZE]
        Tensor c_curr = TensorAdd(left, right);                                 // [N, HIDDEN_SIZE]

        Tensor c_tanh = TensorMap(TANH, c_curr);                                // [N, HIDDEN_SIZE]
        Tensor h_curr = TensorHadamard(o_t, c_tanh);                            // [N, HIDDEN_SIZE]

        TensorFree(f_t); TensorFree(i_t); TensorFree(o_t); TensorFree(c_tilde);
        TensorFree(left); TensorFree(right); TensorFree(c_tanh);
        TensorFree(h_prev); TensorFree(c_prev);
        h_prev = h_curr;
        c_prev = c_curr;
    }
    TensorFree(embedding);
    TensorFree(c_prev);

    Tensor Z = TensorLinearLayer(h_prev, m->W_y, m->b_y, IDENTITY); // [N, NUM_CLASSES]
    TensorFree(h_prev);
    return Z;
}

int main(void)
{
    // Read data into Tensors from .npy files
    Tensor train = TensorRead("data/tinyShakespeare/train.npy"); // [1003854]
    Tensor test  = TensorRead("data/tinyShakespeare/test.npy");  // [111540]

    // Compute number of training/test samples and number of batches
    int trainSamples = TensorShape(train, 0) - SEQ_LENGTH;
    int testSamples  = TensorShape(test,  0) - SEQ_LENGTH;
    int trainBatches = (trainSamples + BATCH_SIZE - 1) / BATCH_SIZE;
    int testBatches  = (testSamples  + BATCH_SIZE - 1) / BATCH_SIZE;
    printf("Train: %d  Test: %d\n\n", trainSamples, testSamples);

    Tensor xTrain = TensorRestride(train, 2, (int[]){trainSamples, SEQ_LENGTH}, (int[]){1, 1}, 0);
    Tensor xTest  = TensorRestride(test,  2, (int[]){testSamples,  SEQ_LENGTH}, (int[]){1, 1}, 0);
    Tensor yTrain = TensorSlice(train, SEQ_LENGTH, trainSamples);
    Tensor yTest  = TensorSlice(test,  SEQ_LENGTH, testSamples);

    Model m;
    m.E   = TensorNew(2, (int[]){NUM_CLASSES, HIDDEN_SIZE}, TRUE);
    m.U_f = TensorNew(2, (int[]){HIDDEN_SIZE, HIDDEN_SIZE}, TRUE);
    m.b_f = TensorNew(2, (int[]){1, HIDDEN_SIZE}, TRUE);
    m.U_i = TensorNew(2, (int[]){HIDDEN_SIZE, HIDDEN_SIZE}, TRUE);
    m.b_i = TensorNew(2, (int[]){1, HIDDEN_SIZE}, TRUE);
    m.U_o = TensorNew(2, (int[]){HIDDEN_SIZE, HIDDEN_SIZE}, TRUE);
    m.b_o = TensorNew(2, (int[]){1, HIDDEN_SIZE}, TRUE);
    m.U_c = TensorNew(2, (int[]){HIDDEN_SIZE, HIDDEN_SIZE}, TRUE);
    m.b_c = TensorNew(2, (int[]){1, HIDDEN_SIZE}, TRUE);
    m.W_y = TensorNew(2, (int[]){HIDDEN_SIZE, NUM_CLASSES}, TRUE);
    m.b_y = TensorNew(2, (int[]){1, NUM_CLASSES}, TRUE);
    TensorXavier(m.E);
    TensorXavier(m.U_f); TensorXavier(m.U_i); TensorXavier(m.U_o); TensorXavier(m.U_c);
    TensorXavier(m.W_y);

    // One-hot encoding of yTrain needed to compute cross entropy loss
    Tensor oneHot = TensorOneHot(yTrain, NUM_CLASSES); // [trainSamples, NUM_CLASSES]

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        Tensor xShuffled = TensorShuffle(xTrain, epoch); // [trainSamples, SEQ_LENGTH]
        Tensor yShuffled = TensorShuffle(oneHot, epoch); // [trainSamples, NUM_CLASSES]

        for (int batch = 0; batch < trainBatches; batch++) {
            int remaining = trainSamples - batch * BATCH_SIZE;
            int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

            Tensor X    = TensorSlice(xShuffled, batch * BATCH_SIZE, N); // [N, SEQ_LENGTH] view
            Tensor Y    = TensorSlice(yShuffled, batch * BATCH_SIZE, N); // [N, NUM_CLASSES]
            Tensor Z    = forward(X, &m);                                // [N, NUM_CLASSES]
            Tensor loss = TensorCrossEntropyLoss(Y, Z, 1.0f);            // [] (Scalar)

            TensorBackward(loss);
            TensorAdam(m.E,   LR, BETA1, BETA2, EPSILON, 0);
            TensorAdam(m.U_f, LR, BETA1, BETA2, EPSILON, 0);
            TensorAdam(m.b_f, LR, BETA1, BETA2, EPSILON, 0);
            TensorAdam(m.U_i, LR, BETA1, BETA2, EPSILON, 0);
            TensorAdam(m.b_i, LR, BETA1, BETA2, EPSILON, 0);
            TensorAdam(m.U_o, LR, BETA1, BETA2, EPSILON, 0);
            TensorAdam(m.b_o, LR, BETA1, BETA2, EPSILON, 0);
            TensorAdam(m.U_c, LR, BETA1, BETA2, EPSILON, 0);
            TensorAdam(m.b_c, LR, BETA1, BETA2, EPSILON, 0);
            TensorAdam(m.W_y, LR, BETA1, BETA2, EPSILON, 0);
            TensorAdam(m.b_y, LR, BETA1, BETA2, EPSILON, 0);

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

            Tensor X = TensorSlice(xTest, batch * BATCH_SIZE, N); // [N, SEQ_LENGTH] view
            Tensor Y = TensorSlice(yTest, batch * BATCH_SIZE, N); // [N]
            Tensor Z           = forward(X, &m);                  // [N, NUM_CLASSES]
            Tensor predictions = TensorSample(Z, 0);              // [N]

            totalErrors  += TensorHammingDistance(predictions, Y);
            totalSamples += TensorSize(predictions);

            TensorFree(Z); TensorFree(predictions);
        }
        printf("Epoch %2d | Test Accuracy: %.2f%%\n", epoch, 100.0f * (totalSamples - totalErrors) / totalSamples);
    }

    TensorFree(m.E);
    TensorFree(m.U_f); TensorFree(m.b_f);
    TensorFree(m.U_i); TensorFree(m.b_i);
    TensorFree(m.U_o); TensorFree(m.b_o);
    TensorFree(m.U_c); TensorFree(m.b_c);
    TensorFree(m.W_y); TensorFree(m.b_y);
    TensorFree(oneHot);
    TensorFree(xTrain); TensorFree(xTest);
    TensorFree(train); TensorFree(test);
    return 0;
}
