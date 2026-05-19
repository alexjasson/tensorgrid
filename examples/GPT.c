#include <stdio.h>
#include "tensorgrid.h"
#include "Parameters.h"

#define NUM_CLASSES 65
#define SEQ_LENGTH  16
#define EMBED_SIZE  64
#define MLP_HIDDEN  4*EMBED_SIZE
#define NUM_LAYERS  2
#define BATCH_SIZE  64
#define EPOCHS      5
#define LOG         50
#define LR          0.003f
#define BETA1       0.9f
#define BETA2       0.999f
#define EPSILON     1e-8f
#define LN_EPSILON  1e-5f

/*
 * GPT-2 style decoder-only Transformer with token and position embeddings.
 * Let N = batch size and T = SEQ_LENGTH.
 *
 * H_0 = E[x] + P
 *
 * For each Transformer block l = 1, ..., L:
 *     Q_l = LN(H_{l-1}) @ W_q + b_q
 *     K_l = LN(H_{l-1}) @ W_k + b_k
 *     V_l = LN(H_{l-1}) @ W_v + b_v
 *
 *     S_l = Q_l @ K_l^T / sqrt(d_head)
 *     S_l = causalMask(S_l)
 *     A_l = softmax(S_l) @ V_l
 *     A_l = A_l @ W_o + b_o
 *
 *     U_l = H_{l-1} + A_l
 *
 *     M_l = gelu(LN(U_l) @ W_1 + b_1)
 *     M_l = M_l @ W_2 + b_2
 *
 *     H_l = U_l + M_l
 *
 * Y = LN(H_L)
 * Z = Y @ E^T
 */
typedef struct {
    Tensor g_LN1, b_LN1; // [EMBED_SIZE], [EMBED_SIZE]
    Tensor W_q, b_q;     // [EMBED_SIZE, EMBED_SIZE], [EMBED_SIZE]
    Tensor W_k, b_k;
    Tensor W_v, b_v;
    Tensor W_o, b_o;
    Tensor g_LN2, b_LN2;
    Tensor W_1, b_1;     // [EMBED_SIZE, MLP_HIDDEN], [MLP_HIDDEN]
    Tensor W_2, b_2;     // [MLP_HIDDEN, EMBED_SIZE], [EMBED_SIZE]
} Block;

typedef struct {
    Tensor E;            // [NUM_CLASSES, EMBED_SIZE]
    Tensor P;            // [SEQ_LENGTH, EMBED_SIZE]
    Block  blocks[NUM_LAYERS];
    Tensor g_LNf, b_LNf; // [EMBED_SIZE], [EMBED_SIZE]
} Model;

static void newBlock(Parameters p, Block *b)
{
    b->g_LN1 = ParametersAdd(p, 1, (int[]){EMBED_SIZE});
    b->b_LN1 = ParametersAdd(p, 1, (int[]){EMBED_SIZE});
    b->W_q   = ParametersAdd(p, 2, (int[]){EMBED_SIZE, EMBED_SIZE});
    b->b_q   = ParametersAdd(p, 1, (int[]){EMBED_SIZE});
    b->W_k   = ParametersAdd(p, 2, (int[]){EMBED_SIZE, EMBED_SIZE});
    b->b_k   = ParametersAdd(p, 1, (int[]){EMBED_SIZE});
    b->W_v   = ParametersAdd(p, 2, (int[]){EMBED_SIZE, EMBED_SIZE});
    b->b_v   = ParametersAdd(p, 1, (int[]){EMBED_SIZE});
    b->W_o   = ParametersAdd(p, 2, (int[]){EMBED_SIZE, EMBED_SIZE});
    b->b_o   = ParametersAdd(p, 1, (int[]){EMBED_SIZE});
    b->g_LN2 = ParametersAdd(p, 1, (int[]){EMBED_SIZE});
    b->b_LN2 = ParametersAdd(p, 1, (int[]){EMBED_SIZE});
    b->W_1   = ParametersAdd(p, 2, (int[]){EMBED_SIZE, MLP_HIDDEN});
    b->b_1   = ParametersAdd(p, 1, (int[]){MLP_HIDDEN});
    b->W_2   = ParametersAdd(p, 2, (int[]){MLP_HIDDEN, EMBED_SIZE});
    b->b_2   = ParametersAdd(p, 1, (int[]){EMBED_SIZE});
    TensorFill(b->g_LN1, 1.0f); TensorFill(b->g_LN2, 1.0f);
    TensorXavier(b->W_q); TensorXavier(b->W_k); TensorXavier(b->W_v); TensorXavier(b->W_o);
    TensorXavier(b->W_1); TensorXavier(b->W_2);
}

static Tensor forwardBlock(Tensor H, Block *b)
{
    Tensor causalMask = TensorCausalMask(SEQ_LENGTH);                      // [T, T]
    causalMask = TensorReshape(causalMask, 3, (int[]){1, SEQ_LENGTH, SEQ_LENGTH});

    Tensor LN_H = TensorLayerNorm(H, b->g_LN1, b->b_LN1, LN_EPSILON);      // [N, T, D]
    Tensor Q    = TensorLinearLayer(LN_H, b->W_q, b->b_q, IDENTITY);       // [N, T, D]
    Tensor K    = TensorLinearLayer(LN_H, b->W_k, b->b_k, IDENTITY);
    Tensor V    = TensorLinearLayer(LN_H, b->W_v, b->b_v, IDENTITY);
    TensorFree(LN_H);

    Tensor K_T    = TensorTranspose(K);                                    // [N, D, T]
    Tensor scores = TensorMultiply(Q, K_T);                                // [N, T, T]
    TensorFree(Q); TensorFree(K); TensorFree(K_T);

    Tensor scale = TensorNew(scores.dimensions, scores.shape, FALSE);
    TensorFill(scale, 1.0f / SQRT.f((float)EMBED_SIZE));
    Tensor scaled = TensorHadamard(scores, scale);                         // [N, T, T]
    TensorFree(scores); TensorFree(scale);

    Tensor S       = TensorAdd(scaled, causalMask);                        // S_l, broadcasts [1, T, T]
    Tensor attn    = TensorSoftmax(S, 1.0f);                               // softmax(S_l)
    Tensor attnOut = TensorMultiply(attn, V);                              // softmax(S_l) @ V_l
    Tensor A       = TensorLinearLayer(attnOut, b->W_o, b->b_o, IDENTITY); // A_l = ... @ W_o + b_o
    TensorFree(causalMask);
    TensorFree(scaled); TensorFree(S); TensorFree(attn); TensorFree(V); TensorFree(attnOut);

    Tensor U = TensorAdd(H, A);                                            // U_l = H_{l-1} + A_l
    TensorFree(A);

    Tensor LN_U = TensorLayerNorm(U, b->g_LN2, b->b_LN2, LN_EPSILON);
    Tensor mlp  = TensorLinearLayer(LN_U, b->W_1, b->b_1, GELU);           // gelu(LN_U @ W_1 + b_1)
    Tensor M    = TensorLinearLayer(mlp, b->W_2, b->b_2, IDENTITY);        // M_l = ... @ W_2 + b_2
    TensorFree(LN_U); TensorFree(mlp);

    Tensor H_l = TensorAdd(U, M);                                          // H_l = U_l + M_l
    TensorFree(U); TensorFree(M);
    return H_l;
}

// X: [N, SEQ_LENGTH]
static Tensor forward(Tensor X, Model *m)
{
    Tensor E_X = TensorEmbedding(m->E, X);                               // E[x], [N, T, D]
    Tensor H   = TensorAdd(E_X, m->P);                                   // H_0 = E[x] + P
    TensorFree(E_X);

    for (int l = 0; l < NUM_LAYERS; l++) {
        Tensor next = forwardBlock(H, &m->blocks[l]);
        TensorFree(H);
        H = next;
    }

    Tensor Y   = TensorLayerNorm(H, m->g_LNf, m->b_LNf, LN_EPSILON);     // Y = layerNorm(H_L)
    Tensor E_T = TensorTranspose(m->E);                                  // E^T, [D, NUM_CLASSES]
    Tensor Z   = TensorMultiply(Y, E_T);                                 // Z = Y @ E^T, [N, T, NUM_CLASSES]
    TensorFree(H); TensorFree(Y); TensorFree(E_T);
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

    Tensor xTrain = TensorRestride(train, 2, (int[]){trainSamples, SEQ_LENGTH}, (int[]){1, 1}, 0);
    Tensor yTrain = TensorRestride(train, 2, (int[]){trainSamples, SEQ_LENGTH}, (int[]){1, 1}, 1);
    Tensor xTest  = TensorRestride(test,  2, (int[]){testSamples,  SEQ_LENGTH}, (int[]){1, 1}, 0);
    Tensor yTest  = TensorRestride(test,  2, (int[]){testSamples,  SEQ_LENGTH}, (int[]){1, 1}, 1);

    Parameters p = ParametersNew();
    Model m;
    m.E = ParametersAdd(p, 2, (int[]){NUM_CLASSES, EMBED_SIZE});
    m.P = ParametersAdd(p, 2, (int[]){SEQ_LENGTH, EMBED_SIZE});
    for (int l = 0; l < NUM_LAYERS; l++) newBlock(p, &m.blocks[l]);
    m.g_LNf = ParametersAdd(p, 1, (int[]){EMBED_SIZE});
    m.b_LNf = ParametersAdd(p, 1, (int[]){EMBED_SIZE});
    TensorXavier(m.E); TensorXavier(m.P);
    TensorFill(m.g_LNf, 1.0f);
    printf("Train Samples: %d\n", trainSamples);
    printf("Test Samples:  %d\n", testSamples);
    printf("Parameters:    %d\n\n", ParametersCount(p));

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        Tensor xShuffled = TensorShuffle(xTrain, epoch); // [trainSamples, SEQ_LENGTH]
        Tensor yShuffled = TensorShuffle(yTrain, epoch); // [trainSamples, SEQ_LENGTH]

        for (int batch = 0; batch < trainBatches; batch++) {
            int remaining = trainSamples - batch * BATCH_SIZE;
            int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

            Tensor X    = TensorSlice(xShuffled, batch * BATCH_SIZE, N);
            Tensor Y    = TensorOneHot(TensorSlice(yShuffled, batch * BATCH_SIZE, N), NUM_CLASSES);
            Tensor Z    = forward(X, &m);                     // [N, SEQ_LENGTH, NUM_CLASSES]
            Tensor loss = TensorCrossEntropyLoss(Y, Z, 1.0f); // [] (Scalar)

            TensorBackward(loss);
            ParametersAdam(p, LR, BETA1, BETA2, EPSILON, 0);

            if (((batch + 1) % LOG == 0) || (batch + 1 == trainBatches)) {
                printf("Epoch %2d | Batch %4d/%d | Loss: ", epoch, batch + 1, trainBatches);
                TensorPrint(loss);
            }

            TensorFree(Y); TensorFree(Z); TensorFree(loss);
        }
        TensorFree(xShuffled); TensorFree(yShuffled);

        int totalSamples = 0, totalErrors = 0;
        for (int batch = 0; batch < testBatches; batch++) {
            int remaining = testSamples - batch * BATCH_SIZE;
            int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

            Tensor X           = TensorSlice(xTest, batch * BATCH_SIZE, N); // [N, SEQ_LENGTH]
            Tensor Y           = TensorSlice(yTest, batch * BATCH_SIZE, N); // [N, SEQ_LENGTH]
            Tensor Z           = forward(X, &m);                            // [N, SEQ_LENGTH, NUM_CLASSES]
            Tensor predictions = TensorSample(Z, 0);                        // [N, SEQ_LENGTH]

            totalErrors  += TensorHammingDistance(predictions, Y);
            totalSamples += TensorSize(predictions);

            TensorFree(Z); TensorFree(predictions);
        }
        printf("Epoch %2d | Test Accuracy: %.2f%%\n", epoch, 100.0f * (totalSamples - totalErrors) / totalSamples);
    }

    ParametersFree(p);
    TensorFree(xTrain); TensorFree(xTest);
    TensorFree(train);  TensorFree(test);
    return 0;
}