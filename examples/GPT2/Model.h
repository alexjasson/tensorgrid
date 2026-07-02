#ifndef GPT2_MODEL_H
#define GPT2_MODEL_H

#include <math.h>
#include "tensorgrid.h"

#define VOCAB_SIZE  1024       // Change to train with different vocabulary size
#define NUM_CLASSES VOCAB_SIZE // Number of output classes (same as vocabulary size)
#define SEQ_LENGTH  16
#define EMBED_SIZE  64
#define MLP_HIDDEN  4*EMBED_SIZE
#define NUM_LAYERS  2
#define LN_EPSILON  1e-5f
#define PARAMETERS  "examples/GPT2/parameters.tgd"

/*
 * GPT2 style decoder only Transformer with token and position embeddings.
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
    Tensor merges;       // [NUM_CLASSES-256, 2] BPE merge pairs
} Model;

static inline Block BlockNew(TensorArray parameters)
{
    Block b;
    b.g_LN1 = TensorArrayAdd(parameters, 1, (int[]){EMBED_SIZE}, TRUE);
    b.b_LN1 = TensorArrayAdd(parameters, 1, (int[]){EMBED_SIZE}, TRUE);
    b.W_q   = TensorArrayAdd(parameters, 2, (int[]){EMBED_SIZE, EMBED_SIZE}, TRUE);
    b.b_q   = TensorArrayAdd(parameters, 1, (int[]){EMBED_SIZE}, TRUE);
    b.W_k   = TensorArrayAdd(parameters, 2, (int[]){EMBED_SIZE, EMBED_SIZE}, TRUE);
    b.b_k   = TensorArrayAdd(parameters, 1, (int[]){EMBED_SIZE}, TRUE);
    b.W_v   = TensorArrayAdd(parameters, 2, (int[]){EMBED_SIZE, EMBED_SIZE}, TRUE);
    b.b_v   = TensorArrayAdd(parameters, 1, (int[]){EMBED_SIZE}, TRUE);
    b.W_o   = TensorArrayAdd(parameters, 2, (int[]){EMBED_SIZE, EMBED_SIZE}, TRUE);
    b.b_o   = TensorArrayAdd(parameters, 1, (int[]){EMBED_SIZE}, TRUE);
    b.g_LN2 = TensorArrayAdd(parameters, 1, (int[]){EMBED_SIZE}, TRUE);
    b.b_LN2 = TensorArrayAdd(parameters, 1, (int[]){EMBED_SIZE}, TRUE);
    b.W_1   = TensorArrayAdd(parameters, 2, (int[]){EMBED_SIZE, MLP_HIDDEN}, TRUE);
    b.b_1   = TensorArrayAdd(parameters, 1, (int[]){MLP_HIDDEN}, TRUE);
    b.W_2   = TensorArrayAdd(parameters, 2, (int[]){MLP_HIDDEN, EMBED_SIZE}, TRUE);
    b.b_2   = TensorArrayAdd(parameters, 1, (int[]){EMBED_SIZE}, TRUE);
    TensorFill(b.g_LN1, 1.0f); TensorFill(b.g_LN2, 1.0f);
    TensorXavier(b.W_q); TensorXavier(b.W_k); TensorXavier(b.W_v); TensorXavier(b.W_o);
    TensorXavier(b.W_1); TensorXavier(b.W_2);
    return b;
}

static inline Tensor BlockForward(Tensor H, Block *b)
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

    Tensor S       = TensorAdd(scores, causalMask);                        // S_l, broadcasts [1, T, T]
    Tensor attn    = TensorSoftmax(S, sqrtf((float)EMBED_SIZE));
    Tensor attnOut = TensorMultiply(attn, V);                              // softmax(S_l) @ V_l
    Tensor A       = TensorLinearLayer(attnOut, b->W_o, b->b_o, IDENTITY); // A_l = ... @ W_o + b_o
    TensorFree(causalMask);
    TensorFree(scores); TensorFree(S); TensorFree(attn); TensorFree(V); TensorFree(attnOut);

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

static inline Model ModelNew(TensorArray parameters)
{
    Model m;
    m.E = TensorArrayAdd(parameters, 2, (int[]){NUM_CLASSES, EMBED_SIZE}, TRUE);
    m.P = TensorArrayAdd(parameters, 2, (int[]){SEQ_LENGTH, EMBED_SIZE}, TRUE);
    for (int l = 0; l < NUM_LAYERS; l++) m.blocks[l] = BlockNew(parameters);
    m.g_LNf = TensorArrayAdd(parameters, 1, (int[]){EMBED_SIZE}, TRUE);
    m.b_LNf = TensorArrayAdd(parameters, 1, (int[]){EMBED_SIZE}, TRUE);
    m.merges = TensorArrayAdd(parameters, 2, (int[]){NUM_CLASSES - 256, 2}, FALSE); // bundled, not trained
    TensorXavier(m.E); TensorXavier(m.P);
    TensorFill(m.g_LNf, 1.0f);
    return m;
}

// X: [N, SEQ_LENGTH] of token ids -> logits [N, SEQ_LENGTH, NUM_CLASSES]
static inline Tensor ModelForward(Tensor X, Model *m)
{
    Tensor E_X = TensorEmbedding(m->E, X);                           // E[x], [N, T, D]
    Tensor H   = TensorAdd(E_X, m->P);                               // H_0 = E[x] + P
    TensorFree(E_X);

    for (int l = 0; l < NUM_LAYERS; l++) {
        Tensor next = BlockForward(H, &m->blocks[l]);
        TensorFree(H);
        H = next;
    }

    Tensor Y   = TensorLayerNorm(H, m->g_LNf, m->b_LNf, LN_EPSILON); // Y = layerNorm(H_L)
    Tensor E_T = TensorTranspose(m->E);                              // E^T, [D, NUM_CLASSES]
    Tensor Z   = TensorMultiply(Y, E_T);                             // Z = Y @ E^T, [N, T, NUM_CLASSES]
    TensorFree(H); TensorFree(Y); TensorFree(E_T);
    return Z;
}

// Sample the next token id given a context window of SEQ_LENGTH token ids
static inline int ModelPredict(Model *m, int *context, float temperature)
{
    Tensor X = TensorNew(2, (int[]){1, SEQ_LENGTH}, 0);
    for (int i = 0; i < SEQ_LENGTH; i++)
        TensorSet(X, (int[]){0, i}, (Entry)context[i]);
    Tensor Z   = ModelForward(X, m);           // [1, SEQ_LENGTH, NUM_CLASSES]
    Tensor ids = TensorSample(Z, temperature); // [1, SEQ_LENGTH]
    int next   = (int)TensorGet(ids, (int[]){0, SEQ_LENGTH - 1});
    TensorFree(X); TensorFree(Z); TensorFree(ids);
    return next;
}

#endif
