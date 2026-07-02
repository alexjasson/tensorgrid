#include <stdio.h>
#include "Model.h"

#define BATCH_SIZE  64
#define EPOCHS      5
#define LOG         50
#define LR          0.003f
#define BETA1       0.9f
#define BETA2       0.999f
#define EPSILON     1e-8f

int main(void)
{
    // Read the corpus as a 1D tensor of bytes and create a new model with uninitialized parameters
    Tensor corpus = TensorReadData("examples/data/tinyShakespeare/input.txt", 1, (int[]){1115394}, 0, DataUint8); // [1115394]
    TensorArray parameters = TensorArrayNew();
    Model m = ModelNew(parameters);

    // Read parameters (includes BPE merges) if they exist, otherwise train the tokenizer
    if (TensorArrayRead(parameters, PARAMETERS)) {
        printf("Parameters loaded from %s\n", PARAMETERS);
    } else {
        printf("Training tokenizer (may take a moment)...\n");
        TensorBPETrain(corpus, m.merges);
    }

    // Encode the corpus with BPE
    Tensor data = TensorBPEEncode(corpus, m.merges);
    TensorFree(corpus);

    // Split 90/10 into train/test
    Tensor train = TensorSlice(data, 0, TensorShape(data, 0) * 9 / 10);
    Tensor test  = TensorSlice(data, TensorShape(train, 0), TensorShape(data, 0) - TensorShape(train, 0));

    // Compute number of training/test samples and number of batches
    int trainSamples = TensorShape(train, 0) - SEQ_LENGTH;
    int testSamples  = TensorShape(test,  0) - SEQ_LENGTH;
    int trainBatches = (trainSamples + BATCH_SIZE - 1) / BATCH_SIZE;
    int testBatches  = (testSamples  + BATCH_SIZE - 1) / BATCH_SIZE;

    Tensor xTrain = TensorRestride(train, 2, (int[]){trainSamples, SEQ_LENGTH}, (int[]){1, 1}, 0);
    Tensor yTrain = TensorRestride(train, 2, (int[]){trainSamples, SEQ_LENGTH}, (int[]){1, 1}, 1);
    Tensor xTest  = TensorRestride(test,  2, (int[]){testSamples,  SEQ_LENGTH}, (int[]){1, 1}, 0);
    Tensor yTest  = TensorRestride(test,  2, (int[]){testSamples,  SEQ_LENGTH}, (int[]){1, 1}, 1);

    printf("Train Samples: %d\n", trainSamples);
    printf("Test Samples:  %d\n", testSamples);
    printf("Parameters:    %d\n\n", TensorArraySize(parameters));

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        Tensor xShuffled = TensorShuffle(xTrain, epoch); // [trainSamples, SEQ_LENGTH]
        Tensor yShuffled = TensorShuffle(yTrain, epoch); // [trainSamples, SEQ_LENGTH]

        for (int batch = 0; batch < trainBatches; batch++) {
            int remaining = trainSamples - batch * BATCH_SIZE;
            int N = remaining > BATCH_SIZE ? BATCH_SIZE : remaining;

            Tensor X    = TensorSlice(xShuffled, batch * BATCH_SIZE, N);
            Tensor Y    = TensorSlice(yShuffled, batch * BATCH_SIZE, N);
            Tensor Z    = ModelForward(X, &m);                // [N, SEQ_LENGTH, NUM_CLASSES]
            Tensor loss = TensorCrossEntropyLoss(Y, Z, 1.0f); // [] (Scalar)

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

            Tensor X           = TensorSlice(xTest, batch * BATCH_SIZE, N); // [N, SEQ_LENGTH]
            Tensor Y           = TensorSlice(yTest, batch * BATCH_SIZE, N); // [N, SEQ_LENGTH]
            Tensor Z           = ModelForward(X, &m);                       // [N, SEQ_LENGTH, NUM_CLASSES]
            Tensor predictions = TensorSample(Z, 0);                        // [N, SEQ_LENGTH]

            totalErrors  += TensorHammingDistance(predictions, Y);
            totalSamples += TensorSize(predictions);

            TensorFree(Z); TensorFree(predictions);
        }
        printf("Epoch %2d | Test Accuracy: %.2f%%\n", epoch, 100.0f * (totalSamples - totalErrors) / totalSamples);
        TensorArrayWrite(parameters, PARAMETERS);
        printf("Parameters saved to %s\n", PARAMETERS);
    }

    TensorArrayFree(parameters);
    TensorFree(xTrain); TensorFree(xTest);
    TensorFree(train);  TensorFree(test); TensorFree(data);
    return 0;
}
