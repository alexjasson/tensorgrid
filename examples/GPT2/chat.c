#include <stdio.h>
#include <string.h>
#include "Model.h"

#define TEMPERATURE 0.8f

int main(void)
{
    TensorArray parameters = TensorArrayNew();
    Model m = ModelNew(parameters);
    TensorArrayRead(parameters, PARAMETERS); // parameters will be random if they don't exist
    int context[SEQ_LENGTH] = {0};
    char line[1024];

    printf("You: ");
    fflush(stdout);
    while (fgets(line, sizeof(line), stdin)) {
        int len = (int)strlen(line);
        Tensor bytesIn = TensorNew(1, (int[]){len}, FLOAT32, 0);
        for (int i = 0; i < len; i++) TensorSet(bytesIn, i, (Real)(unsigned char)line[i]);
        Tensor tokensIn = TensorBPEEncode(bytesIn, m.merges);
        for (int i = 0; i < TensorShape(tokensIn)[0]; i++) {
            for (int j = 0; j < SEQ_LENGTH - 1; j++) context[j] = context[j + 1];
            context[SEQ_LENGTH - 1] = (int)TensorGet(tokensIn, i);
        }
        TensorFree(bytesIn); TensorFree(tokensIn);

        printf("Model: ");
        while (TRUE) {
            int next = ModelPredict(&m, context, TEMPERATURE);
            for (int i = 0; i < SEQ_LENGTH - 1; i++) context[i] = context[i + 1];
            context[SEQ_LENGTH - 1] = next;

            Tensor tokenOut = TensorNew(1, (int[]){1}, FLOAT32, 0);
            TensorSet(tokenOut, 0, (Real)next);
            Tensor bytesOut = TensorBPEDecode(tokenOut, m.merges);
            int newline = 0;
            for (int i = 0; i < TensorShape(bytesOut)[0]; i++) {
                char c = (char)(int)TensorGet(bytesOut, i);
                if (c == '\n') { newline = 1; break; }
                putchar(c);
            }
            TensorFree(tokenOut); TensorFree(bytesOut);
            if (newline) break;
        }
        printf("\nYou: ");
        fflush(stdout);
    }
    printf("\n");

    TensorArrayFree(parameters);
    return 0;
}
