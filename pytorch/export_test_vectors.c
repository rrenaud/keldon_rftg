/*
 * Export test vectors from the C neural network implementation
 * for regression testing against the PyTorch implementation.
 *
 * Outputs JSON format that can be easily parsed by Python.
 *
 * Usage: export_test_vectors <network.net> [num_random_tests]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "net.h"

/* Print a double array as JSON */
void print_double_array(double *arr, int len) {
    printf("[");
    for (int i = 0; i < len; i++) {
        printf("%.15e", arr[i]);
        if (i < len - 1) printf(", ");
    }
    printf("]");
}

/* Run a test case and output results */
void run_test(net *learner, double *inputs, int test_num, int is_last) {
    /* Copy inputs to network */
    memcpy(learner->input_value, inputs, sizeof(double) * learner->num_inputs);
    learner->input_value[learner->num_inputs] = 1.0;  /* bias */

    /* Reset hidden sums and previous inputs to force full recomputation */
    memset(learner->hidden_sum, 0, sizeof(double) * learner->num_hidden);
    memset(learner->prev_input, 0, sizeof(double) * (learner->num_inputs + 1));

    /* Compute network */
    compute_net(learner);

    /* Output test case */
    printf("    {\n");
    printf("      \"test_id\": %d,\n", test_num);
    printf("      \"inputs\": ");
    print_double_array(inputs, learner->num_inputs);
    printf(",\n");
    printf("      \"hidden_result\": ");
    print_double_array(learner->hidden_result, learner->num_hidden);
    printf(",\n");
    printf("      \"net_result\": ");
    print_double_array(learner->net_result, learner->num_output);
    printf(",\n");
    printf("      \"win_prob\": ");
    print_double_array(learner->win_prob, learner->num_output);
    printf("\n");
    printf("    }%s\n", is_last ? "" : ",");
}

int main(int argc, char *argv[])
{
    net learner;
    FILE *fff;
    int input, hidden, output;
    int i;
    char buf[1024];
    int num_random = 10;
    int test_num = 0;
    int total_tests;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <network.net> [num_random_tests]\n", argv[0]);
        return 1;
    }

    if (argc >= 3) {
        num_random = atoi(argv[2]);
    }

    /* Read network dimensions */
    fff = fopen(argv[1], "r");
    if (!fff) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 1;
    }
    fgets(buf, 1024, fff);
    fclose(fff);

    sscanf(buf, "%d %d %d", &input, &hidden, &output);

    /* Create and load network */
    make_learner(&learner, input, hidden, output);
    if (load_net(&learner, argv[1]) != 0) {
        fprintf(stderr, "Failed to load network\n");
        return 1;
    }

    /* Seed random number generator */
    srand(42);  /* Fixed seed for reproducibility */

    /* Allocate input buffer */
    double *inputs = (double *)malloc(sizeof(double) * input);

    /* Calculate total number of tests */
    /* 4 deterministic tests + num_random random tests */
    total_tests = 4 + num_random;

    /* Output JSON header */
    printf("{\n");
    printf("  \"network_file\": \"%s\",\n", argv[1]);
    printf("  \"num_inputs\": %d,\n", input);
    printf("  \"num_hidden\": %d,\n", hidden);
    printf("  \"num_outputs\": %d,\n", output);
    printf("  \"num_training\": %d,\n", learner.num_training);
    printf("  \"test_cases\": [\n");

    /* Test 1: All zeros */
    for (i = 0; i < input; i++) inputs[i] = 0.0;
    run_test(&learner, inputs, test_num++, 0);

    /* Test 2: All ones */
    for (i = 0; i < input; i++) inputs[i] = 1.0;
    run_test(&learner, inputs, test_num++, 0);

    /* Test 3: All negative ones (like dumpnet) */
    for (i = 0; i < input; i++) inputs[i] = -1.0;
    run_test(&learner, inputs, test_num++, 0);

    /* Test 4: Alternating 0 and 1 */
    for (i = 0; i < input; i++) inputs[i] = (i % 2);
    run_test(&learner, inputs, test_num++, num_random == 0 ? 1 : 0);

    /* Random tests */
    for (int r = 0; r < num_random; r++) {
        for (i = 0; i < input; i++) {
            /* Random value in range [-1, 1] */
            inputs[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        }
        run_test(&learner, inputs, test_num++, r == num_random - 1 ? 1 : 0);
    }

    printf("  ]\n");
    printf("}\n");

    free(inputs);
    free_net(&learner);

    return 0;
}
