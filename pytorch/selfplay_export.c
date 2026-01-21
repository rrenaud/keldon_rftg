/*
 * Self-play data exporter for RFTG neural network training.
 *
 * This program plays games using the AI and exports training data
 * in JSON format for use with the PyTorch training pipeline.
 *
 * Output format: JSON Lines (one game per line)
 *
 * Compile:
 *   gcc -o selfplay_export selfplay_export.c \
 *       -I../rftg/include -I../net/include \
 *       -L../build/lib -lrftg -lnet -lm \
 *       -Wl,-rpath,/path/to/rftg/build/lib
 *
 * Usage:
 *   ./selfplay_export -n 100 -p 2 -e 0 > training_data.jsonl
 */

#include <rftg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* External references to AI networks */
extern net eval;
extern net role;

/* Game record data */
typedef struct {
    /* Game config */
    unsigned int random_seed;
    int expansion;
    int num_players;
    int advanced;

    /* Outcome */
    int winner_indices[MAX_PLAYER];
    int num_winners;
    int final_scores[MAX_PLAYER];

    /* Eval states - stored inputs */
    double **eval_inputs;
    int *eval_player;
    int *eval_round;
    int num_eval_states;
    int max_eval_states;

    /* Role decisions */
    double **role_inputs;
    int *role_player;
    int *role_round;
    int *role_chosen;
    double **role_scores;
    int *role_num_actions;
    int num_role_decisions;
    int max_role_decisions;

    /* Current round */
    int current_round;

} game_record;

/* Global game record */
static game_record g_record;

/* Worker ID for parallel execution */
static int worker_id = 0;

/* Suppress normal output */
static int quiet = 1;

/* Initialize game record */
static void init_game_record(int num_inputs_eval, int num_inputs_role)
{
    int i;

    g_record.num_eval_states = 0;
    g_record.max_eval_states = 500;  /* Typical game has ~100-300 states */
    g_record.eval_inputs = malloc(sizeof(double *) * g_record.max_eval_states);
    g_record.eval_player = malloc(sizeof(int) * g_record.max_eval_states);
    g_record.eval_round = malloc(sizeof(int) * g_record.max_eval_states);

    for (i = 0; i < g_record.max_eval_states; i++) {
        g_record.eval_inputs[i] = malloc(sizeof(double) * num_inputs_eval);
    }

    g_record.num_role_decisions = 0;
    g_record.max_role_decisions = 200;
    g_record.role_inputs = malloc(sizeof(double *) * g_record.max_role_decisions);
    g_record.role_player = malloc(sizeof(int) * g_record.max_role_decisions);
    g_record.role_round = malloc(sizeof(int) * g_record.max_role_decisions);
    g_record.role_chosen = malloc(sizeof(int) * g_record.max_role_decisions);
    g_record.role_scores = malloc(sizeof(double *) * g_record.max_role_decisions);
    g_record.role_num_actions = malloc(sizeof(int) * g_record.max_role_decisions);

    for (i = 0; i < g_record.max_role_decisions; i++) {
        g_record.role_inputs[i] = malloc(sizeof(double) * num_inputs_role);
        g_record.role_scores[i] = malloc(sizeof(double) * 100);  /* Max actions */
    }

    g_record.current_round = 0;
}

/* Reset game record for new game */
static void reset_game_record(void)
{
    g_record.num_eval_states = 0;
    g_record.num_role_decisions = 0;
    g_record.num_winners = 0;
    g_record.current_round = 0;
}

/* Store an eval state */
void record_eval_state(int player, double *inputs, int num_inputs)
{
    int idx = g_record.num_eval_states;

    if (idx >= g_record.max_eval_states) {
        /* Expand storage */
        int new_max = g_record.max_eval_states * 2;
        g_record.eval_inputs = realloc(g_record.eval_inputs, sizeof(double *) * new_max);
        g_record.eval_player = realloc(g_record.eval_player, sizeof(int) * new_max);
        g_record.eval_round = realloc(g_record.eval_round, sizeof(int) * new_max);

        for (int i = g_record.max_eval_states; i < new_max; i++) {
            g_record.eval_inputs[i] = malloc(sizeof(double) * num_inputs);
        }
        g_record.max_eval_states = new_max;
    }

    memcpy(g_record.eval_inputs[idx], inputs, sizeof(double) * num_inputs);
    g_record.eval_player[idx] = player;
    g_record.eval_round[idx] = g_record.current_round;
    g_record.num_eval_states++;
}

/* Store a role decision */
void record_role_decision(int player, double *inputs, int num_inputs,
                          int chosen, double *scores, int num_actions)
{
    int idx = g_record.num_role_decisions;

    if (idx >= g_record.max_role_decisions) {
        /* Expand storage */
        int new_max = g_record.max_role_decisions * 2;
        g_record.role_inputs = realloc(g_record.role_inputs, sizeof(double *) * new_max);
        g_record.role_player = realloc(g_record.role_player, sizeof(int) * new_max);
        g_record.role_round = realloc(g_record.role_round, sizeof(int) * new_max);
        g_record.role_chosen = realloc(g_record.role_chosen, sizeof(int) * new_max);
        g_record.role_scores = realloc(g_record.role_scores, sizeof(double *) * new_max);
        g_record.role_num_actions = realloc(g_record.role_num_actions, sizeof(int) * new_max);

        for (int i = g_record.max_role_decisions; i < new_max; i++) {
            g_record.role_inputs[i] = malloc(sizeof(double) * num_inputs);
            g_record.role_scores[i] = malloc(sizeof(double) * 100);
        }
        g_record.max_role_decisions = new_max;
    }

    memcpy(g_record.role_inputs[idx], inputs, sizeof(double) * num_inputs);
    memcpy(g_record.role_scores[idx], scores, sizeof(double) * num_actions);
    g_record.role_player[idx] = player;
    g_record.role_round[idx] = g_record.current_round;
    g_record.role_chosen[idx] = chosen;
    g_record.role_num_actions[idx] = num_actions;
    g_record.num_role_decisions++;
}

/* Print a double array as JSON */
static void print_double_array(FILE *f, double *arr, int len)
{
    fprintf(f, "[");
    for (int i = 0; i < len; i++) {
        fprintf(f, "%.9g", arr[i]);
        if (i < len - 1) fprintf(f, ",");
    }
    fprintf(f, "]");
}

/* Print an int array as JSON */
static void print_int_array(FILE *f, int *arr, int len)
{
    fprintf(f, "[");
    for (int i = 0; i < len; i++) {
        fprintf(f, "%d", arr[i]);
        if (i < len - 1) fprintf(f, ",");
    }
    fprintf(f, "]");
}

/* Output game record as JSON */
static void output_game_record(FILE *f)
{
    int i;

    fprintf(f, "{");

    /* Game ID */
    fprintf(f, "\"game_id\":\"%d_%u_%ld\",", worker_id, g_record.random_seed, time(NULL));

    /* Config */
    fprintf(f, "\"expansion\":%d,", g_record.expansion);
    fprintf(f, "\"num_players\":%d,", g_record.num_players);
    fprintf(f, "\"advanced\":%s,", g_record.advanced ? "true" : "false");
    fprintf(f, "\"random_seed\":%u,", g_record.random_seed);

    /* Outcome */
    fprintf(f, "\"winner_indices\":");
    print_int_array(f, g_record.winner_indices, g_record.num_winners);
    fprintf(f, ",");

    fprintf(f, "\"final_scores\":");
    print_int_array(f, g_record.final_scores, g_record.num_players);
    fprintf(f, ",");

    fprintf(f, "\"num_rounds\":%d,", g_record.current_round);

    /* Eval states */
    fprintf(f, "\"eval_states\":[");
    for (i = 0; i < g_record.num_eval_states; i++) {
        fprintf(f, "{\"player_index\":%d,\"round_num\":%d,\"inputs\":",
                g_record.eval_player[i], g_record.eval_round[i]);
        print_double_array(f, g_record.eval_inputs[i], eval.num_inputs);
        fprintf(f, "}");
        if (i < g_record.num_eval_states - 1) fprintf(f, ",");
    }
    fprintf(f, "],");

    /* Role decisions */
    fprintf(f, "\"role_decisions\":[");
    for (i = 0; i < g_record.num_role_decisions; i++) {
        fprintf(f, "{\"player_index\":%d,\"round_num\":%d,\"chosen_action\":%d,\"inputs\":",
                g_record.role_player[i], g_record.role_round[i], g_record.role_chosen[i]);
        print_double_array(f, g_record.role_inputs[i], role.num_inputs);
        fprintf(f, ",\"action_scores\":");
        print_double_array(f, g_record.role_scores[i], g_record.role_num_actions[i]);
        fprintf(f, "}");
        if (i < g_record.num_role_decisions - 1) fprintf(f, ",");
    }
    fprintf(f, "]");

    fprintf(f, "}\n");
    fflush(f);
}

/* Message handlers - suppress output in quiet mode */
void display_error(char *msg)
{
    if (!quiet) fprintf(stderr, "%s", msg);
}

void message_add(game *g, char *msg)
{
    if (!quiet) printf("%s", msg);
}

void message_add_formatted(game *g, char *msg, char *tag)
{
    message_add(g, msg);
}

int game_rand(game *g)
{
    return simple_rand(&g->random_seed);
}

/*
 * Custom game over handler that records outcome and outputs data.
 */
static void export_game_over(game *g, int who)
{
    int i, max_score = 0;

    /* Only process once (when called for player 0) */
    if (who != 0) return;

    /* Record final scores */
    for (i = 0; i < g->num_players; i++) {
        g_record.final_scores[i] = g->p[i].end_vp;
        if (g->p[i].end_vp > max_score) {
            max_score = g->p[i].end_vp;
        }
    }

    /* Find winners */
    g_record.num_winners = 0;
    for (i = 0; i < g->num_players; i++) {
        if (g->p[i].winner) {
            g_record.winner_indices[g_record.num_winners++] = i;
        }
    }

    /* Output the game record */
    output_game_record(stdout);
}

/*
 * Main entry point.
 */
int main(int argc, char *argv[])
{
    game my_game;
    int i, j, n = 100;
    int num_players = 2;
    int expansion = 0, advanced = 0, promo = 0;
    char buf[1024], *names[MAX_PLAYER];
    double factor = 0.0;  /* No training, just data collection */

    /* Set random seed */
    my_game.random_seed = time(NULL);

    /* Read card database */
    if (read_cards(NULL) < 0) {
        fprintf(stderr, "Failed to read card database\n");
        exit(1);
    }

    /* Parse arguments */
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-v")) {
            quiet = 0;
        }
        else if (!strcmp(argv[i], "-p")) {
            num_players = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-a")) {
            advanced = 1;
        }
        else if (!strcmp(argv[i], "-e")) {
            expansion = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-o")) {
            promo = 1;
        }
        else if (!strcmp(argv[i], "-n")) {
            n = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-r")) {
            my_game.random_seed = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-w")) {
            worker_id = atoi(argv[++i]);
        }
    }

    /* Store config in record */
    g_record.expansion = expansion;
    g_record.num_players = num_players;
    g_record.advanced = advanced;

    /* Set up game */
    my_game.num_players = num_players;
    my_game.expanded = expansion;
    my_game.advanced = advanced;
    my_game.promo = promo;
    my_game.goal_disabled = 0;
    my_game.takeover_disabled = 0;
    my_game.camp = NULL;

    /* Initialize players */
    for (i = 0; i < num_players; i++) {
        sprintf(buf, "Player %d", i);
        my_game.p[i].name = strdup(buf);
        names[i] = my_game.p[i].name;
        my_game.p[i].control = &ai_func;
        my_game.p[i].control->init(&my_game, i, factor);
        my_game.p[i].choice_log = malloc(sizeof(int) * 4096);
        my_game.p[i].choice_size = 0;
        my_game.p[i].choice_pos = 0;
    }

    /* Initialize game record storage */
    init_game_record(eval.num_inputs, role.num_inputs);

    /* Play games */
    for (i = 0; i < n; i++) {
        /* Reset record */
        reset_game_record();

        /* Initialize game */
        init_game(&my_game);
        g_record.random_seed = my_game.start_seed;

        /* Begin game */
        begin_game(&my_game);

        /* Play rounds */
        while (game_round(&my_game)) {
            g_record.current_round++;
        }

        /* Score game */
        score_game(&my_game);

        /* Declare winner */
        declare_winner(&my_game);

        /* Export game data */
        export_game_over(&my_game, 0);

        /* Call player game over functions (for cleanup) */
        for (j = 0; j < num_players; j++) {
            my_game.p[j].choice_size = 0;
            my_game.p[j].choice_pos = 0;
        }

        /* Reset player names */
        for (j = 0; j < num_players; j++) {
            my_game.p[j].name = names[j];
        }

        /* Progress indicator */
        if (!quiet && (i + 1) % 10 == 0) {
            fprintf(stderr, "Completed %d/%d games\n", i + 1, n);
        }
    }

    /* Shutdown */
    for (i = 0; i < num_players; i++) {
        my_game.p[i].control->shutdown(&my_game, i);
    }

    return 0;
}
