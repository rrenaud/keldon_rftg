# RFTG AI Architecture Report

## Executive Summary

The Race for the Galaxy (RFTG) AI, implemented by Keldon Jones circa 2009-2011, uses two neural networks for decision-making: an **eval network** for state evaluation and a **role network** for action selection. This report analyzes the current implementation, its learning objectives, and proposes modern improvements using PyTorch.

---

## 1. Current Architecture

### 1.1 Network Structure

Both networks use identical two-layer feedforward architectures:

```
Input Layer → Hidden Layer (50 nodes, tanh) → Output Layer (softmax)
```

| Network | Inputs | Hidden | Outputs | Purpose |
|---------|--------|--------|---------|---------|
| **Eval** | 704 (2p base) | 50 | 2 | Win probability estimation |
| **Role** | 605 (2p base) | 50 | 7 | Action selection |

The input sizes vary by expansion and player count (up to ~1800 inputs for 6-player games with expansions).

### 1.2 Input Features

**Eval Network Inputs** (`ai.c:eval_game`):
- Game state: VP pool size, round indicators, game-over flag
- Per-card indicators: Which of ~200 cards are in hand, played, or in tableau
- Goal states: Active goals, claimed goals (expansion 1+)
- Relative position: Distance behind leader in VP, buildings, cards, goods

**Role Network Inputs** (`ai.c:predict_action_player`):
- Similar game state features
- Action-specific bonuses and powers available
- Opponent modeling features

### 1.3 How Decisions Are Made

#### Eval Network Usage
The eval network estimates P(win | game_state) for each player. It's called thousands of times per turn during **Monte Carlo tree search**:

```c
// Simplified decision flow
for each possible_action:
    simulate_action(game_copy, action)
    score = eval_game(game_copy, player)  // Neural net evaluation
    if score > best_score:
        best_action = action
```

The AI explores future game states by:
1. Simulating actions
2. Evaluating resulting positions with the neural net
3. Selecting actions that maximize expected win probability

#### Role Network Usage
The role network predicts which action (Explore, Develop, Settle, Trade, Consume, Produce) the AI or opponents will choose. This serves two purposes:

1. **Opponent Modeling**: Predict what opponents will do to narrow the search space
2. **Self-Prediction**: The network learns to predict its own "best" actions, acting as a fast policy approximation

---

## 2. Learning Objectives

### 2.1 Eval Network: TD(λ) Learning

The eval network uses **Temporal Difference learning with eligibility traces** (TD(λ), λ=0.7):

```c
// From ai.c:perform_training
double lambda = 1.0;

// Store current state
store_net(&eval, who);

// At game end, train backward through stored states
for (i = num_past - 2; i >= 0; i--) {
    // Load past state
    memcpy(eval.input_value, past_input[i], ...);
    compute_net(&eval);

    // Train toward outcome (or next state's prediction)
    train_net(&eval, lambda, target);

    // Decay eligibility trace
    lambda *= 0.7;
}
```

**Learning Signal**: Final game outcome (1.0 for winner, 0.0 for losers)

**Key Insight**: TD(λ) propagates the game result backward through all visited states, with exponentially decaying credit. States closer to the end receive stronger training signal.

### 2.2 Role Network: Policy Gradient (Softmax)

The role network uses a form of **policy gradient** training:

```c
// From ai.c:ai_choose_action_advanced (lines 4050-4068)

// Compute softmax over action scores (temperature=20/best_score)
for (i = 0; i < num_actions; i++) {
    sum += exp(20 * (scores[i] / best_score));
}

// Target distribution based on search results
for (i = 0; i < num_actions; i++) {
    desired[i] = exp(20 * (scores[i] / best_score)) / sum;
}

// Train network to predict this distribution
train_net(&role, 1.0, desired);
```

**Learning Signal**: Softmax distribution over action scores from tree search

**Key Insight**: The role network learns to imitate the tree search policy. Better actions (as determined by search) get higher target probabilities.

### 2.3 Gradient Computation

Both networks use standard backpropagation with:
- **Output layer**: Cross-entropy gradient (softmax derivative)
- **Hidden layer**: tanh derivative backpropagated through output weights
- **Learning rate**: α = 0.0001 (eval), α = 0.0005 (role)

```c
// From net.c:train_net
// Output gradient (softmax cross-entropy)
deriv = win_prob[i] * (1.0 - win_prob[i]);
corr = -error * hidden_result[j] * deriv;

// Hidden gradient (tanh derivative)
deriv = 1 - (hidden_result[i] * hidden_result[i]);
hidden_corr[i] = deriv * -hidden_error[i] * alpha;
```

---

## 3. Limitations of Current Approach

### 3.1 Shallow Network Architecture
- Only 50 hidden nodes limits representational capacity
- Single hidden layer cannot learn complex feature interactions
- No residual connections, attention, or normalization

### 3.2 Simple Features
- Hand-crafted binary/ternary input features
- No learned embeddings for cards
- Limited positional/relational information

### 3.3 Online Learning Instability
- Single-sample updates (batch size = 1)
- No experience replay
- No target network stabilization
- Learning rate fixed throughout training

### 3.4 Inefficient Search
- Full tree enumeration for small action spaces
- Monte Carlo sampling for larger spaces
- No learned search guidance (like MCTS with UCB)

---

## 4. Modern Improvements

### 4.1 Network Architecture

**Recommended: Residual MLP or Transformer**

```python
class ModernRFTGNet(nn.Module):
    def __init__(self, num_inputs, hidden_dim=256, num_layers=4, num_outputs=2):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(num_inputs, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])

        # Output heads
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_outputs)
        )

        self.policy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 7)  # 7 possible actions
        )

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.blocks:
            x = x + block(x)  # Residual connection

        value = F.softmax(self.value_head(x), dim=-1)
        policy = F.softmax(self.policy_head(x), dim=-1)

        return value, policy
```

**Benefits**:
- Unified network for both value and policy (parameter sharing)
- Deeper architecture with residual connections
- Layer normalization for training stability
- GELU activation (smoother gradients than tanh)

### 4.2 Card Embeddings

Replace binary card indicators with learned embeddings:

```python
class CardEmbeddingNet(nn.Module):
    def __init__(self, num_cards=200, embed_dim=32, ...):
        super().__init__()

        # Learned card embeddings
        self.card_embeddings = nn.Embedding(num_cards, embed_dim)

        # Attention over cards in hand/tableau
        self.hand_attention = nn.MultiheadAttention(embed_dim, num_heads=4)
        self.tableau_attention = nn.MultiheadAttention(embed_dim, num_heads=4)

    def encode_cards(self, card_ids, attention_layer):
        # card_ids: [batch, num_cards] indices of cards
        embeds = self.card_embeddings(card_ids)  # [batch, num_cards, embed_dim]
        attended, _ = attention_layer(embeds, embeds, embeds)
        return attended.mean(dim=1)  # Pool to fixed size
```

**Benefits**:
- Similar cards (e.g., military worlds) have similar representations
- Attention captures card synergies
- Generalizes better to unseen card combinations

### 4.3 Modern RL Algorithms

#### Option A: PPO (Proximal Policy Optimization)

```python
class PPOTrainer:
    def __init__(self, model, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def compute_loss(self, states, actions, old_log_probs, advantages, returns):
        values, policy = self.model(states)

        # Policy loss with clipping
        log_probs = torch.log(policy.gather(1, actions.unsqueeze(1)))
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus for exploration
        entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=-1).mean()

        return policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
```

#### Option B: AlphaZero-Style MCTS + Self-Play

```python
class AlphaZeroTrainer:
    """
    1. Use neural network to guide MCTS
    2. Train on (state, MCTS_policy, game_outcome) tuples
    """

    def mcts_search(self, game_state, num_simulations=100):
        root = MCTSNode(game_state)

        for _ in range(num_simulations):
            node = root

            # Selection: UCB with neural network prior
            while not node.is_leaf():
                node = node.select_child(self.model)

            # Expansion and evaluation
            value, policy = self.model(node.state)
            node.expand(policy)

            # Backpropagation
            node.backup(value)

        return root.get_policy()  # Visit counts → policy

    def train_step(self, replay_buffer):
        states, mcts_policies, outcomes = replay_buffer.sample(batch_size=256)

        values, policies = self.model(states)

        value_loss = F.mse_loss(values, outcomes)
        policy_loss = F.cross_entropy(policies, mcts_policies)

        return value_loss + policy_loss
```

### 4.4 Efficient Training Infrastructure

```python
class EfficientTrainingPipeline:
    """
    Modern training infrastructure for RFTG.
    """

    def __init__(self, model, num_workers=8):
        self.model = model

        # Experience replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=1_000_000)

        # Parallel game generation
        self.game_workers = [
            GameWorker(model.clone()) for _ in range(num_workers)
        ]

        # Target network for stability
        self.target_model = copy.deepcopy(model)
        self.target_update_freq = 1000

    def generate_games(self, num_games):
        """Generate games in parallel using current policy."""
        futures = []
        for worker in self.game_workers:
            futures.append(worker.play_games_async(num_games // len(self.game_workers)))

        experiences = []
        for future in futures:
            experiences.extend(future.result())

        # Add to replay buffer with priorities
        for exp in experiences:
            td_error = self.compute_td_error(exp)
            self.replay_buffer.add(exp, priority=abs(td_error))

        return len(experiences)

    def train_epoch(self, batch_size=512, steps=1000):
        """Train on replay buffer."""
        for step in range(steps):
            # Sample prioritized batch
            batch, weights, indices = self.replay_buffer.sample(batch_size)

            # Compute loss with importance sampling weights
            loss = self.compute_loss(batch, weights)

            # Update priorities
            new_priorities = self.compute_td_errors(batch)
            self.replay_buffer.update_priorities(indices, new_priorities)

            # Periodic target network update
            if step % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        return loss
```

---

## 5. Recommended Implementation Plan

### Phase 1: Baseline PyTorch Port (Complete)
- [x] Port network architecture to PyTorch
- [x] Load existing .net weights
- [x] Verify numerical equivalence
- [x] Export training data from C code
- [x] Train from self-play data

### Phase 2: Modern Architecture
- [ ] Implement residual MLP with value/policy heads
- [ ] Add card embeddings
- [ ] Implement proper batch training with replay buffer
- [ ] Add target network for stability

### Phase 3: Advanced RL
- [ ] Implement PPO or AlphaZero-style training
- [ ] Add MCTS with neural network guidance
- [ ] Parallel self-play with GPU inference
- [ ] Curriculum learning (start with simpler scenarios)

### Phase 4: Optimization
- [ ] Profile and optimize inference speed
- [ ] Quantization for faster evaluation
- [ ] Distributed training across multiple machines
- [ ] Integration with C game engine (PyTorch C++ API or ONNX)

---

## 6. Expected Improvements

| Aspect | Current | With Modern Approach |
|--------|---------|---------------------|
| Network depth | 2 layers | 4-8 layers |
| Hidden size | 50 | 256-512 |
| Training stability | Poor (online) | Good (replay + PPO) |
| Sample efficiency | Low | High (prioritized replay) |
| Feature learning | Manual | Automatic (embeddings) |
| Search guidance | Random/heuristic | Learned (MCTS+NN) |
| Parallelization | Limited | Full GPU + multi-worker |

Based on similar improvements in other games (e.g., AlphaGo → AlphaZero), we could expect:
- **2-3x stronger play** from architecture improvements alone
- **5-10x faster training** from modern infrastructure
- **Better generalization** to expansion cards and edge cases

---

## 7. Conclusion

The original RFTG AI was remarkably effective for its time, achieving strong play with minimal computational resources. The key innovations were:

1. **TD(λ) for value estimation** - Efficient credit assignment across game states
2. **Policy distillation** - Training the role network to imitate tree search
3. **Opponent modeling** - Using the same network to predict opponent actions

Modern improvements would focus on:

1. **Deeper networks** with residual connections and normalization
2. **Learned representations** for cards and game states
3. **Stable training** with experience replay and PPO/AlphaZero algorithms
4. **Efficient infrastructure** with parallel self-play and GPU acceleration

The existing codebase provides an excellent foundation for these improvements, with clean separation between game logic, AI decision-making, and neural network operations.
