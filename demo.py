"""Quick demo: verify both baselines work on CPU."""

import torch
from dp_rag_eval.wrappers.koga_imports import LDGumbelMechanism, ExpMechanism, majority_vote
from dp_rag_eval.wrappers.grislain_imports import DPGenerationConfig, DPLogitsAggregator

print("=== Koga: majority vote (no DP) ===")
tokens = torch.tensor([42, 42, 42, 7, 7, 99])
token, votes = majority_vote(tokens, dim=100)
print(f"Tokens: {tokens.tolist()}")
print(f"Winner: {token.item()}, votes: {votes}")

print()
print("=== Koga: majority vote (with DP, high eps = low noise) ===")
engine = LDGumbelMechanism(eps=100.0, delta=1e-5)
token, votes = majority_vote(tokens, dim=100, dp_engine=engine)
print(f"Winner: {token.item()}")
print(f"Privacy spent: eps={engine.get_dp_expense()[0]:.4f}, delta={engine.get_dp_expense()[1]:.2e}")

print()
print("=== Koga: exponential mechanism ===")
scores = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
mech = ExpMechanism(eps=50.0, target_eps=100.0, target_delta=1e-5)
top2 = mech.get_topk(scores, k=2)
print(f"Scores: {scores.tolist()}")
print(f"Top-2 indices: {top2.tolist()}")

print()
print("=== Grislain: token epsilon computation ===")
config = DPGenerationConfig(epsilon=2.0, delta=1e-3, max_new_tokens=50)
print(f"Total epsilon: {config.epsilon}")
print(f"Per-token epsilon: {config.token_epsilon():.4f}")

print()
print("=== Grislain: DP logit aggregation ===")
aggregator = DPLogitsAggregator(config)
scores = torch.randn(4, 500)  # 1 public prior + 3 document-conditioned
input_ids = torch.zeros(4, 5, dtype=torch.long)
result = aggregator(input_ids, scores)
print(f"Input:  {scores.shape[0]} prompts x {scores.shape[1]} vocab")
print(f"Output: {result.shape[0]} aggregated x {result.shape[1]} vocab")

print()
print("All good.")
