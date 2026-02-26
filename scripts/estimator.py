import os
import torch
import torch.nn.functional as F


class BaseEstimator:
    """Default estimator."""
    def __repr__(self):
        return "Base"

    def __call__(
        self,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        logits: torch.Tensor,
    ) -> float:
        return 0.0

    @property
    def higher_is_better(self) -> bool:
        return True


class RandomEstimator(BaseEstimator):
    def __init__(self):
        self._rng = torch.Generator()

    def __repr__(self):
        return "Random"
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        logits: torch.Tensor,
    ) -> float:
        seed = int.from_bytes(os.urandom(8), "little")
        self._rng.manual_seed(seed)
        return float(torch.rand(1, generator=self._rng, device=logits.device).detach().cpu())


class MSP(BaseEstimator):
    """
    Calculates the uncertainty score using Maximum Sequence Probability (MSP).

    This method computes the total log-probability of a given sequence (`input_ids`)
    based on the model's `logits`. It's a common baseline for uncertainty
    estimation, where a higher score indicates higher model confidence.

    The score is calculated by summing the log-softmax probabilities of the
    actual tokens in the input sequence.
    """
    def __repr__(self):
        return "MSP"

    def __call__(
        self,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        logits: torch.Tensor,
    ) -> float:
        mask = torch.isfinite(logprobs)
        valid_logprobs = logprobs[mask]
        if valid_logprobs.numel() == 0:
            return -1e10
        return float(valid_logprobs.sum().detach().cpu())

    @property
    def higher_is_better(self) -> bool:
        return True
    
class MTE(BaseEstimator):
    def __repr__(self):
        return "MTE"
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        logits: torch.Tensor,
    ) -> float:
        """Mean Token Entropy"""
        if logits.dim() == 3:
            logits = logits.squeeze(0)

        mask = torch.isfinite(logprobs)
        if mask.sum() == 0:
            return float('inf')

        all_log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(all_log_probs)
        
        token_entropies = -torch.sum(
            torch.where(
                probs > 0,
                probs * all_log_probs,
                torch.zeros_like(probs)
            ),
            dim=-1
        )

        return float(token_entropies.mean().detach().cpu())
    
    @property
    def higher_is_better(self) -> bool:
        return False


class Perplexity(BaseEstimator):
    def __call__(
        self,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.isfinite(logprobs)
        valid_logprobs = logprobs[mask]
        
        if valid_logprobs.numel() == 0:
            return float('inf')
        
        avg_neg_logprob = -valid_logprobs.mean()
        perplexity = torch.exp(avg_neg_logprob)
        
        return float(perplexity.detach().cpu())

    @property
    def higher_is_better(self) -> bool:
        return False

ESTIMATOR_MAP = {
    "None": BaseEstimator(),
    "Random": RandomEstimator(),
    "MSP": MSP(),
    "MTE": MTE(),
    "PPL": Perplexity(), 
}
