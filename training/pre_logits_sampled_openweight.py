import argparse
import hashlib
import os
import sys
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Monte Carlo next-token log-prob estimates for BiasNet training."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Path to the local/open-weight causal LM or Hugging Face model id.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cached_logits/llama32_1b_inst_mc100_floor",
        help="Directory where .pt cache files will be saved.",
    )
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--samples_per_token", type=int, default=100)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--observed_alpha", type=float, default=0.1)
    parser.add_argument("--floor_mass", type=float, default=1e-4)
    parser.add_argument(
        "--max_answer_tokens",
        type=int,
        default=None,
        help="Optional cap on target answer tokens, useful for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="LLM-LAT/harmful-dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--start_index",
        type=int,
        default=100,
        help="Dataset index to start from. The full-logprob script starts at 100.",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--store_dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Stored log-prob tensor dtype.",
    )
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--risk_gate_checkpoint",
        type=str,
        default=None,
        help="Optional prefix-risk checkpoint. When set, cache a per-token mask for risk-gated training.",
    )
    parser.add_argument(
        "--risk_gate_threshold",
        type=float,
        default=0.1,
        help="Risk scores below this threshold are marked active for BiasNet training.",
    )
    parser.add_argument(
        "--risk_gate_batch_size",
        type=int,
        default=16,
        help="Batch size used while scoring teacher-forced prefixes with the risk head.",
    )
    parser.add_argument(
        "--risk_gate_max_length",
        type=int,
        default=None,
        help="Optional max token length override for the risk-head tokenizer.",
    )
    parser.add_argument(
        "--risk_gate_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Backbone dtype used by the risk gate.",
    )
    parser.add_argument(
        "--risk_gate_device",
        type=str,
        default=None,
        help="Device for the risk gate. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--risk_gate_model_name",
        type=str,
        default=None,
        help="Optional backbone model override for the risk gate.",
    )
    parser.add_argument("--risk_gate_load_in_4bit", action="store_true")
    parser.add_argument("--risk_gate_trust_remote_code", action="store_true")
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def load_risk_gate(args):
    if args.risk_gate_checkpoint is None:
        return None
    if args.risk_gate_batch_size <= 0:
        raise ValueError("--risk_gate_batch_size must be positive.")

    from risk_gate import PrefixRiskGate

    device = torch.device(
        args.risk_gate_device
        if args.risk_gate_device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return PrefixRiskGate(
        checkpoint=args.risk_gate_checkpoint,
        device=device,
        threshold=args.risk_gate_threshold,
        top_k=1,
        batch_size=args.risk_gate_batch_size,
        max_length=args.risk_gate_max_length,
        dtype=args.risk_gate_dtype,
        model_name=args.risk_gate_model_name,
        load_in_4bit=args.risk_gate_load_in_4bit,
        trust_remote_code=args.risk_gate_trust_remote_code,
    )


def validate_existing_risk_gate_cache(output_dir: str, filenames: list[str]) -> None:
    missing: list[str] = []
    for filename in filenames:
        payload = torch.load(os.path.join(output_dir, filename), map_location="cpu")
        if "risk_gate_mask" not in payload:
            missing.append(filename)
    if missing:
        preview = ", ".join(missing[:3])
        if len(missing) > 3:
            preview += ", ..."
        raise ValueError(
            "Existing cache files do not contain risk_gate_mask "
            f"({preview}). Use a fresh --output_dir or regenerate the cache before "
            "risk-gated training."
        )


def align_prompt_and_answer(tokenizer, question: str, answer: str) -> tuple[torch.Tensor, torch.Tensor]:
    def as_single_row_token_ids(value, source: str) -> torch.Tensor:
        if isinstance(value, dict):
            value = value.get("input_ids")
        try:
            token_ids = torch.as_tensor(value, dtype=torch.long)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} chat template did not return token ids") from exc

        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError(
                f"{source} chat template must produce one token sequence, "
                f"got shape {tuple(token_ids.shape)}"
            )
        return token_ids

    user_message = {"role": "user", "content": question}
    prompt_ids = as_single_row_token_ids(
        tokenizer.apply_chat_template(
            [user_message],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ),
        "Prompt",
    )
    full_ids = as_single_row_token_ids(
        tokenizer.apply_chat_template(
            [
                user_message,
                {"role": "assistant", "content": answer},
            ],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ),
        "Full conversation",
    )

    common_prefix_length = 0
    comparison_length = min(prompt_ids.shape[1], full_ids.shape[1])
    while (
        common_prefix_length < comparison_length
        and prompt_ids[0, common_prefix_length]
        == full_ids[0, common_prefix_length]
    ):
        common_prefix_length += 1

    if not answer:
        raise ValueError("The assistant answer is empty and contributes no answer tokens")
    if common_prefix_length == 0:
        raise ValueError("Prompt and full conversation have no non-empty common prefix")
    if common_prefix_length >= full_ids.shape[1]:
        raise ValueError(
            "The aligned prompt is not a strict prefix of the full conversation; "
            "the full conversation has no answer tokens"
        )

    aligned_prompt_ids = full_ids[:, :common_prefix_length]
    return full_ids, aligned_prompt_ids


def sample_token_ids(
    logits: torch.Tensor,
    samples_per_token: int,
    temperature: float,
    top_p: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if samples_per_token <= 0:
        raise ValueError("--samples_per_token must be positive.")
    if temperature <= 0:
        raise ValueError("--sample_temperature must be positive for Monte Carlo sampling.")
    if top_p <= 0 or top_p > 1:
        raise ValueError("--top_p must be in the interval (0, 1].")

    scaled_logits = logits.float() / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        probs = torch.softmax(sorted_logits, dim=-1)
        sampled_sorted = torch.multinomial(
            probs,
            num_samples=samples_per_token,
            replacement=True,
            generator=generator,
        )
        return sorted_indices.gather(dim=-1, index=sampled_sorted)

    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(
        probs,
        num_samples=samples_per_token,
        replacement=True,
        generator=generator,
    )


def sampled_ids_to_log_probs(
    sampled_token_ids: torch.Tensor,
    vocab_size: int,
    observed_alpha: float = 0.1,
    floor_mass: float = 1e-4,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    if sampled_token_ids.dim() != 2:
        raise ValueError("sampled_token_ids must have shape [seq_len, samples_per_token].")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1.")
    if observed_alpha < 0:
        raise ValueError("--observed_alpha must be non-negative.")
    if floor_mass < 0 or floor_mass >= 1:
        raise ValueError("--floor_mass must be in the interval [0, 1).")

    seq_len, samples_per_token = sampled_token_ids.shape
    rows = torch.empty((seq_len, vocab_size), dtype=torch.float32)
    for row_idx in range(seq_len):
        row_samples = sampled_token_ids[row_idx].detach().cpu().long()
        unique_ids, counts = torch.unique(row_samples, sorted=False, return_counts=True)
        observed_count = unique_ids.numel()
        unseen_count = vocab_size - observed_count
        if unseen_count <= 0:
            unseen_log_prob = -torch.inf
            observed_mass = 1.0
        else:
            unseen_log_prob = torch.log(torch.tensor(floor_mass / unseen_count, dtype=torch.float32))
            observed_mass = 1.0 - floor_mass

        rows[row_idx].fill_(unseen_log_prob)
        denom = float(samples_per_token) + observed_alpha * float(observed_count)
        observed_probs = observed_mass * (counts.float() + observed_alpha) / denom
        rows[row_idx, unique_ids] = observed_probs.log()

    return rows.to(dtype)


@torch.no_grad()
def build_risk_gate_payload(
    risk_gate,
    tokenizer,
    question: str,
    answer_ids: torch.Tensor,
    base_token_ids: torch.Tensor,
) -> dict:
    if risk_gate is None:
        return {}
    if answer_ids.dim() != 2 or answer_ids.shape[0] != 1:
        raise ValueError("answer_ids must have shape [1, seq_len].")
    if base_token_ids.dim() != 1:
        raise ValueError("base_token_ids must have shape [seq_len].")
    if answer_ids.shape[1] != base_token_ids.shape[0]:
        raise ValueError("answer_ids and base_token_ids must have matching sequence lengths.")

    target_ids = answer_ids[0].detach().cpu().long().tolist()
    current_ids = base_token_ids.detach().cpu().long().tolist()
    prompts = [question] * len(current_ids)
    answer_prefixes = []
    for position, token_id in enumerate(current_ids):
        prefix_ids = target_ids[:position] + [int(token_id)]
        answer_prefixes.append(tokenizer.decode(prefix_ids, skip_special_tokens=False))

    scores = risk_gate.score_prefixes(prompts, answer_prefixes)
    mask = scores < risk_gate.threshold
    return {
        "risk_gate_mask": mask.unsqueeze(0).cpu().bool(),
        "risk_gate_scores": scores.unsqueeze(0).cpu().float(),
        "risk_gate_token_ids": base_token_ids.unsqueeze(0).detach().cpu().long(),
    }


@torch.no_grad()
def get_sampled_logprobs(
    model,
    tokenizer,
    question: str,
    answer: str,
    args,
    generator,
    risk_gate=None,
) -> dict:
    device = model_device(model)
    input_ids, question_ids = align_prompt_and_answer(tokenizer, question, answer)
    input_ids = input_ids.to(device)
    question_ids = question_ids.to(device)

    answer_ids = input_ids[:, question_ids.shape[1] :]
    if args.max_answer_tokens is not None:
        answer_len = min(args.max_answer_tokens, answer_ids.shape[1])
        if answer_len <= 0:
            return None
        input_ids = input_ids[:, : question_ids.shape[1] + answer_len]
        answer_ids = answer_ids[:, :answer_len]

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, question_ids.shape[1] - 1 : input_ids.shape[1] - 1, :][0]
    if answer_ids.shape[1] != logits.shape[0]:
        raise ValueError("Answer token count does not match sampled logits positions.")

    sampled_ids = sample_token_ids(
        logits=logits,
        samples_per_token=args.samples_per_token,
        temperature=args.sample_temperature,
        top_p=args.top_p,
        generator=generator,
    )
    store_dtype = resolve_torch_dtype(args.store_dtype)
    sampled_log_probs = sampled_ids_to_log_probs(
        sampled_token_ids=sampled_ids,
        vocab_size=logits.shape[-1],
        observed_alpha=args.observed_alpha,
        floor_mass=args.floor_mass,
        dtype=store_dtype,
    )
    base_token_ids = logits.argmax(dim=-1)
    result = {
        "log_probs": sampled_log_probs.unsqueeze(0).cpu(),
        "labels": answer_ids.cpu().long(),
        "metadata": {
            "samples_per_token": args.samples_per_token,
            "sample_temperature": args.sample_temperature,
            "top_p": args.top_p,
            "observed_alpha": args.observed_alpha,
            "floor_mass": args.floor_mass,
            "source": "sampled_openweight",
        },
    }
    result.update(
        build_risk_gate_payload(
            risk_gate=risk_gate,
            tokenizer=tokenizer,
            question=question,
            answer_ids=answer_ids,
            base_token_ids=base_token_ids,
        )
    )
    if risk_gate is not None:
        result["metadata"].update(
            {
                "risk_gate_checkpoint": args.risk_gate_checkpoint,
                "risk_gate_threshold": args.risk_gate_threshold,
                "risk_gate_token_source": "base_argmax",
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive.")
    if args.max_answer_tokens is not None and args.max_answer_tokens <= 0:
        raise ValueError("--max_answer_tokens must be positive when provided.")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    device = model_device(model)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    risk_gate = load_risk_gate(args)

    dataset = load_dataset(args.dataset_name)
    train_data = dataset[args.dataset_split]
    existing_files = [name for name in os.listdir(args.output_dir) if name.endswith(".pt")]
    if args.risk_gate_checkpoint is not None and existing_files:
        validate_existing_risk_gate_cache(args.output_dir, existing_files)
    saved_count = len(existing_files)
    if saved_count >= args.max_samples:
        print(f"Output dir already has {saved_count} cache files; nothing to do.")
        return

    total_candidates = max(args.max_samples - saved_count, 0)
    progress = tqdm(total=total_candidates, desc="Sampling log-prob caches")
    for dataset_idx in range(args.start_index, len(train_data)):
        if saved_count >= args.max_samples:
            break
        item = train_data[dataset_idx]
        question = item["prompt"]
        answer = item["rejected"]
        data_hash = hashlib.md5((question + answer).encode("utf-8")).hexdigest()
        output_path = os.path.join(args.output_dir, f"{data_hash}.pt")
        if os.path.exists(output_path):
            continue

        result = get_sampled_logprobs(
            model,
            tokenizer,
            question,
            answer,
            args,
            generator,
            risk_gate=risk_gate,
        )
        if result is None:
            continue
        torch.save(result, output_path)
        saved_count += 1
        progress.update(1)

    progress.close()
    if saved_count < args.max_samples:
        print(
            f"Warning: only saved {saved_count} files before the dataset ended.",
            file=sys.stderr,
        )
    print(f"Processed {saved_count} samples. Sampled logits saved in {args.output_dir}.")


if __name__ == "__main__":
    main()
