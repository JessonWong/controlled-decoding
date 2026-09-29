import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]

# Keep this unit test offline and independent of optional model/data packages.
datasets_stub = types.ModuleType("datasets")
datasets_stub.load_dataset = None
transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoModelForCausalLM = object
transformers_stub.AutoTokenizer = object
tqdm_stub = types.ModuleType("tqdm")
tqdm_stub.tqdm = lambda iterable, *args, **kwargs: iterable
module_spec = importlib.util.spec_from_file_location(
    "_pre_logits_sampled_openweight_for_alignment_test",
    ROOT / "training" / "pre_logits_sampled_openweight.py",
)
module = importlib.util.module_from_spec(module_spec)
with mock.patch.dict(
    sys.modules,
    {
        "datasets": datasets_stub,
        "transformers": transformers_stub,
        "tqdm": tqdm_stub,
    },
):
    module_spec.loader.exec_module(module)
align_prompt_and_answer = module.align_prompt_and_answer


class FakeTokenizer:
    def __init__(self, prompt_ids, full_ids):
        self.prompt_ids = prompt_ids
        self.full_ids = full_ids
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_tensors,
    ):
        self.calls.append((messages, add_generation_prompt))
        if tokenize is not True or return_tensors != "pt":
            raise AssertionError("The chat template must return PyTorch token ids")
        if add_generation_prompt:
            expected = [{"role": "user", "content": "question"}]
            if messages != expected:
                raise AssertionError("Unexpected prompt messages")
            return torch.tensor([self.prompt_ids], dtype=torch.long)

        if len(messages) != 2:
            raise AssertionError("Expected a user/assistant conversation")
        if messages[0] != {"role": "user", "content": "question"}:
            raise AssertionError("Unexpected user message")
        if messages[1].get("role") != "assistant":
            raise AssertionError("Unexpected assistant message")
        return torch.tensor([self.full_ids], dtype=torch.long)


class AlignPromptAndAnswerTest(unittest.TestCase):
    def test_normal_prompt_is_preserved_as_strict_prefix(self):
        tokenizer = FakeTokenizer(
            prompt_ids=[1, 10, 20],
            full_ids=[1, 10, 20, 30, 31, 2],
        )

        full_ids, prompt_ids = align_prompt_and_answer(
            tokenizer, "question", "answer"
        )

        self.assertTrue(torch.equal(full_ids, torch.tensor([[1, 10, 20, 30, 31, 2]])))
        self.assertTrue(torch.equal(prompt_ids, torch.tensor([[1, 10, 20]])))
        self.assertEqual(tuple(full_ids.shape), (1, 6))
        self.assertEqual(tuple(prompt_ids.shape), (1, 3))
        self.assertTrue(torch.equal(prompt_ids, full_ids[:, : prompt_ids.shape[1]]))
        self.assertLess(prompt_ids.shape[1], full_ids.shape[1])

    def test_prompt_only_trailing_special_token_is_excluded(self):
        tokenizer = FakeTokenizer(
            prompt_ids=[1, 10, 20, 99],
            full_ids=[1, 10, 20, 30, 31, 2],
        )

        full_ids, prompt_ids = align_prompt_and_answer(
            tokenizer, "question", "answer"
        )

        self.assertTrue(torch.equal(prompt_ids, torch.tensor([[1, 10, 20]])))
        self.assertTrue(torch.equal(prompt_ids, full_ids[:, :3]))

    def test_no_nonempty_common_prefix_raises(self):
        tokenizer = FakeTokenizer(prompt_ids=[1, 10], full_ids=[9, 30, 2])

        with self.assertRaisesRegex(ValueError, "non-empty common prefix"):
            align_prompt_and_answer(tokenizer, "question", "answer")

    def test_empty_answer_raises_even_when_template_adds_boundary_token(self):
        tokenizer = FakeTokenizer(prompt_ids=[1, 10, 20], full_ids=[1, 10, 20, 2])

        with self.assertRaisesRegex(ValueError, "empty"):
            align_prompt_and_answer(tokenizer, "question", "")

    def test_full_conversation_without_answer_suffix_raises(self):
        tokenizer = FakeTokenizer(prompt_ids=[1, 10, 20], full_ids=[1, 10, 20])

        with self.assertRaisesRegex(ValueError, "strict prefix"):
            align_prompt_and_answer(tokenizer, "question", "answer")


if __name__ == "__main__":
    unittest.main()
