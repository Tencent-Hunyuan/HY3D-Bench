"""CPU attention regressions; no checkpoints or optional model dependencies required."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "baselines/core/models/autoencoders/attention_processors.py"
)
spec = importlib.util.spec_from_file_location("attention_processors", MODULE_PATH)
processors = importlib.util.module_from_spec(spec)
# Exercise the standard PyTorch backend even when SageAttention is enabled locally.
with patch.dict("os.environ", {"CA_USE_SAGEATTN": "0"}):
    spec.loader.exec_module(processors)


class FlashVDMMergeAttentionTests(unittest.TestCase):
    def make_inputs(self, batch=1, length=101):
        k = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, 2, 1, 1)
        v = torch.arange(batch * 2 * 3 * 3, dtype=torch.float32).reshape(batch, 2, 3, 3)
        q = torch.zeros(batch, 2, length, 3)
        q[..., 0] = 40
        return q, k, v

    def test_coarse_call_accepts_stride_and_resets_to_dense(self):
        q, k, v = self.make_inputs()
        v[:, :, 1] += 100  # Keep the omitted key distinguishable from the mean of keys 0 and 2.
        q[:, :, 30] = torch.tensor([0., 40., 0.])
        q[:, :, 100] = torch.tensor([0., 0., 40.])
        processor = processors.FlashVDMTopMCrossAttentionProcessor(topk=True)
        # The coarse path samples queries 0 and 100, selecting keys 0 and 2.
        expected = F.scaled_dot_product_attention(q, k[:, :, [0, 2]], v[:, :, [0, 2]])
        torch.testing.assert_close(processor(None, q, k, v), expected)
        self.assertIs(processor.topk, False)
        # The next call must use all keys, including key 1 for query 30.
        dense = F.scaled_dot_product_attention(q, k, v)
        self.assertFalse(torch.allclose(dense, expected))
        torch.testing.assert_close(processor(None, q, k, v), dense)

    def test_coarse_call_retains_distinct_batch_values(self):
        q = torch.zeros(2, 2, 8, 4)
        k = torch.zeros(2, 2, 12, 4)
        v = torch.ones_like(k)
        v[1] = 9
        processor = processors.FlashVDMTopMCrossAttentionProcessor(topk=True)
        actual = processor(None, q, k, v)
        expected = torch.ones_like(q)
        expected[1] = 9
        torch.testing.assert_close(actual, expected)

    def test_selection_preserves_each_batch_keys_and_values(self):
        q, k, v = self.make_inputs(batch=2, length=2)
        q.zero_()  # Uniform scores activate every key.
        k[1] *= 2
        processor = processors.FlashVDMTopMCrossAttentionProcessor()
        selected_k, selected_v = processor.select_topkv(q, k, v, topk=1)
        torch.testing.assert_close(selected_k, k)
        torch.testing.assert_close(selected_v, v)

    def test_merge_retains_union_of_active_indices_across_batches(self):
        q, k, v = self.make_inputs(batch=2, length=31)
        # Batch 0 activates key 0; batch 1 activates key 1. Key 2 is inactive.
        q[1] = torch.tensor([0., 40., 0.])
        # An unsampled query would activate key 2; it must not affect the union.
        q[:, :, 1] = torch.tensor([0., 0., 40.])
        processor = processors.FlashVDMTopMCrossAttentionProcessor()
        selected_k, selected_v = processor.select_topkv(q, k, v, topk=1)
        torch.testing.assert_close(selected_k, k[:, :, [0, 1]])
        torch.testing.assert_close(selected_v, v[:, :, [0, 1]])

    def test_default_stride_retains_thirty_query_spacing(self):
        q, k, v = self.make_inputs()
        q[:, :, 30] = torch.tensor([0., 40., 0.])
        q[:, :, 100] = torch.tensor([0., 0., 40.])
        processor = processors.FlashVDMTopMCrossAttentionProcessor()
        selected_k, selected_v = processor.select_topkv(q, k, v, topk=1)
        torch.testing.assert_close(selected_k, k[:, :, [0, 1]])
        torch.testing.assert_close(selected_v, v[:, :, [0, 1]])

    def test_explicit_stride_changes_sampled_queries(self):
        q, k, v = self.make_inputs()
        q[:, :, 30] = torch.tensor([0., 40., 0.])
        q[:, :, 100] = torch.tensor([0., 0., 40.])
        processor = processors.FlashVDMTopMCrossAttentionProcessor()
        selected_k, selected_v = processor.select_topkv(q, k, v, topk=1, stride=100)
        torch.testing.assert_close(selected_k, k[:, :, [0, 2]])
        torch.testing.assert_close(selected_v, v[:, :, [0, 2]])

    def test_chunked_merge_preserves_batch_values_and_query_order(self):
        q, k, v = self.make_inputs(batch=2, length=5)
        q[:, :, 2:] = torch.tensor([0., 40., 0.])
        processor = processors.FlashVDMTopMCrossAttentionProcessor(topk=([7, 11], [2, 3]))
        actual = processor(None, q, k, v)
        # Each chunk selects exactly one key, so its output is that key's value.
        expected = torch.cat((v[:, :, 0:1].expand(-1, -1, 2, -1),
                              v[:, :, 1:2].expand(-1, -1, 3, -1)), dim=-2)
        torch.testing.assert_close(actual, expected)
        self.assertIs(processor.topk, False)

    def test_dense_merge_mode_matches_standard_attention(self):
        q, k, v = self.make_inputs(batch=2, length=3)
        processor = processors.FlashVDMTopMCrossAttentionProcessor(topk=False)
        torch.testing.assert_close(processor(None, q, k, v), F.scaled_dot_product_attention(q, k, v))

    def test_mean_mode_still_selects_top_keys_per_batch_and_head(self):
        q = torch.ones(2, 2, 4, 3)
        k = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1).repeat(2, 2, 1, 3)
        v = torch.arange(72, dtype=torch.float32).reshape(2, 2, 6, 3)
        processor = processors.FlashVDMCrossAttentionProcessor(topk=True)
        expected = F.scaled_dot_product_attention(q, k[:, :, [5, 4]], v[:, :, [5, 4]])
        torch.testing.assert_close(processor(None, q, k, v), expected)
        self.assertIs(processor.topk, False)


if __name__ == "__main__":
    unittest.main()
