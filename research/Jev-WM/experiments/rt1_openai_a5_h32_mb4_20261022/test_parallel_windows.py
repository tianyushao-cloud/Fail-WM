"""Numerical checks for episode isolation and equivalent accumulated gradients."""
import copy
import unittest

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from futureworlds.memory import frame_logp
from futureworlds.parallel_windows import bind_condition, pack_condition, rollout_windows
from futureworlds.search import rollout_group_beam
from futureworlds.text_conditioning import TextWorldModel
from futureworlds.training import policy_backward


def model_from(state=None):
    core = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=512, attention_dropout=0.0,
        attn_implementation="sdpa"))
    model = TextWorldModel(core, text_width=16, layers=(0, 1), inner=16).eval()
    if state is None:
        for adapter in model.adapters.values():
            torch.nn.init.normal_(adapter.out.weight, std=.03)
    else:
        model.load_state_dict(state)
    return model


class ParallelWindowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261022)
        self.model = model_from()
        self.conditions = [dict(text_features=torch.randn(1, n, 16),
            text_mask=torch.ones(1, n, dtype=torch.bool)) for n in [2, 5, 3, 4]]
        self.context = torch.randint(16, 32, (4, 5))
        self.anchor = torch.randint(0, 16, (4, 3))
        self.actions = torch.randint(32, 64, (4, 4, 2))

    def test_condition_padding_and_window_expansion(self):
        packed = pack_condition(self.conditions, 8)
        self.assertEqual(packed['text_features'].shape, (32, 5, 16))
        for i, condition in enumerate(self.conditions):
            n = condition['text_features'].shape[1]
            for j in range(8):
                torch.testing.assert_close(packed['text_features'][i * 8 + j, :n], condition['text_features'][0])
                self.assertFalse(packed['text_mask'][i * 8 + j, n:].any())

    def test_batched_beam_matches_four_independent_windows(self):
        combined = rollout_windows(self.model, self.context, self.anchor, self.actions,
            horizon=2, penalty=.1, conditions=self.conditions, visual=16, width=3)
        for i in range(4):
            bind_condition(self.model, self.conditions[i])
            serial, _ = rollout_group_beam(self.model, self.context[i:i + 1],
                self.anchor[i:i + 1], self.actions[i], horizon=2, penalty=.1, visual=16, width=3)
            for actual, expected in zip(combined, serial):
                for key in ('prefix', 'response'):
                    self.assertTrue(torch.equal(actual[key][i * 4:(i + 1) * 4], expected[key]))
                torch.testing.assert_close(actual['model_logp'][i * 4:(i + 1) * 4],
                    expected['model_logp'], atol=2e-6, rtol=2e-6)

    def test_changing_one_episode_does_not_change_other_beams(self):
        original = rollout_windows(self.model, self.context, self.anchor, self.actions,
            2, .1, self.conditions, visual=16, width=3)
        context = self.context.clone()
        context[0] = 63 - context[0]
        changed = rollout_windows(self.model, context, self.anchor, self.actions,
            2, .1, self.conditions, visual=16, width=3)
        for first, second in zip(original, changed):
            self.assertTrue(torch.equal(first['response'][4:], second['response'][4:]))
            self.assertTrue(torch.equal(first['prefix'][4:], second['prefix'][4:]))

    def test_batched_gradients_match_four_accumulated_window_losses(self):
        serial = model_from(self.model.state_dict())
        batched = model_from(self.model.state_dict())
        reference = model_from(self.model.state_dict()).requires_grad_(False)
        reference_state = copy.deepcopy(reference.state_dict())
        with torch.no_grad():
            reference.core.lm_head.weight.mul_(.97)
        condition = pack_condition(self.conditions, 4)
        bind_condition(batched, condition)
        responses = [torch.randint(0, 16, (16, 80)) for _ in range(2)]
        prefixes = [self.context.repeat_interleave(4, 0)]
        prefixes.append(torch.cat([prefixes[0], responses[0]], 1))
        with torch.no_grad():
            frames = [dict(prefix=p, response=v, model_logp=frame_logp(batched, p, v))
                      for p, v in zip(prefixes, responses)]
        rewards = torch.tensor([[1., -.5, .2, -.7], [0., 0., 0., 0.],
                                [.1, .4, -.2, -.3], [2., -1., -1., 0.]])
        serial_loss = 0.
        for i in range(4):
            bind_condition(serial, self.conditions[i])
            bind_condition(reference, self.conditions[i])
            selected = [{k: v[i * 4:(i + 1) * 4] for k, v in f.items()} for f in frames]
            serial_loss += policy_backward(serial, serial, reference, selected, rewards[i],
                self.conditions[i], 2, .2, .001, scale=1/4)
        bind_condition(reference, condition)
        batched_loss = policy_backward(batched, batched, reference, frames, rewards.flatten(),
            condition, 2, .2, .001, group_ids=np.repeat(np.arange(4), 4))
        self.assertAlmostEqual(serial_loss, batched_loss, places=6)
        worst = 0.
        for (name, p), (other_name, q) in zip(serial.named_parameters(), batched.named_parameters()):
            self.assertEqual(name, other_name)
            if p.grad is None:
                self.assertIsNone(q.grad)
                continue
            worst = max(worst, float((p.grad - q.grad).abs().max()))
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-3, msg=name)
        self.assertTrue(any(p.grad is not None and p.grad.abs().max() > 0 for p in batched.parameters()))
        self.assertTrue(all(p.grad is None for p in reference.parameters()))
        print('maximum gradient absolute difference:', worst)

    def test_reference_text_hooks_are_independent(self):
        actor = self.model
        reference = model_from(actor.state_dict()).requires_grad_(False)
        condition = pack_condition(self.conditions, 4)
        bind_condition(actor, condition)
        bind_condition(reference, condition)
        ids = self.context.repeat_interleave(4, 0)
        with torch.no_grad():
            before = reference(input_ids=ids).logits.clone()
            for adapter in actor.adapters.values():
                adapter.out.weight.add_(1)
            after = reference(input_ids=ids).logits
        self.assertTrue(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
