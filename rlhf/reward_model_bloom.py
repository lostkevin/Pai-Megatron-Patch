import torch
from torch import nn

from transformers import AutoModelForCausalLM, AutoTokenizer


class BLOOMRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(
            self.config, 'hidden_size') else self.config.n_embd
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        # self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-1b1')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)['input_ids'][0]

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # print("self.config.n_embd: ", self.config.n_embd)
        hidden_states = transformer_outputs[0]
        # print("hidden_states: ", hidden_states.size())
        # print("v_head(hidden_states): ", self.v_head(hidden_states).size())

        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        # print("input_ids: ", input_ids)
        # print("input_ids: ", input_ids.size())
        # print("rewards: ", rewards.size())

        # Compute pairwise loss. Only backprop on the last value before padding
        loss = 0
        inference = False
        for i in range(bs):
            # print("chosen: ", self.tokenizer.decode(chosen[i]))
            # print("chosen[i]: ", chosen[i])
            # print("rejected: ", self.tokenizer.decode(rejected[i]))
            # print("rejected[i]: ", rejected[i])
            # print("torch.eq(chosen[i], rejected[i]): ", torch.eq(chosen[i], rejected[i]))
            # print("torch.all(torch.eq(chosen[i], rejected[i])): ", torch.all(torch.eq(chosen[i], rejected[i])))
            # print("torch.all(torch.eq(chosen[i], rejected[i])).item(): ", torch.all(torch.eq(chosen[i], rejected[i])).item())
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item(
                ) if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            # print("PAD_ID_pad_token: ", self.tokenizer(self.tokenizer.pad_token))
            # print("PAD_ID_input_ids: ", self.tokenizer(self.tokenizer.pad_token)["input_ids"])
            # print("PAD_ID: ", self.PAD_ID)
            # print("chosen[i]: ", chosen[i])
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            # print("c_ind: ", c_ind)
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            # print("r_ind: ", r_ind)
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            # print("chosen_rewards.size: ", chosen_rewards.size())
            # print("chosen_rewards[i].size: ", chosen_rewards[i].size())
            # print("divergence_ind: ", divergence_ind)
            # print("end_ind: ", end_ind)
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            # print("c_truncated_reward.size: ", c_truncated_reward.size())
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            # Compute loss
            loss += -torch.log(
                torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {'chosen_end_scores': chosen_end_scores}

        return {
            'loss': loss,
            'chosen_end_scores': chosen_end_scores,
            'rejected_end_scores': rejected_end_scores,
        }
