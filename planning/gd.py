import torch
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device


class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        action_noise,
        sample_type,
        lr,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.action_noise = action_noise
        self.sample_type = sample_type
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix

    def init_actions(self, obs_0, actions=None):
        """
        Initializes or appends actions for planning, ensuring the output shape is (b, self.horizon, action_dim).
        """
        n_evals = obs_0["visual"].shape[0]
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.sample_type == "randn":
                new_actions = torch.randn(n_evals, remaining_t, self.action_dim)
            elif self.sample_type == "zero":  # zero action of env
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
                new_actions = rearrange(
                    new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
                )
                new_actions = self.preprocessor.normalize_actions(new_actions)
                new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            actions = torch.cat([actions, new_actions.to(device)], dim=1)
        return actions

    def get_action_optimizer(self, actions):
        return torch.optim.SGD([actions], lr=self.lr)

    def _compute_loss(self, z_obs_pred, z_obs_tgt):
        """
        Custom loss function that can handle both visual and text-based goals
        
        Args:
            z_obs_pred: Predicted observations in latent space
            z_obs_tgt: Target observations in latent space
            
        Returns:
            loss: Loss tensor of shape (B,)
        """
        # Check if we're using text-based goals
        if "text" in z_obs_tgt:
            # For text goals, use cosine similarity between visual and text embeddings
            # Get the last predicted visual embedding
            visual_emb = z_obs_pred["visual"][:, -1]  # Shape: [B, 1, D]
            
            # Get the text embedding
            text_emb = z_obs_tgt["text"]  # Shape: [B, 1, 1, D]
            
            text_emb = text_emb.squeeze(1).squeeze(1)  # [B, D]
            visual_emb = visual_emb.squeeze(1)  # [B, D]
                
            # Compute cosine similarity
            visual_norm = torch.nn.functional.normalize(visual_emb, p=2, dim=1)
            text_norm = torch.nn.functional.normalize(text_emb, p=2, dim=1)
            
            # Compute dot product
            similarity = torch.sum(visual_norm * text_norm, dim=1)
            
            # Convert similarity to distance (1 - similarity)
            loss = 1 - similarity
            return loss
        else:
            # Use the original objective function for visual goals
            return self.objective_fn(z_obs_pred, z_obs_tgt)

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            obs_0: Initial observations
            obs_g: Goal observations (can be visual or text)
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        
        # Handle text-based goals differently
        if "text" in obs_g:
            # Text goals don't need transformation
            trans_obs_g = {"text": obs_g["text"]}
        else:
            # Transform visual goals
            trans_obs_g = move_to_device(
                self.preprocessor.transform_obs(obs_g), self.device
            )
            
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        z_obs_g_detached = {key: value.detach() for key, value in z_obs_g.items()}

        actions = self.init_actions(obs_0, actions).to(self.device)
        actions.requires_grad = True
        optimizer = self.get_action_optimizer(actions)
        n_evals = actions.shape[0]

        # all_losses = []
        for i in range(self.opt_steps):
            optimizer.zero_grad()
            i_z_obses, i_zs = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
            # Use the custom loss function instead of directly using objective_fn
            loss = self._compute_loss(i_z_obses, z_obs_g_detached)  # (n_evals, )
            # if i%100 == 0:
            #     all_losses.append(loss.detach().cpu().numpy())
            total_loss = loss.mean() * n_evals  # loss for each eval is independent
            if i % self.eval_every == 0:
                print(f"loss: {total_loss.item()}")
            total_loss.backward()
            with torch.no_grad():
                actions_new = actions - optimizer.param_groups[0]["lr"] * actions.grad
                actions_new += (
                    torch.randn_like(actions_new) * self.action_noise
                )  # Add Gaussian noise
                actions.copy_(actions_new)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": total_loss.item(), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success
        # print(f"all_losses: {all_losses}")
        return actions, np.full(n_evals, np.inf)  # all actions are valid
