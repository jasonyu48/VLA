import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device
import torch.nn as nn


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
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
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.device = next(wm.parameters()).device

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

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
            try:
                # For text goals, use cosine similarity between visual and text embeddings
                # Get the last predicted visual embedding
                visual_emb = z_obs_pred["visual"][:, -1]  # Shape: [B, 1, D]
                
                # Get the text embedding
                text_emb = z_obs_tgt["text"]  # Shape: [B, 1, 1, D]
                
                # Print shapes for debugging
                # print(f"Visual embedding shape: {visual_emb.shape}")
                # print(f"Text embedding shape: {text_emb.shape}")
                
                text_emb = text_emb.squeeze(1).squeeze(1)  # [B, D]
                visual_emb = visual_emb.squeeze(1)  # [B, D]
                    
                # print(f"Final visual shape: {visual_emb.shape}, text shape: {text_emb.shape}")
                
                # Compute cosine similarity
                visual_norm = torch.nn.functional.normalize(visual_emb, p=2, dim=1)
                text_norm = torch.nn.functional.normalize(text_emb, p=2, dim=1)

                # # Compute L2 distance
                # loss = torch.norm(visual_norm - text_norm, p=2, dim=1)
                
                # Compute dot product
                similarity = torch.sum(visual_norm * text_norm, dim=1)
                
                # Convert similarity to distance (1 - similarity)
                loss = 1 - similarity # torch.Size([300]) This loss is not L2 loss!!!!
                # print(f"Loss shape: {loss.shape}, values: {loss[:5]}")  # Print first 5 values
                return loss
                
            except Exception as e:
                print(f"Error in computing text-based loss: {e}")
                print("Falling back to random ranking")
                
                # Fallback: return random values as loss
                batch_size = z_obs_pred["visual"].shape[0]
                return torch.rand(batch_size, device=z_obs_pred["visual"].device)
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
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        # Transform initial observations
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        
        # Handle text-based goals differently
        if "text" in obs_g:
            # Text goals don't need transformation
            trans_obs_g = {"text": obs_g["text"]}
            # trans_obs_g = move_to_device(trans_obs_g, self.device) #---------not needed
        else:
            # Transform visual goals
            trans_obs_g = move_to_device(
                self.preprocessor.transform_obs(obs_g), self.device
            )
            
        # Encode observations
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                loss = self._compute_loss(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid
