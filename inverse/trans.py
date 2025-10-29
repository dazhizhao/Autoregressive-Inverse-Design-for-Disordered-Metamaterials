import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from main import instantiate_from_config
from transformers.generation.utils import top_k_top_p_filtering
import numpy as np

class GridTransformer(pl.LightningModule):
    """
    GridTransformer Model

    This model is specialized for autoregressive generation on a 2D grid data.
    It does not rely on first-stage models like VQGAN, but deals directly with grids that have been encoded as one-hot vectors.

    Input (training): a tensor of shape [B, C, H, W], where C is num_classes, and is one-hot in the C dimension.
                   where C is the number of classes (num_classes) and is one-hot coded in C dimensions.
    Output (when generating): a tensor of shape [B, 1, H, W], where each value is the category index of the corresponding position.
    """

    def __init__(self,
                 transformer_config, 
                 grid_size=(10, 10),
                 num_classes=9, 
                 learning_rate=1e-4,
                 cond_dim=None): 
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.cond_dim = cond_dim
        self.transformer = instantiate_from_config(config=transformer_config)

        embed_dim = self.transformer.config.n_embd
        
        self.input_proj = nn.Linear(self.num_classes, embed_dim)
        
        self.cond_encoder = nn.Sequential(
            nn.Linear(self.cond_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
                
        self.sos_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.seq_len = self.grid_size[0] * self.grid_size[1]

    def forward(self, batch):
        condition, x_indices = batch
        b = x_indices.shape[0]
        x_one_hot = F.one_hot(x_indices, num_classes=self.num_classes).float()
        x_one_hot = x_one_hot.permute(0, 3, 1, 2)
        target_indices = torch.argmax(x_one_hot, dim=1)
        target = target_indices.view(b, -1)
        x_flat_one_hot = x_one_hot.permute(0, 2, 3, 1).reshape(b, self.seq_len, self.num_classes)
        x_proj = self.input_proj(x_flat_one_hot)
        cond_embed = self.cond_encoder(condition).unsqueeze(1)
        x_input_shifted = x_proj[:, :-1, :]
        sos = self.sos_token.expand(b, -1, -1)
        transformer_input = torch.cat((cond_embed, sos, x_input_shifted), dim=1)
        logits, _ = self.transformer(idx=None, embeddings=transformer_input)
        logits_aligned = logits[:, 1:, :]
        
        return logits_aligned, target

    @torch.no_grad()
    def sample(self,
               condition,
               steps,
               batch_size=1,
               temperature=1.0,
               top_p=0.9):
        """
        A streamlined version of the sampling function specialized for Top-P (core sampling) + Temperature strategies.
        """
        self.eval()
        device = self.sos_token.device
        
        cond_embed = self.cond_encoder(condition.to(device)).unsqueeze(1)
        sos = self.sos_token.expand(batch_size, -1, -1)
        input_seq = torch.cat((cond_embed, sos), dim=1)

        generated_indices = []
        for _ in range(steps):
            logits_full, _ = self.transformer(idx=None, embeddings=input_seq)
            next_token_logits = logits_full[:, -1, :]

            next_token_logits = next_token_logits / temperature
            
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)
            
            probs = F.softmax(next_token_logits, dim=-1)
            
            next_index = torch.multinomial(probs, num_samples=1)
            
            generated_indices.append(next_index)

            next_one_hot = F.one_hot(next_index, num_classes=self.num_classes).float()

            next_embed = self.input_proj(next_one_hot)

            input_seq = torch.cat((input_seq, next_embed), dim=1)

        generated_indices_tensor = torch.cat(generated_indices, dim=1)
        final_grid = generated_indices_tensor.view(batch_size, 1, *self.grid_size)
        
        self.train()
        return final_grid
    
    @torch.no_grad()
    def sample_and_visualize_attention(
        self,
        condition,
        steps,
        output_dir,
        batch_size=1,
        temperature=1.0,
        top_p=0.9,
        layer_to_vis=5,
        head_to_vis=9
    ):
        """
        Perform sampling and visualize the final sequence with attention.
        """
        self.eval()
        device = self.sos_token.device
        
        cond_embed = self.cond_encoder(condition.to(device)).unsqueeze(1)
        sos = self.sos_token.expand(batch_size, -1, -1)
        input_seq = torch.cat((cond_embed, sos), dim=1)

        for _ in range(steps):
            logits_full, _ = self.transformer(idx=None, embeddings=input_seq)
            next_token_logits = logits_full[:, -1, :]
            
            next_token_logits = next_token_logits / temperature
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)
            probs = F.softmax(next_token_logits, dim=-1)
            next_index = torch.multinomial(probs, num_samples=1)
            
            next_one_hot = F.one_hot(next_index, num_classes=self.num_classes).float()
            next_embed = self.input_proj(next_one_hot)
            input_seq = torch.cat((input_seq, next_embed), dim=1)

        # Perform forward propagation to capture attention graph
        _, all_attentions = self.transformer.forward_for_visualization(idx=None, embeddings=input_seq)

        attention_slice = all_attentions[layer_to_vis][0, head_to_vis, :, :].cpu().numpy()
        
        tokens = ['Cond', 'SOS'] + [f'T{i+1}' for i in range(steps)]
        
        mask = np.triu(np.ones_like(attention_slice, dtype=bool), k=1)

        plt.figure(figsize=(16, 14))
        
        sns.heatmap(
            attention_slice, 
            xticklabels=tokens, 
            yticklabels=tokens, 
            cmap="coolwarm",
            mask=mask,
            cbar_kws={"shrink": .75}
        )
        
        ax = plt.gca()
        ax.set_facecolor('white')

        plt.title(f"Causal Attention Map (Layer {layer_to_vis+1}, Head {head_to_vis+1})\n"
                  f"Temp: {temperature}, Top-P: {top_p}", fontsize=16)
        plt.ylabel("Query", fontsize=12)
        plt.xlabel("Key", fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"attention_L{layer_to_vis+1}_H{head_to_vis+1}_t{temperature}_p{top_p}.png")
        plt.savefig(save_path, dpi=200, facecolor='white')
        plt.close()

    def shared_step(self, batch, batch_idx):
        x_one_hot = batch
        logits, target = self(x_one_hot)
        loss = F.cross_entropy(logits.view(-1, self.num_classes), target.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        logits, target = self(batch)
        loss = F.cross_entropy(
            logits.contiguous().view(-1, self.num_classes), 
            target.view(-1)
        )
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, target = self(batch)
        loss = F.cross_entropy(
            logits.contiguous().view(-1, self.num_classes), 
            target.view(-1)
        )
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer