import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ===========
# Utilities
# ===========


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------- temporal residual block ----------
def conv1d_wt_norm(in_ch, out_ch, kernel_size, dilation=1, padding=0, stride=1):
    conv = nn.Conv1d(
        in_ch,
        out_ch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    return nn.utils.weight_norm(conv)


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, dilation, padding, dropout=0.2
    ):
        super().__init__()
        self.conv1 = conv1d_wt_norm(
            n_inputs, n_outputs, kernel_size, dilation=dilation, padding=padding
        )
        self.conv2 = conv1d_wt_norm(
            n_outputs, n_outputs, kernel_size, dilation=dilation, padding=padding
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

        # init
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)
            nn.init.constant_(self.downsample.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        res = x if self.downsample is None else self.downsample(x)
        
        # Crop output to match residual length (handle any padding mismatch)
        if out.shape[-1] != res.shape[-1]:
            min_len = min(out.shape[-1], res.shape[-1])
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]
        
        return self.relu(out + res)  # (B, n_outputs, T)


# ---------- TCN that can return per-layer maps ----------
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            pad = (kernel_size - 1) * dilation  # causal-style padding keeping length
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, dilation, padding=pad, dropout=dropout
                )
            )
        self.network = nn.ModuleList(layers)

    def forward(self, x, return_intermediate=False):
        out = x
        intermediates = []
        for layer in self.network:
            out = layer(out)
            intermediates.append(out)  # keep full (B, channels_i, T)
        if return_intermediate:
            return intermediates
        return out  # last temporal map (B, channels_last, T)


# ---------- classifier (fixed) ----------
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, emb):
        x = self.fc1(emb)
        x = self.act(x)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits


# ---------- encoder (collect per-layer maps, reduce, concat, compress, pool) ----------
class TempEncoder(nn.Module):
    def __init__(
        self,
        in_ch,
        channels,
        kernel_size=3,
        reduced_channels=16,
        bottleneck_channels=32,
        latent_time=1,
        dropout=0.2,
    ):
        """
        latent_time: desired time-length after pooling (1 gives vector bottleneck)
        """
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs=in_ch,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.reduced_convs = nn.ModuleList(
            [nn.Conv1d(ch, reduced_channels, kernel_size=1) for ch in channels]
        )
        self.compress = nn.Conv1d(
            reduced_channels * len(channels), bottleneck_channels, kernel_size=1
        )
        self.latent_time = latent_time
        # use adaptive pool to support variable input lengths
        self.pool = nn.AdaptiveAvgPool1d(latent_time)

    def forward(self, x):
        # x: (B, in_ch, T)
        T_in = x.shape[-1]
        intermediates = self.tcn(x, return_intermediate=True)  # list of (B, ch_i, T)
        
        # Get the actual time dimension after TCN (may have changed due to padding)
        T_after_tcn = intermediates[0].shape[-1]
        
        reduced = []
        for feat_map, reduce_conv in zip(intermediates, self.reduced_convs):
            # Ensure all feature maps have same time dimension
            if feat_map.shape[-1] != T_after_tcn:
                feat_map = feat_map[:, :, :T_after_tcn]
            # reduce to small channel count but keep time dimension
            reduced_map = reduce_conv(feat_map)  # (B, reduced_channels, T)
            reduced.append(reduced_map)
        
        concat = torch.cat(reduced, dim=1)  # (B, reduced_channels * num_layers, T)
        bottleneck_map = self.compress(concat)  # (B, bottleneck_channels, T)
        z = self.pool(bottleneck_map)  # (B, bottleneck_channels, latent_time)
        return z, reduced, intermediates, T_after_tcn  # return actual time dimension


# ---------- decoder (upsample -> reverse tcn -> reconstruct) ----------
class TempDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        channels,
        kernel_size=3,
        reduced_channels=16,
        bottleneck_channels=32,
        latent_time=1,
        dropout=0.2,
    ):
        super().__init__()
        self.latent_time = latent_time
        self.reduced_channels = reduced_channels
        num_layers = len(channels)
        
        self.compress_back = nn.Conv1d(
            bottleneck_channels, reduced_channels * num_layers, kernel_size=1
        )
        
        # reversed channels for decoder TCN
        rev_channels = list(reversed(channels))
        self.tcn = TemporalConvNet(
            num_inputs=reduced_channels * num_layers,
            num_channels=rev_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        
        # final reductions (1x1 conv from each rev layer -> reduced_channels)
        self.reduction_layers = nn.ModuleList(
            [nn.Conv1d(ch, reduced_channels, kernel_size=1) for ch in rev_channels]
        )
        
        # Skip fusion layers: project TCN output + skip to reduced_channels
        self.skip_fusion_layers = nn.ModuleList()
        for ch in rev_channels:
            # Takes TCN output (ch channels) and skip (reduced_channels) concatenated
            self.skip_fusion_layers.append(
                nn.Conv1d(ch + reduced_channels, reduced_channels, kernel_size=1)
            )
        
        self.final = nn.Conv1d(
            reduced_channels * num_layers, out_channels, kernel_size=1
        )

    def forward(self, z, encoder_time_length, skips_reduced, skips_intermediate, tcn_time_length):
        # z: (B, bottleneck_channels, latent_time)
        B = z.shape[0]
        
        # Expand bottleneck back to TCN time length
        x = self.compress_back(z)  # (B, reduced_channels * num_layers, latent_time)
        x = F.interpolate(x, size=tcn_time_length, mode="linear", align_corners=False)

        # Pass through decoder TCN to get all intermediate features
        dec_intermediates = []
        out = x
        for layer in self.tcn.network:
            out = layer(out)
            dec_intermediates.append(out)
        
        # Now fuse decoder features with encoder skips
        dec_maps = []
        reversed_skips = list(reversed(skips_reduced))
        
        for dec_feat, skip, fusion_layer in zip(dec_intermediates, reversed_skips, self.skip_fusion_layers):
            # Match time dimensions
            target_len = min(dec_feat.shape[-1], skip.shape[-1])
            dec_feat = dec_feat[:, :, :target_len]
            skip = skip[:, :, :target_len]
            
            # Concatenate decoder feature and skip, then fuse
            combined = torch.cat([dec_feat, skip], dim=1)  # (B, ch + reduced_channels, T)
            fused = fusion_layer(combined)  # (B, reduced_channels, T)
            dec_maps.append(fused)

        concat = torch.cat(dec_maps, dim=1)
        out = self.final(concat)  # (B, out_channels, T)
        
        # Ensure output matches original input length
        if out.shape[-1] != encoder_time_length:
            if out.shape[-1] > encoder_time_length:
                out = out[:, :, :encoder_time_length]
            else:
                out = F.interpolate(out, size=encoder_time_length, mode="linear", align_corners=False)
        
        return out


# ----------AutoEncoder Wrapper ----------
class TemporalAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        channels=None,
        kernel_size=3,
        reduced_channels=16,
        bottleneck_channels=32,
        latent_time=1,
        dropout=0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 64, 64, 64]
        self.encoder = TempEncoder(
            in_channels,
            channels,
            kernel_size=kernel_size,
            reduced_channels=reduced_channels,
            bottleneck_channels=bottleneck_channels,
            latent_time=latent_time,
            dropout=dropout,
        )
        self.decoder = TempDecoder(
            in_channels,
            channels,
            kernel_size=kernel_size,
            reduced_channels=reduced_channels,
            bottleneck_channels=bottleneck_channels,
            latent_time=latent_time,
            dropout=dropout,
        )

    def forward(self, x):
        # x: (B, in_channels, T)
        T_original = x.shape[-1]
        z, reduced_skips, intermediates, T_tcn = self.encoder(x)  # z: (B, bottleneck, latent_time)
        recon = self.decoder(
            z,
            encoder_time_length=T_original,
            skips_reduced=reduced_skips,
            skips_intermediate=intermediates,
            tcn_time_length=T_tcn,
        )
        
        # Final safety check: ensure reconstruction matches input exactly
        if recon.shape[-1] != T_original:
            if recon.shape[-1] > T_original:
                recon = recon[:, :, :T_original]
            else:
                recon = F.interpolate(recon, size=T_original, mode="linear", align_corners=False)
        
        return recon, z, reduced_skips, intermediates


# Full ML Wrapper that incorporates the TCN + Classifier and Temporal AutoEncoder
class TechnoNet(nn.Module):
    def __init__(
        self,
        TCN_num_inputs,
        TCN_num_channels,
        Classifier_input_dim,
        Classifier_hidden_dim,
        Classifier_output_dim,
        TAE_in_channels,
        TCN_kernel_size=3,
        TCN_dropout=0.2,
        Classifier_dropout=0.2,
        TAE_channels=None,
        TAE_kernel_size=3,
        TAE_reduced_channels=16,
        TAE_bottleneck_channels=32,
        TAE_latent_time=1,
        TAE_dropout=0.2,
    ):
        super().__init__()
        self.tcn = TemporalConvNet(
            TCN_num_inputs, TCN_num_channels, TCN_kernel_size, TCN_dropout
        )
        self.classifier = Classifier(
            Classifier_input_dim,
            Classifier_hidden_dim,
            Classifier_output_dim,
            Classifier_dropout,
        )
        self.tae = TemporalAutoEncoder(
            TAE_in_channels,
            TAE_channels,
            TAE_kernel_size,
            TAE_reduced_channels,
            TAE_bottleneck_channels,
            TAE_latent_time,
            TAE_dropout,
        )

    # Fixed TechnoNet.forward() method
# Replace the existing forward method in your models.py

def forward(self, x, dyson_threshold=0.88):
    """
    Two-stage anomaly detection pipeline for Dyson sphere detection.
    
    Stage 1: TCN Classifier (Coarse Filter)
      - Class 0: Non-Dysonian (normal stars, variables, etc.)
      - Class 1: Dysonian (potential megastructure signatures)
      - Threshold: prob(class 1) > dyson_threshold → immediate flag
    
    Stage 2: TAE Anomaly Detector (Fine Filter)
      - Runs ONLY on non-Dysonian samples (below threshold)
      - Finds subtle anomalies via reconstruction error + latent space analysis
      - Training: TAE trained only on normal light curves
      - Inference: Anything "normal" by TCN but weird by TAE → flagged
    
    Args:
        x: Input light curves [B, 1, T]
        dyson_threshold: Probability threshold for Dysonian classification
                        (default 0.88 = high confidence required)
    
    Returns:
        outputs: Dictionary with:
          - TCN outputs: logits, probs, preds, confidence
          - Dyson candidates: dyson_mask, dyson_indices
          - TAE outputs (only for non-Dyson samples): recon, recon_error, latent_embedding
    """
    batch_size = x.shape[0]
    
    # ============================================
    # Stage 1: TCN Classification (All Samples)
    # ============================================
    tcn_out = self.tcn(x)
    pooled = F.adaptive_avg_pool1d(tcn_out, 1).squeeze(-1)
    logits = self.classifier(pooled)
    probs = F.softmax(logits, dim=-1)
    conf, preds = probs.max(dim=-1)
    
    # Dysonian probability (class 1)
    dyson_probs = probs[:, 1]
    
    # High-confidence Dyson sphere candidates (SKIP TAE for these)
    dyson_mask = dyson_probs >= dyson_threshold
    dyson_indices = torch.nonzero(dyson_mask, as_tuple=True)[0]
    
    # Non-Dysonian samples (SEND TO TAE for anomaly analysis)
    non_dyson_mask = ~dyson_mask
    non_dyson_indices = torch.nonzero(non_dyson_mask, as_tuple=True)[0]
    
    outputs = {
        "logits": logits,
        "probs": probs,
        "preds": preds,
        "confidence": conf,
        "dyson_probs": dyson_probs,
        "dyson_mask": dyson_mask,
        "dyson_indices": dyson_indices,
        "total_dyson_candidates": dyson_mask.sum().item(),
    }
    
    # ============================================
    # Stage 2: TAE Anomaly Detection (Non-Dyson Samples Only)
    # ============================================
    if non_dyson_mask.any():
        # Extract non-Dysonian samples
        x_non_dyson = x[non_dyson_mask]
        
        # Run TAE reconstruction
        recon, z, _, _ = self.tae(x_non_dyson)
        
        # Per-sample reconstruction error (MAE)
        recon_error = torch.abs(recon - x_non_dyson).mean(dim=[1, 2])  # [N]
        
        # Store results with indices mapping back to original batch
        outputs.update({
            "non_dyson_mask": non_dyson_mask,
            "non_dyson_indices": non_dyson_indices,
            "recon": recon,  # Reconstructions for non-Dyson samples
            "recon_error": recon_error,  # [N] where N = non_dyson_mask.sum()
            "latent_embedding": z,  # Latent embeddings for Mahalanobis distance
            "total_tae_analyzed": non_dyson_mask.sum().item(),
        })
    else:
        # All samples are Dyson candidates (rare, but handle it)
        outputs.update({
            "non_dyson_mask": non_dyson_mask,
            "non_dyson_indices": torch.tensor([], dtype=torch.long, device=x.device),
            "total_tae_analyzed": 0,
        })
    
    return outputs