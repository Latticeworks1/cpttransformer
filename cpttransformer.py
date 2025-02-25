import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union

class PositionalEncoding(nn.Module):
    """
    Multi-frequency sinusoidal positional encoding for depth information.
    Incorporates both absolute and log-scaled depth representations.
    """
    def __init__(self, d_model: int, max_depth: float = 100.0, n_freqs: int = 16, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_depth = max_depth
        self.d_model = d_model
        self.n_freqs = n_freqs
        
        # Register buffer to avoid recomputing frequency bands
        freq_bands = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs // 2)
        self.register_buffer("freq_bands", freq_bands)
        
    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: Tensor of shape [batch_size, seq_len, 1] containing depth values in meters
        Returns:
            Positional encodings of shape [batch_size, seq_len, d_model]
        """
        # Normalize depths to [0, 1]
        depth_norm = depth / self.max_depth
        
        # Apply log scaling for better representation of shallow depths
        log_depth = torch.log(depth + 1.0) / math.log(self.max_depth + 1.0)
        
        # Combine raw and log-scaled depths
        depth_features = torch.cat([depth_norm, log_depth], dim=-1)
        
        # Compute sinusoidal encodings for multiple frequencies
        args = depth_norm * self.freq_bands.view(1, 1, -1)
        
        # Compute embeddings
        pos_encodings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # Add relative depth differences (gradient-like features)
        depth_diff = F.pad(depth[:, 1:] - depth[:, :-1], (0, 0, 0, 1), "constant", 0)
        depth_diff_norm = torch.tanh(depth_diff * 20.0)  # Scale and bound differences
        
        # Combine all features
        pos_encodings = torch.cat([pos_encodings, depth_features, depth_diff_norm], dim=-1)
        
        # Project to d_model dimensions
        if pos_encodings.shape[-1] != self.d_model:
            pos_enc_projection = nn.Linear(pos_encodings.shape[-1], self.d_model).to(pos_encodings.device)
            pos_encodings = pos_enc_projection(pos_encodings)
            
        return self.dropout(pos_encodings)


class SpatialEncoding(nn.Module):
    """
    Geospatial encoding using Random Fourier Features (RFF) to capture location-dependent patterns.
    """
    def __init__(self, d_model: int, n_freqs: int = 32, scale: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_freqs = n_freqs
        self.scale = scale
        
        # Random projection matrix for RFF
        self.register_buffer("projection", torch.randn(2, n_freqs) * scale)
        
        # Projection to d_model if needed
        self.output_projection = nn.Linear(2 * n_freqs, d_model)
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Tensor of shape [batch_size, 2] containing coordinates (x, y) in meters
                   using Equal Earth Projection
        Returns:
            Spatial encodings of shape [batch_size, d_model]
        """
        # Project coordinates
        projection = torch.matmul(coords, self.projection)
        
        # Apply sine and cosine
        rff_features = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        
        # Project to d_model dimensions
        spatial_encoding = self.output_projection(rff_features)
        
        return spatial_encoding


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention module that captures soil patterns at different scales.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int = 8, 
        dropout: float = 0.1,
        window_size: int = 16,
        dilation: int = 1,
        downsample: int = 1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.dilation = dilation
        self.downsample = downsample
        
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _downsample(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Downsample sequence for global attention"""
        batch_size, seq_len, d_model = x.shape
        if self.downsample > 1:
            # Ensure sequence length is divisible by downsample factor
            pad_len = (self.downsample - (seq_len % self.downsample)) % self.downsample
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))
                seq_len = x.shape[1]
                
            # Reshape and pool
            x = x.reshape(batch_size, seq_len // self.downsample, self.downsample, d_model)
            x = x.mean(dim=2)  # Average pooling for downsampling
            return x, seq_len
        return x, seq_len
    
    def _apply_dilation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dilation to capture patterns at different scales"""
        if self.dilation > 1:
            batch_size, seq_len, d_model = x.shape
            # Take every dilation-th element
            x = x[:, ::self.dilation, :]
            return x
        return x
    
    def _apply_windowing(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Apply local windowed attention"""
        batch_size, seq_len, d_model = q.shape
        if self.window_size < seq_len and self.window_size > 0:
            # Create sliding windows
            windows = []
            for i in range(0, seq_len, self.window_size // 2):
                end = min(i + self.window_size, seq_len)
                start = max(0, end - self.window_size)
                windows.append((start, end))
                if end == seq_len:
                    break
                    
            # Process each window
            attn_outputs = []
            for start, end in windows:
                q_win = q[:, start:end, :]
                k_win = k[:, start:end, :]
                v_win = v[:, start:end, :]
                
                # Compute attention
                scores = torch.matmul(q_win, k_win.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if mask is not None:
                    win_mask = mask[:, start:end, start:end]
                    scores = scores.masked_fill(win_mask == 0, -1e9)
                
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                attn_output = torch.matmul(attn_weights, v_win)
                attn_outputs.append((start, end, attn_output))
                
            # Merge overlapping windows with linear blending
            output = torch.zeros_like(q)
            counts = torch.zeros((batch_size, seq_len, 1), device=q.device)
            
            for start, end, attn_output in attn_outputs:
                output[:, start:end, :] += attn_output
                counts[:, start:end, :] += 1
                
            # Average overlapping regions
            output = output / counts
            return output
        else:
            # Fall back to standard attention for small sequences
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
                
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            return torch.matmul(attn_weights, v)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input of shape [batch_size, seq_len, d_model]
            mask: Attention mask of shape [batch_size, seq_len, seq_len]
        Returns:
            Output of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply dilation if needed
        if self.dilation > 1:
            x_dilated = self._apply_dilation(x)
        else:
            x_dilated = x
            
        # Apply downsampling if needed
        if self.downsample > 1:
            x_ds, orig_len = self._downsample(x)
        else:
            x_ds = x_dilated
            
        # Project to Q, K, V
        q = self.q_proj(x_ds).reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_ds).reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_ds).reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention mechanism based on window_size
        if self.window_size > 0 and self.dilation == 1 and self.downsample == 1:
            # Reshape for windowed attention
            q_reshaped = q.transpose(1, 2).reshape(batch_size, -1, self.d_model)
            k_reshaped = k.transpose(1, 2).reshape(batch_size, -1, self.d_model)
            v_reshaped = v.transpose(1, 2).reshape(batch_size, -1, self.d_model)
            
            # Apply windowed attention
            attn_output = self._apply_windowing(q_reshaped, k_reshaped, v_reshaped, mask)
            
            # Reshape back
            attn_output = attn_output.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            # Standard multi-head attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply mask if provided
            if mask is not None:
                if self.downsample > 1:
                    # Adjust mask for downsampled sequence
                    mask = mask[:, ::self.downsample, ::self.downsample]
                if self.dilation > 1:
                    mask = mask[:, ::self.dilation, ::self.dilation]
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
                
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
            
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        
        # Upsample if needed
        if self.downsample > 1:
            # Interpolate back to original sequence length
            attn_output = F.interpolate(
                attn_output.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class HierarchicalSelfAttention(nn.Module):
    """
    Hierarchical self-attention module that processes CPT data at multiple scales.
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Fine-scale local attention (thin layers)
        self.local_attn = MultiScaleAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            window_size=16,  # Local window of ~0.8m with 0.05m intervals
            dilation=1,
            downsample=1
        )
        
        # Mid-scale dilated attention (layer transitions)
        self.mid_attn = MultiScaleAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            window_size=32,  # Wider context window
            dilation=2,      # Skip every other point
            downsample=1
        )
        
        # Large-scale downsampled attention (broad trends)
        self.global_attn = MultiScaleAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            window_size=0,   # Full sequence attention
            dilation=1,
            downsample=4     # Downsample by factor of 4
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Residual projection
        self.residual_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input of shape [batch_size, seq_len, d_model]
            mask: Attention mask of shape [batch_size, seq_len, seq_len]
        Returns:
            Output of shape [batch_size, seq_len, d_model]
        """
        # Process at different scales
        local_out = self.local_attn(x, mask)
        mid_out = self.mid_attn(x, mask)
        global_out = self.global_attn(x, mask)
        
        # Concatenate multi-scale outputs
        multi_scale = torch.cat([local_out, mid_out, global_out], dim=-1)
        
        # Fusion layer
        fused = self.fusion(multi_scale)
        
        # Residual connection
        residual = self.residual_proj(x)
        
        return fused + residual


class FeedForward(nn.Module):
    """Feed-forward network with Gaussian Error Linear Units (GELU)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape [batch_size, seq_len, d_model]
        Returns:
            Output of shape [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer with hierarchical self-attention."""
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = HierarchicalSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input of shape [batch_size, seq_len, d_model]
            mask: Attention mask of shape [batch_size, seq_len, seq_len]
        Returns:
            Output of shape [batch_size, seq_len, d_model]
        """
        # Pre-LayerNorm architecture (better training stability)
        x_norm = self.norm1(x)
        attn_output = self.attn(x_norm, mask)
        x = x + self.dropout(attn_output)
        
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        x = x + self.dropout(ff_output)
        
        return x


class CPTTransformer(nn.Module):
    """
    Transformer-based model for CPT data analysis with uncertainty quantification.
    Handles missing pore pressure data and provides calibrated geotechnical parameters.
    """
    def __init__(
        self,
        input_dim: int = 3,  # qc, fs, (optional u2)
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_depth: float = 100.0,
        n_soil_classes: int = 9,  # Robertson 9-class system
        regional_embedding_dim: int = 16,
        n_regions: int = 10,  # Number of geological regions to support
        enable_uncertainty: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.enable_uncertainty = enable_uncertainty
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_depth=max_depth,
            dropout=dropout
        )
        
        # Spatial encoding
        self.spatial_encoding = SpatialEncoding(
            d_model=d_model,
            n_freqs=32,
            scale=0.1
        )
        
        # Regional embedding (learned for different geological regions)
        self.region_embedding = nn.Embedding(n_regions, regional_embedding_dim)
        self.region_projection = nn.Linear(regional_embedding_dim, d_model)
        
        # Combined embedding
        self.embed_norm = nn.LayerNorm(d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Output norm
        self.output_norm = nn.LayerNorm(d_model)
        
        # Missing data indicator
        self.has_u2_embedding = nn.Embedding(2, d_model // 4)  # 0=missing, 1=present
        self.data_quality_projection = nn.Linear(d_model // 4, d_model)
        
        # Output heads
        # 1. Soil Behavior Type classification
        self.sbt_classifier = nn.Linear(d_model, n_soil_classes)
        
        # 2. Regression heads for geotechnical properties
        self.property_heads = nn.ModuleDict({
            'qt': nn.Linear(d_model, 1),                 # Corrected tip resistance
            'Fr': nn.Linear(d_model, 1),                 # Friction ratio
            'sigma_v0': nn.Linear(d_model, 1),           # Effective overburden stress
            'su': nn.Linear(d_model, 1),                 # Undrained shear strength
            'phi': nn.Linear(d_model, 1),                # Friction angle
            'E': nn.Linear(d_model, 1),                  # Soil modulus
            'permeability': nn.Linear(d_model, 1),       # Permeability estimate
        })
        
        # Uncertainty heads (optional)
        if enable_uncertainty:
            self.uncertainty_heads = nn.ModuleDict({
                key: nn.Linear(d_model, 1) for key in self.property_heads.keys()
            })
            
        # Pore pressure estimation (when missing)
        self.pore_pressure_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def _make_square_mask(self, sz: int) -> torch.Tensor:
        """Create square attention mask for sequence of length sz."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return ~mask  # True values will be attended to
    
    def forward(
        self, 
        cpt_data: torch.Tensor,                  # [batch_size, seq_len, input_dim]
        depth: torch.Tensor,                     # [batch_size, seq_len, 1]
        coords: torch.Tensor,                    # [batch_size, 2]
        region_ids: Optional[torch.Tensor] = None, # [batch_size]
        has_u2: Optional[torch.Tensor] = None,   # [batch_size] - binary indicator if u2 is present
        mask: Optional[torch.Tensor] = None      # [batch_size, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            cpt_data: CPT measurements [qc, fs, u2] (u2 may be missing)
            depth: Depth values in meters
            coords: Coordinates in Equal Earth projection (x, y) in meters
            region_ids: Optional geological region IDs
            has_u2: Binary indicator if pore pressure (u2) is present
            mask: Sequence mask where True values are valid and False values are padding
        Returns:
            Dictionary of predictions including soil classification and geotechnical properties
            with optional uncertainty estimates
        """
        batch_size, seq_len, _ = cpt_data.shape
        
        # Create default attention mask if not provided
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=cpt_data.device)
            
        # Create square attention mask
        square_mask = self._make_square_mask(seq_len).to(cpt_data.device)
        square_mask = square_mask.unsqueeze(0).expand(batch_size, -1, -1)
        # Apply padding mask
        square_mask = square_mask & mask.unsqueeze(1)
        
        # Input embedding
        x = self.input_embedding(cpt_data)
        
        # Positional encoding
        pos_enc = self.pos_encoding(depth)
        x = x + pos_enc
        
        # Spatial encoding
        spatial_enc = self.spatial_encoding(coords).unsqueeze(1).expand(-1, seq_len, -1)
        x = x + spatial_enc
        
        # Regional embedding (if provided)
        if region_ids is not None:
            region_emb = self.region_embedding(region_ids).unsqueeze(1).expand(-1, seq_len, -1)
            region_emb = self.region_projection(region_emb)
            x = x + region_emb
            
        # Data quality indicator (presence of u2)
        if has_u2 is not None:
            u2_indicator = torch.zeros(batch_size, dtype=torch.long, device=cpt_data.device)
            u2_indicator[has_u2] = 1
            quality_emb = self.has_u2_embedding(u2_indicator).unsqueeze(1).expand(-1, seq_len, -1)
            quality_emb = self.data_quality_projection(quality_emb)
            x = x + quality_emb
            
        # Apply encoder layers
        x = self.embed_norm(x)
        for layer in self.encoder_layers:
            x = layer(x, square_mask)
            
        # Final normalization
        x = self.output_norm(x)
        
        # Soil classification
        sbt_logits = self.sbt_classifier(x)
        
        # Estimate u2 (pore pressure) when missing
        u2_estimated = self.pore_pressure_estimator(x)
        
        # Geotechnical properties regression
        properties = {name: head(x) for name, head in self.property_heads.items()}
        
        # Compute uncertainties (if enabled)
        if self.enable_uncertainty:
            # Log-variance parametrization for numerical stability
            uncertainties = {
                name: torch.exp(head(x)) for name, head in self.uncertainty_heads.items()
            }
            
            # Create prediction dictionary with uncertainties
            predictions = {
                'sbt_logits': sbt_logits,
                'u2_estimated': u2_estimated
            }
            
            # Add mean and uncertainty for each property
            for name in properties.keys():
                predictions[name] = properties[name]
                predictions[f'{name}_uncertainty'] = uncertainties[name]
        else:
            # Create prediction dictionary without uncertainties
            predictions = {
                'sbt_logits': sbt_logits,
                'u2_estimated': u2_estimated,
                **properties
            }
            
        return predictions


class CPTLoss(nn.Module):
    """
    Multi-task loss function for CPT data analysis, supporting:
    - SBT classification with label smoothing
    - Regression with uncertainty quantification
    - Regional calibration
    """
    def __init__(
        self, 
        n_soil_classes: int = 9, 
        enable_uncertainty: bool = True,
        label_smoothing: float = 0.1,
        property_weights: Dict[str, float] = None
    ):
        super().__init__()
        self.n_soil_classes = n_soil_classes
        self.enable_uncertainty = enable_uncertainty
        
        # Classification loss
        self.cls_criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )
        
        # Default property weights if not provided
        if property_weights is None:
            self.property_weights = {
                'qt': 1.0,
                'Fr': 1.0,
                'sigma_v0': 1.0,
                'su': 1.0,
                'phi': 1.0,
                'E': 1.0,
                'permeability': 1.0
            }
        else:
            self.property_weights = property_weights
            
    def gaussian_nll_loss(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor, 
        var: torch.Tensor, 
        full: bool = False
    ) -> torch.Tensor:
        """
        Gaussian negative log-likelihood loss with uncertainty.
        
        Args:
            input: Predicted mean [batch_size, seq_len, 1]
            target: Ground truth value [batch_size, seq_len, 1]
            var: Predicted variance [batch_size, seq_len, 1]
            full: If True, return unreduced loss
        
        Returns:
            Loss value
        """
        # Variance must be positive
        var = torch.clamp(var, min=1e-6)
        
        # Compute negative log-likelihood
        nll = 0.5 * (torch.log(var) + (input - target)**2 / var)
        
        if full:
            return nll
        return torch.mean(nll)
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model predictions including classification logits and property estimates
            targets: Ground truth values
            mask: Mask where True values are valid predictions and False are padding
            
        Returns:
            Dictionary of losses and total loss
        """
        batch_size, seq_len = predictions['sbt_logits'].shape[:2]
        
        # Apply mask if provided
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=predictions['sbt_logits'].device)
            
        # Expand mask for element-wise operations
        expanded_mask = mask.unsqueeze(-1).float()
        
        # Classification loss
        sbt_loss = torch.zeros(1, device=predictions['sbt_logits'].device)
        if 'sbt_target' in targets:
            cls_loss = self.cls_criterion(
                predictions['sbt_logits'].reshape(-1, self.n_soil_classes),
                targets['sbt_target'].reshape(-1)
            )
            # Apply mask
            cls_loss = cls_loss.reshape(batch_size, seq_len) * mask.float()
            sbt_loss = cls_loss.sum() / mask.float().sum().clamp(min=1.0)
            
        # Property regression losses
        property_losses = {}
        total_loss = sbt_loss
        
        if self.enable_uncertainty:
            # Uncertainty-aware regression losses
            for name in self.property_weights.keys():
                if name in targets and name in predictions:
                    # Get predictions and targets
                    pred = predictions[name]
                    target = targets[name]
                    uncertainty = predictions.get(f'{name}_uncertainty')
                    
                    if uncertainty is not None:
                        # Compute NLL loss with uncertainty
                        loss = self.gaussian_nll_loss(pred, target, uncertainty, full=True)
                        # Apply mask
                        masked_loss = (loss * expanded_mask).sum() / expanded_mask.sum().clamp(min=1.0)
                        # Weight by property importance
                        weighted_loss = masked_loss * self.property_weights[name]
                        
                        property_losses[name] = weighted_loss
                        total_loss = total_loss + weighted_loss
        else:
            # Standard MSE regression losses
            for name in self.property_weights.keys():
                if name in targets and name in predictions:
                    # Get predictions and targets
                    pred = predictions[name]
                    target = targets[name]
                    
                    # Compute MSE loss
                    loss = F.mse_loss(pred, target, reduction='none')
                    # Apply mask
                    masked_loss = (loss * expanded_mask).sum() / expanded_mask.sum().clamp(min=1.0)
                    # Weight by property importance
                    weighted_loss = masked_loss * self.property_weights[name]
                    
                    property_losses[name] = weighted_loss
                    total_loss = total_loss + weighted_loss
                    
        # Collect all losses
        losses = {
            'total': total_loss,
            'sbt': sbt_loss,
            **property_losses
        }
        
        return losses


# Training function
def train_cpt_transformer(
    model: CPTTransformer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: CPTLoss,
    device: torch.device,
    epochs: int = 100,
    early_stopping_patience: int = 10,
    checkpointing_path: str = './checkpoints'
):
    """
    Train the CPTTransformer model with early stopping and checkpointing.
    
    Args:
        model: CPTTransformer model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to use
        epochs: Number of epochs
        early_stopping_patience: Patience for early stopping
        checkpointing_path: Path to save checkpoints
    """
    import os
    from pathlib import Path
    import time
    
    # Create checkpointing directory
    Path(checkpointing_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_metrics = {}
        
        start_time = time.time()
        
        # Training loop
        for batch in train_dataloader:
            # Move batch to device
            cpt_data = batch['cpt_data'].to(device)
            depth = batch['depth'].to(device)
            coords = batch['coords'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)
                
            region_ids = batch.get('region_ids', None)
            if region_ids is not None:
                region_ids = region_ids.to(device)
                
            has_u2 = batch.get('has_u2', None)
            if has_u2 is not None:
                has_u2 = has_u2.to(device)
                
            # Forward pass
            predictions = model(cpt_data, depth, coords, region_ids, has_u2, mask)
            
            # Compute loss
            targets = {k: v.to(device) for k, v in batch.items() if k in 
                      ['sbt_target', 'qt', 'Fr', 'sigma_v0', 'su', 'phi', 'E', 'permeability']}
            
            losses = criterion(predictions, targets, mask)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            train_loss += losses['total'].item()
            for name, loss in losses.items():
                if name not in train_metrics:
                    train_metrics[name] = 0.0
                train_metrics[name] += loss.item()
                
        # Calculate average training metrics
        train_loss /= len(train_dataloader)
        for name in train_metrics:
            train_metrics[name] /= len(train_dataloader)
            
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_metrics = {}
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                cpt_data = batch['cpt_data'].to(device)
                depth = batch['depth'].to(device)
                coords = batch['coords'].to(device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(device)
                    
                region_ids = batch.get('region_ids', None)
                if region_ids is not None:
                    region_ids = region_ids.to(device)
                    
                has_u2 = batch.get('has_u2', None)
                if has_u2 is not None:
                    has_u2 = has_u2.to(device)
                    
                # Forward pass
                predictions = model(cpt_data, depth, coords, region_ids, has_u2, mask)
                
                # Compute loss
                targets = {k: v.to(device) for k, v in batch.items() if k in 
                          ['sbt_target', 'qt', 'Fr', 'sigma_v0', 'su', 'phi', 'E', 'permeability']}
                
                losses = criterion(predictions, targets, mask)
                
                # Update metrics
                val_loss += losses['total'].item()
                for name, loss in losses.items():
                    if name not in val_metrics:
                        val_metrics[name] = 0.0
                    val_metrics[name] += loss.item()
                    
        # Calculate average validation metrics
        val_loss /= len(val_dataloader)
        for name in val_metrics:
            val_metrics[name] /= len(val_dataloader)
            
        # Update LR scheduler
        scheduler.step(val_loss)
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, os.path.join(checkpointing_path, 'best_model.pt'))
        else:
            patience_counter += 1
            
        # Print epoch metrics
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Print detailed metrics
        print("Training Metrics:")
        for name, value in train_metrics.items():
            print(f"  {name}: {value:.4f}")
            
        print("Validation Metrics:")
        for name, value in val_metrics.items():
            print(f"  {name}: {value:.4f}")
            
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss
        }, os.path.join(checkpointing_path, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Check if early stopping criterion is met
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
    # Load best model
    checkpoint = torch.load(os.path.join(checkpointing_path, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


# Helper function for regional calibration
def calibrate_model_for_region(
    model: CPTTransformer,
    calibration_dataloader: torch.utils.data.DataLoader,
    region_id: int,
    device: torch.device,
    learning_rate: float = 1e-4,
    epochs: int = 20
):
    """
    Fine-tune the model for a specific geological region using regional calibration data.
    
    Args:
        model: Pre-trained CPTTransformer model
        calibration_dataloader: DataLoader for calibration data from the target region
        region_id: ID of the region to calibrate for
        device: Device to use
        learning_rate: Learning rate for fine-tuning
        epochs: Number of epochs for fine-tuning
    
    Returns:
        Calibrated model
    """
    # Freeze most model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze region-specific embeddings
    model.region_embedding.weight[region_id].requires_grad = True
    
    # Unfreeze output layers
    for param in model.sbt_classifier.parameters():
        param.requires_grad = True
        
    for head in model.property_heads.values():
        for param in head.parameters():
            param.requires_grad = True
            
    if model.enable_uncertainty:
        for head in model.uncertainty_heads.values():
            for param in head.parameters():
                param.requires_grad = True
                
    # Setup optimizer (only for unfrozen parameters)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Loss function
    criterion = CPTLoss(enable_uncertainty=model.enable_uncertainty)
    
    # Fine-tuning loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in calibration_dataloader:
            # Move batch to device
            cpt_data = batch['cpt_data'].to(device)
            depth = batch['depth'].to(device)
            coords = batch['coords'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)
                
            # Set region ID for all samples in batch
            batch_size = cpt_data.shape[0]
            batch_region_ids = torch.full((batch_size,), region_id, dtype=torch.long, device=device)
            
            has_u2 = batch.get('has_u2', None)
            if has_u2 is not None:
                has_u2 = has_u2.to(device)
                
            # Forward pass
            predictions = model(cpt_data, depth, coords, batch_region_ids, has_u2, mask)
            
            # Compute loss
            targets = {k: v.to(device) for k, v in batch.items() if k in 
                      ['sbt_target', 'qt', 'Fr', 'sigma_v0', 'su', 'phi', 'E', 'permeability']}
            
            losses = criterion(predictions, targets, mask)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            total_loss += losses['total'].item()
            
        # Print epoch loss
        avg_loss = total_loss / len(calibration_dataloader)
        print(f"Region {region_id} - Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    # Re-freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model


# Example of usage - preprocessing and model creation
def create_cpt_transformer_pipeline(
    input_dim: int = 3,
    enable_uncertainty: bool = True,
    n_layers: int = 4,
    d_model: int = 128,
    n_heads: int = 8
):
    """
    Create a CPT data processing and transformer model pipeline.
    
    Args:
        input_dim: Number of input channels (typically 2 or 3)
        enable_uncertainty: Whether to enable uncertainty quantification
        n_layers: Number of transformer layers
        d_model: Model dimension
        n_heads: Number of attention heads
        
    Returns:
        Model and criterion
    """
    # Create model
    model = CPTTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4,
        dropout=0.1,
        max_depth=100.0,
        n_soil_classes=9,  # Robertson 9-class system
        regional_embedding_dim=16,
        n_regions=10,
        enable_uncertainty=enable_uncertainty
    )
    
    # Create loss function
    criterion = CPTLoss(
        n_soil_classes=9,
        enable_uncertainty=enable_uncertainty,
        label_smoothing=0.1,
        property_weights={
            'qt': 1.0,
            'Fr': 1.0,
            'sigma_v0': 0.5,
            'su': 1.0,
            'phi': 1.0,
            'E': 0.8,
            'permeability': 0.5
        }
    )
    
    return model, criterion


if __name__ == "__main__":
    # Example configuration
    import torch
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and criterion
    model, criterion = create_cpt_transformer_pipeline(
        input_dim=2,  # Only qc and fs (no u2)
        enable_uncertainty=True
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"CPTTransformer Model Summary:")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Model structure:\n{model}")
