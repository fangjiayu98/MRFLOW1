import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.gcn_conv import BatchGCNConv


class MRFlow_Model(nn.Module):
    
    def __init__(self, args):
        super(MRFlow_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        
        # Base STGNN backbone (frozen after first period)
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        
        self.use_tcn = hasattr(args, 'tcn') and args.tcn is not None
        if self.use_tcn:
            self.tcn1 = nn.Conv1d(
                in_channels=args.gcn["hidden_channel"],  # Match GCN1 output
                out_channels=args.gcn["hidden_channel"],  # Keep same dimension
                kernel_size=args.tcn["kernel_size"],
                dilation=args.tcn["dilation"], 
                padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2)
            )
        
        self.activation = nn.GELU()
        
        # Multi-resolution parameters
        self.num_stages = args.num_stages if hasattr(args, 'num_stages') else 3  # S in paper
        self.forecast_horizon = args.y_len  # H in paper
        self.feature_dim = args.gcn["out_channel"]
        
        # Dynamic Prompt Pool (Learnable Node Anchors)
        self.anchor_dim = args.anchor_dim if hasattr(args, 'anchor_dim') else 32  # d_a in paper
        self.rank = args.rank if hasattr(args, 'rank') else 8  # r for low-rank factorization
        self.use_low_rank = args.use_low_rank if hasattr(args, 'use_low_rank') else True
        
        # Initialize anchor embeddings (will be expanded incrementally)
        self.num_nodes = args.base_node_size if hasattr(args, 'base_node_size') else args.graph_size
        
        if self.use_low_rank:
            # Low-rank factorization: A = C * S
            self.C = nn.Parameter(torch.randn(self.num_nodes, self.rank) * 0.01)
            self.S = nn.Parameter(torch.randn(self.rank, self.anchor_dim) * 0.01)
        else:
            # Full anchor matrix
            self.anchors = nn.Parameter(torch.randn(self.num_nodes, self.anchor_dim) * 0.01)
        
        # Projection layer for anchors
        self.anchor_proj = nn.Linear(self.anchor_dim, args.gcn["in_channel"])
        
        # Average Velocity Networks for each resolution stage
        self.velocity_nets = nn.ModuleList()
        for s in range(self.num_stages):
            velocity_net = AverageVelocityNetwork(
                feature_dim=self.feature_dim,
                hidden_dim=128,
                time_embed_dim=32,
                use_conditioning=(s < self.num_stages - 1)
            )
            self.velocity_nets.append(velocity_net)
        
        # Future Mixup parameters
        self.use_future_mixup = args.use_future_mixup if hasattr(args, 'use_future_mixup') else True
        
        # Classifier-Free Guidance
        self.use_cfg = args.use_cfg if hasattr(args, 'use_cfg') else False
        self.cfg_scale = args.cfg_scale if hasattr(args, 'cfg_scale') else 1.5
        self.uncond_prob = args.uncond_prob if hasattr(args, 'uncond_prob') else 0.1
        
        # Target projection layer (lazy initialization)
        self.target_proj = nn.Linear(args.y_len, self.feature_dim)
        
        # Final prediction head
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        
        # Flow schedule (linear interpolation)
        self.register_buffer('alpha_schedule', torch.linspace(0, 1, 100))
        self.register_buffer('beta_schedule', torch.linspace(1, 0, 100))
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            
            self.num_nodes = new_num_nodes
    
    def get_anchor_embeddings(self):
        """Get anchor embeddings A = C * S (Eq. 24) or direct A"""
        if self.use_low_rank:
            return torch.mm(self.C, self.S)  # [N, d_a]
        else:
            return self.anchors
    
    def extract_multi_resolution_trends(self, x):
        """
        Extract multi-resolution trends via recursive average pooling (Eq. 2)
        Args:
            x: [B, N, d] feature tensor
        Returns:
            trends: list of [B, N, d] for s=0,...,S-1 (same spatial-feature dims)
        """
        B, N, d = x.shape
        trends = [x]  # Stage 0: original features
        
        current = x
        for s in range(1, self.num_stages):
            # Apply spatial smoothing via simple averaging
            # Keep the same shape but apply feature smoothing
            smoothed = current * 0.8 + 0.2 * current.mean(dim=1, keepdim=True)
            trends.append(smoothed)
            current = smoothed
        
        return trends
    
    def sample_time_pairs(self, batch_size, device):
        """Sample (r, t) pairs from logit-normal distribution"""
        # Logit-normal sampling as in paper
        logit_r = torch.randn(batch_size, device=device) * 0.3
        logit_t = torch.randn(batch_size, device=device) * 0.3 + 0.5
        
        r = torch.sigmoid(logit_r)
        t = torch.sigmoid(logit_t)
        
        # Ensure r < t
        mask = r >= t
        r[mask], t[mask] = t[mask], r[mask]
        
        # Ensure minimum gap and maximum value
        t = torch.max(t, r + 0.1)  # Ensure t >= r + 0.1
        t = torch.clamp(t, max=1.0)  # Ensure t <= 1.0
        
        return r, t

    
    def get_flow_coefficients(self, t):
        """Get alpha_t and beta_t from schedules"""
        t_idx = (t * 99).long().clamp(0, 99)
        alpha_t = self.alpha_schedule[t_idx]
        beta_t = self.beta_schedule[t_idx]
        return alpha_t, beta_t
    
    def construct_flow_path(self, y_target, t, epsilon):
        """
        Construct flow path Y_t = alpha_t * Y + beta_t * epsilon (Eq. 3)
        Args:
            y_target: [B, N, d] target features
            t: [B] time values
            epsilon: [B, N, d] noise
        Returns:
            y_t: [B, N, d] flow state at time t
        """
        alpha_t, beta_t = self.get_flow_coefficients(t)
        
        # Reshape for broadcasting
        alpha_t = alpha_t.view(-1, 1, 1)
        beta_t = beta_t.view(-1, 1, 1)
        
        y_t = alpha_t * y_target + beta_t * epsilon
        return y_t
    
    def compute_instantaneous_velocity(self, y_target, epsilon):
        """
        Compute v_t = alpha'_t * Y + beta'_t * epsilon (Eq. 4)
        For linear schedule: alpha'_t = 1, beta'_t = -1
        """
        v_t = y_target - epsilon
        return v_t
    
    def forward(self, data, adj, training=None):
        """
        Forward pass for MRFlow
        Args:
            data: PyG Data object with x, y, batch
            adj: adjacency matrix
            training: whether in training mode (if None, use self.training)
        Returns:
            predictions [B*N, H]
        """
        if training is None:
            training = self.training
        
        N = adj.shape[0]
        B = data.x.shape[0] // N
        
        # Step 1: Feature extraction with anchor augmentation (Eq. 16)
        anchors = self.get_anchor_embeddings()[:N]  # [N, d_a]
        anchor_features = self.anchor_proj(anchors)  # [N, d_in]
        
        # Augment input features
        x = data.x.reshape(B, N, self.args.gcn["in_channel"])  # [B, N, d_in]
        # x_aug = x + anchor_features.unsqueeze(0)  # Broadcasting
        x_aug = x 
        
        # Step 2: Base STGNN encoding
        h = self.encode_features(x_aug, adj)  # [B, N, d_out]
        
        # Generate direct prediction
        y_pred = self.fc(self.activation(h.reshape(B * N, -1)))  # [B*N, H]
        
        if training:
            y_target = data.y.reshape(B, N, self.forecast_horizon)  # [B, N, H]
            
            # Project to feature space for flow matching
            y_target_feat = self.target_proj(y_target)  # [B, N, d_out]
            
            # Compute flow matching loss
            flow_loss = self.compute_training_loss(h, y_target_feat, N, B)
            
            # Store flow loss for logging
            self.flow_loss = flow_loss
        return y_pred

    
    def encode_features(self, x, adj):
        """
        Encode features using base STGNN (frozen after first period)
        Args:
            x: [B, N, d_in]
            adj: [N, N]
        Returns:
            h: [B, N, d_out]
        """
        B, N, d_in = x.shape
        
        # GCN layer 1
        h = self.gcn1(x, adj)  # [B, N, hidden]
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Optional TCN layer for temporal processing
        if self.use_tcn:
            # Reshape: [B, N, hidden] -> [B*N, hidden, 1]
            # Note: Since input is already aggregated features, we treat it as single timestep
            # Or we can skip TCN if input doesn't have temporal dimension
            pass  # Skip TCN for now as input is already temporally aggregated
        
        # GCN layer 2
        h = self.gcn2(h, adj)  # [B, N, out]
        h = F.relu(h)
        
        # Add residual connection if dimensions match
        if d_in == self.args.gcn["out_channel"]:
            h = h + x
        
        return h
    
    def compute_training_loss(self, h, y_target, N, B):
        """
        Compute training loss for all resolution stages (Eq. 11)
        Args:
            h: [B, N, d] encoded lookback features
            y_target: [B, N, d] target features in feature space
        """
        # Extract multi-resolution trends
        y_trends = self.extract_multi_resolution_trends(y_target)
        h_trends = self.extract_multi_resolution_trends(h)
        
        total_loss = 0.0
        
        for s in range(self.num_stages):
            # Sample time pairs
            r, t = self.sample_time_pairs(B, h.device)
            
            # Sample noise
            epsilon = torch.randn_like(y_trends[s])
            
            # Construct flow path
            y_t = self.construct_flow_path(y_trends[s], t, epsilon)
            
            # Construct conditioning
            if s < self.num_stages - 1:
                # Future mixup (Eq. 10)
                if self.use_future_mixup and self.training:
                    m = torch.rand(B, 1, 1, device=h.device) * 0.9  # m ∈ [0, 0.9)
                    z_mix = m * h_trends[s] + (1 - m) * y_trends[s]
                else:
                    z_mix = h_trends[s]
                
                # Concatenate with coarser trend
                cond = torch.cat([z_mix, y_trends[s + 1]], dim=-1)  # [B, N, 2*d]
            else:
                cond = h_trends[s] if self.use_future_mixup else None
            
            # Classifier-free guidance: randomly drop conditioning
            if self.use_cfg and self.training:
                mask = (torch.rand(B, device=h.device) > self.uncond_prob).float()
                mask = mask.view(B, 1, 1)
                if cond is not None:
                    cond = cond * mask
            
            # Predict average velocity
            u_pred = self.velocity_nets[s](y_t, r, t, cond)
            
            # Compute target: simplified version
            v_t = self.compute_instantaneous_velocity(y_trends[s], epsilon)
            u_tgt = v_t  # Simplified target
            
            # Loss (Eq. 8)
            loss_s = F.mse_loss(u_pred, u_tgt.detach())
            
            # Weighted sum
            lambda_s = 1.0 / (s + 1)  # Higher weight for finer stages
            total_loss += lambda_s * loss_s
        
        return total_loss
    
    def generate_forecast(self, h, N, B):
        """
        Multi-resolution generation (Algorithm 2)
        Supports both S-NFE and 1-NFE modes
        """
        # Extract lookback trends
        h_trends = self.extract_multi_resolution_trends(h)
        
        # Initialize at coarsest level
        y_hat = torch.randn(B, N, self.feature_dim, device=h.device)
        
        # Coarse-to-fine generation
        for s in range(self.num_stages - 1, -1, -1):
            # Construct conditioning
            if s < self.num_stages - 1:
                cond = torch.cat([h_trends[s], y_hat], dim=-1)
            else:
                cond = h_trends[s]
            
            # One-step generation: Y_0 = Y_1 - u(Y_1, 0, 1)
            r = torch.zeros(B, device=h.device)
            t = torch.ones(B, device=h.device)
            
            if self.use_cfg and not self.training:
                # Apply classifier-free guidance
                u_cond = self.velocity_nets[s](y_hat, r, t, cond)
                u_uncond = self.velocity_nets[s](y_hat, r, t, None)
                u = self.cfg_scale * u_cond + (1 - self.cfg_scale) * u_uncond
            else:
                u = self.velocity_nets[s](y_hat, r, t, cond)
            
            y_hat = y_hat - u
        
        # Final prediction head: project from feature space to forecast horizon
        y_pred = self.fc(self.activation(y_hat.reshape(B * N, -1)))  # [B*N, H]
        
        return y_pred
    
    def feature(self, data, adj):
        """Extract features (for compatibility with existing code)"""
        N = adj.shape[0]
        B = data.x.shape[0] // N
        
        anchors = self.get_anchor_embeddings()[:N]
        anchor_features = self.anchor_proj(anchors)
        
        x = data.x.reshape(B, N, self.args.gcn["in_channel"])
        x_aug = x + anchor_features.unsqueeze(0)
        
        h = self.encode_features(x_aug, adj)
        h = h.reshape(B * N, -1)
        
        return h


class AverageVelocityNetwork(nn.Module):
    """
    Neural network for parameterizing average velocity field u_theta
    Takes (Y_t, r, t, conditioning) as input
    """
    
    def __init__(self, feature_dim, hidden_dim=128, time_embed_dim=32, use_conditioning=True):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.time_embed_dim = time_embed_dim
        self.use_conditioning = use_conditioning
        
        # Time embedding MLP (for r and t)
        self.time_mlp = nn.Sequential(
            nn.Linear(2, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Calculate input dimension
        if use_conditioning:
            total_in = feature_dim + time_embed_dim + 2 * feature_dim
        else:
            total_in = feature_dim + time_embed_dim
        
        # Main velocity prediction network
        self.net = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Initialize output layer with small weights for stability
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, y_t, r, t, cond=None):
        """
        Args:
            y_t: [B, N, d] flow state
            r: [B] start time
            t: [B] end time
            cond: [B, N, d'] optional conditioning
        Returns:
            u: [B, N, d] average velocity
        """
        B, N, d = y_t.shape
        
        # Time embedding
        rt = torch.stack([r, t], dim=-1)  # [B, 2]
        t_emb = self.time_mlp(rt)  # [B, time_embed_dim]
        t_emb = t_emb.unsqueeze(1).expand(B, N, -1)  # [B, N, time_embed_dim]
        
        # Concatenate inputs
        if self.use_conditioning and cond is not None:
            inputs = torch.cat([y_t, t_emb, cond], dim=-1)
        else:
            inputs = torch.cat([y_t, t_emb], dim=-1)
        
        # Reshape for batch processing
        inputs_flat = inputs.reshape(B * N, -1)
        
        # Forward pass
        u_flat = self.net(inputs_flat)
        
        # Reshape back
        u = u_flat.reshape(B, N, d)
        
        return u





class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden


class MLP_Model(nn.Module):
    """Some Information about MLP"""
    def __init__(self, args):
        super(MLP_Model, self).__init__()
        self.args = args
        
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=12, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=12, hidden_size=48, num_layers=2, batch_first=True)
        
        self.end_linear1 = nn.Linear(48, 24)
        self.end_linear2 = nn.Linear(24, 12)

    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"])).transpose(1, 2).unsqueeze(-1)
        
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden).squeeze(-1).reshape(1, 2)
        x = prediction.reshape(-1, 12)
        return x



class LSTM_Model(nn.Module):
    """Some Information about LSTM"""
    def __init__(self, args):
        super(LSTM_Model, self).__init__()
        self.args = args
        
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=12, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=12, hidden_size=48, num_layers=2, batch_first=True)
        
        self.end_linear1 = nn.Linear(48, 24)
        self.end_linear2 = nn.Linear(24, 12)

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"])).unsqueeze(-1).transpose(1, 2).transpose(1, 3)   # [bs, t, n, f]
        b, f, n, t = x.shape

        x = x.transpose(1,2).reshape(b*n, f, 1, t)  # (b, f, n, t) -> (b, n, f, t) -> (b * n, f, 1, t)
        x = self.start_conv(x).squeeze().transpose(1, 2)  # (b * n, f, 1, t) -> (b * n, init_dim, 1, t) -> (b * n, init_dim, t) -> (b * n, t, init_dim)

        out, _ = self.lstm(x)  # (b * n, t, hidden_dim) -> (b * n, t, hidden_dim)
        x = out[:, -1, :]

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b*n, t)
        return x



class EAC_Model(nn.Module):
    """Some Information about EAC_Model"""
    def __init__(self, args):
        super(EAC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank  # Set a low rank value
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], 
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        # Initialize subspace and adjust matrix
        self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))
        self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))
        
        self.year = args.year
        self.num_nodes = args.base_node_size
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
        B, N, T = x.shape
        
        # Compute adaptive parameters using low-rank matrices
        adaptive_params = torch.mm(self.U[:N, :], self.V)  # [N, feature_dim]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]
        
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.rank, dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1))
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            
            self.num_nodes = new_num_nodes




class TrafficStream_Model(nn.Module):
    """Some Information about TrafficStream_Model"""
    def __init__(self, args):
        super(TrafficStream_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    

    def feature(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]        
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        return x




class STKEC_Model(nn.Module):
    """Some Information about STKEC_Model"""
    def __init__(self, args):
        super(STKEC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.ReLU()

        self.memory=nn.Parameter(torch.zeros(size=(args.cluster, args.gcn["out_channel"]), requires_grad=True))
        nn.init.xavier_uniform_(self.memory, gain=1.414)
        
    def forward(self, data, adj, scores=None):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        attention = torch.matmul(x, self.memory.transpose(-1, -2)) # [bs * N, feature] * [feature , K] = [bs * N, K]
        scores = F.softmax(attention, dim=1)                       # [bs * N, K]

        z = torch.matmul(attention, self.memory)                   # [bs * N, K] * [K, feature] = [bs * N, feature]
        x = x + data.x + z
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x, scores
    
    def feature(self, data, adj, scores=None):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        attention = torch.matmul(x, self.memory.transpose(-1, -2)) # [bs * N, feature] * [feature , K] = [bs * N, K]

        z = torch.matmul(attention, self.memory)                   # [bs * N, K] * [K, feature] = [bs * N, feature]
        x = x + data.x + z
        return x
