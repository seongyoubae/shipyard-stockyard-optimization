import torch
import torch.nn as nn
import torch.nn.functional as F
import math

MAX_SOURCE = 30
MAX_DEST = 30

def init_weights(model, init_std):
    if model is None: return
    modules_to_init = model.modules() if not isinstance(model, nn.Sequential) else model
    for m in modules_to_init:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight, gain=init_std)
            if m.out_proj.weight is not None:
                nn.init.xavier_uniform_(m.out_proj.weight, gain=init_std)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, activation=nn.GELU, use_dropout=False, use_layernorm=False):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.use_dropout = use_dropout
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        if self.use_layernorm:
            self.norm = nn.LayerNorm(hidden_dim)
        self.activation = activation()
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        identity = x
        out = self.fc(x)
        if self.use_layernorm:
            out = self.norm(out)
        out = self.activation(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = out + identity
        return out

def build_mlp_with_residuals(input_dim, output_dim, hidden_dim, num_layers, activation=nn.GELU, use_dropout=False,
                             use_layernorm=False):
    if num_layers < 1: return nn.Identity()
    if num_layers == 1: return nn.Linear(input_dim, output_dim)
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    if use_layernorm: layers.append(nn.LayerNorm(hidden_dim))
    layers.append(activation())
    if use_dropout: layers.append(nn.Dropout(p=0.1))
    for _ in range(num_layers - 2):
        layers.append(ResidualBlock(hidden_dim, activation, use_dropout, use_layernorm))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class PiGRUEncoder(nn.Module):
    def __init__(self, embed_dim, mode='add'):
        super().__init__()
        self.mode = mode

        self.prio_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, embed_dim)
        )

        if mode == 'concat':
            self.concat_proj = nn.Linear(embed_dim * 2, embed_dim)
            nn.init.xavier_uniform_(self.concat_proj.weight)
        elif mode == 'gated_add':
            self.gate_generator = nn.Linear(embed_dim, embed_dim)
            nn.init.constant_(self.gate_generator.weight, 0)
            nn.init.constant_(self.gate_generator.bias, 0)
        elif mode == 'film':
            self.film_generator = nn.Linear(embed_dim, embed_dim * 2)
            nn.init.constant_(self.film_generator.weight, 0)
            nn.init.constant_(self.film_generator.bias, 0)

        self.norm = nn.LayerNorm(embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x, raw_features):
        prio_val = raw_features[:, :, 0:1]
        pile_type = raw_features[:, :, -2:-1]

        valid_pile = (pile_type > 0.0).float()

        # 패딩은 prio_encoder 출력부터 원천 차단
        prio_feat = self.prio_encoder(prio_val) * valid_pile

        if self.mode == 'concat':
            fused = torch.cat([x, prio_feat], dim=-1)
            fused = self.concat_proj(fused)
        elif self.mode == 'gated_add':
            raw_gate = torch.sigmoid(self.gate_generator(prio_feat))
            scale = 1.0 + raw_gate
            fused = x + (prio_feat * scale)
        elif self.mode == 'film':
            params = self.film_generator(prio_feat)
            gamma, beta = torch.chunk(params, 2, dim=-1)
            fused = (1.0 + gamma) * x + beta
        elif self.mode == 'mult':
            fused = x * (1.0 + torch.sigmoid(prio_feat))
        else:
            fused = x + prio_feat

        # [핵심] padding row는 GRU 입력 전 fused 단계에서 다시 제거!
        fused = fused * valid_pile

        x_norm = self.norm(fused)
        gru_out, _ = self.gru(x_norm)

        # 출력도 혹시 모를 노이즈 방지를 위해 한 번 더 마스킹하면 안전합니다.
        out = x + self.out_proj(gru_out)
        return out * valid_pile

class StandardLSTMEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        # GRU 대신 LSTM 사용
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)
        init_weights(self, 1.0)

    def forward(self, x, raw_features=None):
        # LSTM은 output과 (hidden, cell)을 반환함. output만 필요.
        lstm_out, _ = self.lstm(self.norm(x))
        # Residual Connection
        return x + self.out_proj(lstm_out)

class StandardGRUEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)
        init_weights(self, 1.0)

    def forward(self, x, raw_features=None):
        gru_out, _ = self.gru(self.norm(x))
        return x + self.out_proj(gru_out)


class MLPEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=None):
        super().__init__()
        self.block = ResidualBlock(embed_dim, use_layernorm=True)
        init_weights(self, 1.0)

    def forward(self, x, raw_features=None):  # P 무시
        return self.block(x)


class AttentionEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        init_weights(self, 1.0)

    def forward(self, x, raw_features=None):  # P 무시
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + self.out_proj(attn_out)


# [수정됨] MLP도 Concat 모드 지원
class PiMLPEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=None, mode='add'):  # mode 인자 추가
        super().__init__()
        self.mode = mode

        # 우선순위 처리용 (작은 MLP)
        self.prio_encoder = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, embed_dim))

        # [New] Concat Projection Layer
        if mode == 'concat':
            self.concat_proj = nn.Linear(embed_dim * 2, embed_dim)
            nn.init.xavier_uniform_(self.concat_proj.weight)

        # 메인 MLP
        self.block = ResidualBlock(embed_dim, use_layernorm=True)
        init_weights(self, 1.0)

    def forward(self, x, raw_features):
        # 1. 우선순위 정보 추출
        prio_val = raw_features[:, :, 0:1]
        prio_feat = self.prio_encoder(prio_val)

        # 2. Fusion (Add vs Concat)
        if self.mode == 'concat':
            fused = torch.cat([x, prio_feat], dim=-1)  # (B, N, 2*D)
            x = self.concat_proj(fused)  # (B, N, D)
        else:
            x = x + prio_feat  # (B, N, D)

        # 3. MLP 통과
        return self.block(x)


# [수정됨] Attention도 Concat 모드 지원
class PiAttentionEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mode='add'):  # mode 인자 추가
        super().__init__()
        self.mode = mode
        self.prio_encoder = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, embed_dim))

        # [New] Concat Projection Layer
        if mode == 'concat':
            self.concat_proj = nn.Linear(embed_dim * 2, embed_dim)
            nn.init.xavier_uniform_(self.concat_proj.weight)

        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        init_weights(self, 1.0)

    def forward(self, x, raw_features):
        prio_val = raw_features[:, :, 0:1]
        prio_feat = self.prio_encoder(prio_val)

        # Fusion (Add vs Concat)
        if self.mode == 'concat':
            fused = torch.cat([x, prio_feat], dim=-1)
            x = self.concat_proj(fused)
        else:
            x = x + prio_feat

        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + self.out_proj(attn_out)


# [수정됨] LSTM도 Concat 모드 지원
class PiLSTMEncoder(nn.Module):
    def __init__(self, embed_dim, mode='add'):
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim

        # 우선순위 인코더
        self.prio_encoder = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, embed_dim))

        # [New] Concat Projection Layer
        if mode == 'concat':
            self.concat_proj = nn.Linear(embed_dim * 2, embed_dim)
            nn.init.xavier_uniform_(self.concat_proj.weight)

        # 기존 모드들 (Gated, Film 등)
        elif self.mode == 'gated_add':
            self.gate_generator = nn.Linear(embed_dim, embed_dim)
            nn.init.constant_(self.gate_generator.weight, 0)
            nn.init.constant_(self.gate_generator.bias, 0)
        elif self.mode == 'film':
            self.film_generator = nn.Linear(embed_dim, embed_dim * 2)
            nn.init.constant_(self.film_generator.weight, 0)
            nn.init.constant_(self.film_generator.bias, 0)

        self.norm = nn.LayerNorm(embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x, raw_features):
        prio_val = raw_features[:, :, 0:1]
        prio_feat = self.prio_encoder(prio_val)

        # Fusion Logic
        if self.mode == 'concat':
            fused = torch.cat([x, prio_feat], dim=-1)
            fused_input = self.concat_proj(fused)
        elif self.mode == 'gated_add':
            gate = torch.tanh(self.gate_generator(prio_feat))
            fused_input = x + (prio_feat * (1.0 + gate))
        elif self.mode == 'film':
            gamma, beta = torch.chunk(self.film_generator(prio_feat), 2, dim=-1)
            fused_input = (1.0 + gamma) * x + beta
        elif self.mode == 'mult':
            fused_input = x * (1.0 + torch.sigmoid(prio_feat))
        else:
            fused_input = x + prio_feat  # Default Add

        x_norm = self.norm(fused_input)
        lstm_out, _ = self.lstm(x_norm)
        return x + self.out_proj(lstm_out)


@torch.jit.script
def fused_lstm_loop(seq_len: int, ih_precomputed: torch.Tensor, p_precomputed: torch.Tensor,
                    hh_weight: torch.Tensor, hh_bias: torch.Tensor, h_init: torch.Tensor,
                    c_init: torch.Tensor) -> torch.Tensor:
    h, c = h_init, c_init
    outputs = []
    for t in range(seq_len):
        txt = ih_precomputed[:, t, :]
        pt = p_precomputed[:, t, :]
        gates = txt + torch.matmul(h, hh_weight.t()) + hh_bias
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
        f_gate = f_gate + pt
        c = (torch.sigmoid(f_gate) * c) + (torch.sigmoid(i_gate) * torch.tanh(c_gate))
        h = torch.sigmoid(o_gate) * torch.tanh(c)
        outputs.append(h)
    return torch.stack(outputs, dim=1)


class PaLstmEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.hidden_size = embed_dim
        self.weight_ih = nn.Linear(embed_dim, 4 * embed_dim)
        self.weight_hh = nn.Linear(embed_dim, 4 * embed_dim)
        self.weight_priority = nn.Linear(1, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters(): weight.data.uniform_(-stdv, stdv)

    def forward(self, x, raw_features):
        prio_val = raw_features[:, :, 0:1]
        batch_size, seq_len, _ = x.size()
        ih_precomputed = self.weight_ih(x)
        p_precomputed = self.weight_priority(prio_val)
        h_init = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_init = torch.zeros(batch_size, self.hidden_size, device=x.device)
        rnn_out = fused_lstm_loop(seq_len, ih_precomputed, p_precomputed, self.weight_hh.weight, self.weight_hh.bias,
                                  h_init, c_init)
        return self.norm(x + rnn_out)


class IdentityEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads): super().__init__()

    def forward(self, x, raw_features=None): return x

class SteelPlateConditionalMLPModel(nn.Module):
    def __init__(self, embed_dim, num_actor_layers, num_critic_layers, actor_init_std, critic_init_std,
                 pile_feature_dim, num_heads, encoder_type='pi_gru_add',
                 use_dropout_actor=False, use_dropout_critic=False, num_from_piles=MAX_SOURCE, num_to_piles=MAX_DEST,
                 use_simple_head=True, **kwargs):
        super().__init__()
        self.encoder_type = encoder_type
        self.num_from_piles = num_from_piles
        self.num_to_piles = num_to_piles
        self.embed_dim = embed_dim
        self.use_simple_head = use_simple_head

        # 1. Pile Embedding
        self.pile_encoder = nn.Sequential(
            nn.Linear(pile_feature_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim)
        )
        init_weights(self.pile_encoder, 1.0)
        self.pile_id_embedding = nn.Parameter(torch.randn(1, 60, embed_dim))
        nn.init.xavier_uniform_(self.pile_id_embedding)

        # 2. Encoder Mapping
        if encoder_type == 'standard_gru':
            self.context_encoder = StandardGRUEncoder(embed_dim)
        elif encoder_type == 'standard_lstm':
            self.context_encoder = StandardLSTMEncoder(embed_dim)
        elif encoder_type == 'mlp':
            self.context_encoder = MLPEncoder(embed_dim, num_heads)
        elif encoder_type == 'attention':
            self.context_encoder = AttentionEncoder(embed_dim, num_heads)
        elif 'pi_mlp' in encoder_type:
            mode = 'concat' if 'concat' in encoder_type else 'add'
            self.context_encoder = PiMLPEncoder(embed_dim, num_heads, mode=mode)
        elif 'pi_attention' in encoder_type:
            mode = 'concat' if 'concat' in encoder_type else 'add'
            self.context_encoder = PiAttentionEncoder(embed_dim, num_heads, mode=mode)
        elif encoder_type == 'pa_lstm':
            self.context_encoder = PaLstmEncoder(embed_dim)
        elif 'pi_lstm' in encoder_type:
            mode = 'concat' if 'concat' in encoder_type else encoder_type.replace('pi_lstm_', '')
            if mode not in ['add', 'concat', 'gated_add', 'film', 'mult']: mode = 'add'
            self.context_encoder = PiLSTMEncoder(embed_dim, mode=mode)
        elif 'pi_gru' in encoder_type:
            mode = encoder_type.replace('pi_gru_', '')
            if mode not in ['add', 'concat', 'gated_add', 'film', 'mult']: mode = 'add'
            self.context_encoder = PiGRUEncoder(embed_dim, mode=mode)
        else:
            self.context_encoder = IdentityEncoder(embed_dim, num_heads)

        # 3. Heads
        self.source_logit_calculator = build_mlp_with_residuals(
            embed_dim * 2, 1, embed_dim, num_actor_layers, nn.GELU, use_dropout_actor, True)
        init_weights(self.source_logit_calculator, actor_init_std)

        self.dest_query_combiner = nn.Sequential(
            nn.LayerNorm(embed_dim * 2), nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(), nn.LayerNorm(embed_dim))
        init_weights(self.dest_query_combiner, actor_init_std)

        self.dest_embeddings = nn.Parameter(torch.randn(self.num_to_piles, embed_dim))
        nn.init.xavier_uniform_(self.dest_embeddings)

        if self.use_simple_head:
            self.dest_scorer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1))
            init_weights(self.dest_scorer, actor_init_std)
            self.dest_kv_combiner, self.dest_attention, self.dest_fc = None, None, None
        else:
            self.dest_kv_combiner = nn.Sequential(nn.LayerNorm(embed_dim * 2), nn.Linear(embed_dim * 2, embed_dim),
                                                  nn.GELU(), nn.LayerNorm(embed_dim))
            init_weights(self.dest_kv_combiner, actor_init_std)
            self.dest_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            init_weights(self.dest_attention, actor_init_std)
            self.dest_fc = nn.Linear(embed_dim, self.num_to_piles)
            init_weights(self.dest_fc, actor_init_std)

        self.critic_net = build_mlp_with_residuals(
            embed_dim, 1, embed_dim, num_critic_layers, nn.GELU, use_dropout_critic, True)
        init_weights(self.critic_net, critic_init_std)

        if not self.use_simple_head:
            self.critic_query = nn.Parameter(torch.randn(1, 1, embed_dim))
            nn.init.xavier_uniform_(self.critic_query)
            self.critic_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            init_weights(self.critic_attention, critic_init_std)
        else:
            self.critic_query, self.critic_attention = None, None

    def actor_parameters(self):
        return list(self.parameters())

    def critic_parameters(self):
        return list(self.critic_net.parameters())

    def forward(self, pile_features, debug=False):
        B, N, _ = pile_features.shape

        # ------------------------------------------------------------
        # 0. Valid pile mask
        # ------------------------------------------------------------
        pile_type_idx = -2
        valid_pile_mask = pile_features[:, :, pile_type_idx] > 0.0
        valid_mask_f = valid_pile_mask.float().unsqueeze(-1)

        # ------------------------------------------------------------
        # 1. Base Embedding
        # ------------------------------------------------------------
        pile_emb = self.pile_encoder(pile_features) + self.pile_id_embedding[:, :N, :]
        pile_emb = pile_emb * valid_mask_f

        # ------------------------------------------------------------
        # 2. Context Encoding
        # ------------------------------------------------------------
        x_encoded = self.context_encoder(pile_emb, raw_features=pile_features)
        x_encoded = x_encoded * valid_mask_f

        # ------------------------------------------------------------
        # 3. Masked global representation
        # ------------------------------------------------------------
        denom = valid_mask_f.sum(dim=1).clamp(min=1.0)
        actor_global_repr = x_encoded.sum(dim=1) / denom

        # ------------------------------------------------------------
        # 4. Source head
        # ------------------------------------------------------------
        source_reprs = x_encoded[:, :self.num_from_piles, :]
        global_expanded_src = actor_global_repr.unsqueeze(1).expand(-1, self.num_from_piles, -1)
        combined_source_input = torch.cat((source_reprs, global_expanded_src), dim=-1)
        source_logits = self.source_logit_calculator(combined_source_input).squeeze(-1)

        # [안전장치 추가] Source가 전부 비어있는 경우 (NaN 방지)
        source_valid_mask = valid_pile_mask[:, :self.num_from_piles]
        no_valid_source = ~source_valid_mask.any(dim=1)
        if no_valid_source.any():
            source_valid_mask = source_valid_mask.clone()
            source_valid_mask[no_valid_source, 0] = True

        source_logits_for_repr = source_logits.masked_fill(~source_valid_mask, -1e9)
        source_policy = F.softmax(source_logits_for_repr, dim=-1)

        expected_source_repr = torch.sum(source_policy.unsqueeze(-1) * source_reprs, dim=1)
        combined_query_input = torch.cat((expected_source_repr, actor_global_repr), dim=-1)
        query_dest = self.dest_query_combiner(combined_query_input).unsqueeze(1)

        # ------------------------------------------------------------
        # 5. Destination head
        # ------------------------------------------------------------
        if self.use_simple_head:
            query_expanded = query_dest.expand(-1, self.num_to_piles, -1)
            dest_reprs = x_encoded[:, self.num_from_piles:, :]
            combined_dest = torch.cat([query_expanded, dest_reprs], dim=-1)
            dest_logits = self.dest_scorer(combined_dest).squeeze(-1)
        else:
            dest_reprs = x_encoded[:, self.num_from_piles:, :]
            dest_valid_mask = valid_pile_mask[:, self.num_from_piles:]

            # [안전장치 추가] Dest가 전부 비어있는 경우 (NaN 방지)
            no_valid_dest = ~dest_valid_mask.any(dim=1)
            if no_valid_dest.any():
                dest_valid_mask = dest_valid_mask.clone()
                dest_valid_mask[no_valid_dest, 0] = True

            dest_emb_expanded = self.dest_embeddings.unsqueeze(0).expand(B, -1, -1)
            combined_dest_input = torch.cat((dest_emb_expanded, dest_reprs), dim=-1)
            dest_kv = self.dest_kv_combiner(combined_dest_input)

            attn_output_dest, _ = self.dest_attention(
                query=query_dest, key=dest_kv, value=dest_kv, key_padding_mask=~dest_valid_mask
            )
            dest_logits = self.dest_fc(attn_output_dest.squeeze(1))

        # ------------------------------------------------------------
        # 6. Critic
        # ------------------------------------------------------------
        if self.use_simple_head:
            critic_input = actor_global_repr
            value = self.critic_net(critic_input)
        else:
            critic_q = self.critic_query.expand(B, -1, -1)
            critic_pooled_output, _ = self.critic_attention(
                query=critic_q, key=x_encoded, value=x_encoded, key_padding_mask=~valid_pile_mask
            )
            value = self.critic_net(critic_pooled_output.squeeze(1))

        return source_logits, dest_logits, value

    @torch.no_grad()
    def act_batch(self, pile_features, source_masks=None, dest_masks=None, greedy=False, debug=False):
        self.eval()
        source_logits, dest_logits, value = self.forward(pile_features, debug=debug)
        if source_masks is not None: source_logits = source_logits.masked_fill(~source_masks, -1e9)
        if dest_masks is not None: dest_logits = dest_logits.masked_fill(~dest_masks, -1e9)
        source_policy = F.softmax(source_logits, dim=-1)
        dest_policy = F.softmax(dest_logits, dim=-1)
        source_policy = source_policy / (source_policy.sum(dim=-1, keepdim=True) + 1e-10)
        dest_policy = dest_policy / (dest_policy.sum(dim=-1, keepdim=True) + 1e-10)
        if greedy:
            selected_source = source_policy.argmax(dim=-1)
            selected_dest = dest_policy.argmax(dim=-1)
        else:
            selected_source = torch.multinomial(source_policy, 1).squeeze(-1)
            selected_dest = torch.multinomial(dest_policy, 1).squeeze(-1)
        chosen_src_logprob = torch.log(source_policy.gather(1, selected_source.unsqueeze(1)) + 1e-10).squeeze(1)
        chosen_dest_logprob = torch.log(dest_policy.gather(1, selected_dest.unsqueeze(1)) + 1e-10).squeeze(1)
        joint_logprob = chosen_src_logprob + chosen_dest_logprob
        actions = torch.stack([selected_source, selected_dest], dim=-1)
        return actions, joint_logprob, value.squeeze(-1), None

    def evaluate(self, batch_pile_features, batch_source_mask, batch_dest_mask, batch_action):
        source_logits, dest_logits, value = self.forward(batch_pile_features)
        if batch_source_mask is not None: source_logits = source_logits.masked_fill(~batch_source_mask, -1e9)
        if batch_dest_mask is not None: dest_logits = dest_logits.masked_fill(~batch_dest_mask, -1e9)
        src_dist = torch.distributions.Categorical(logits=source_logits)
        dst_dist = torch.distributions.Categorical(logits=dest_logits)
        src_logprob = src_dist.log_prob(batch_action[:, 0])
        dst_logprob = dst_dist.log_prob(batch_action[:, 1])
        joint_logprob = src_logprob + dst_logprob
        joint_entropy = src_dist.entropy() + dst_dist.entropy()
        return joint_logprob.unsqueeze(-1), value, joint_entropy.unsqueeze(-1)
