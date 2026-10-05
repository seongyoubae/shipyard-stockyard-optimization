"""Shared limits and supported encoder names for the retained research network."""

MAX_SOURCE = 30
MAX_DEST = 30

ENCODER_TYPES = (
    "none",
    "mlp",
    "attention",
    "standard_gru",
    "standard_lstm",
    "pa_lstm",
    "pi_mlp_add",
    "pi_mlp_concat",
    "pi_attention",
    "pi_attention_concat",
    "pi_gru_add",
    "pi_gru_concat",
    "pi_gru_mult",
    "pi_gru_gated_add",
    "pi_gru_film",
    "pi_lstm_add",
    "pi_lstm_concat",
    "pi_lstm_mult",
    "pi_lstm_gated_add",
    "pi_lstm_film",
)

ENCODER_ALIASES = {
    "lstm": "standard_lstm",
    "priority": "pa_lstm",
    "pi_lstm_gated": "pi_lstm_gated_add",
}
