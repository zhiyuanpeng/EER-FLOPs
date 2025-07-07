def flops_decoder(input_tokens, output_tokens, num_layers, dimensions, ffsize, ratio):
    """
    Estimate the FLOPs (floating point operations) for a decoder-only Transformer model.

    Parameters:
        input_tokens (int): Number of tokens in the input sequence (context length).
        output_tokens (int): Number of tokens to be generated (output length).
        num_layers (int): Number of decoder layers.
        dimensions (int): Hidden dimension size of the model.
        ffsize (int): Feed-forward network size.
        ratio (int): GQA ratio — number of query heads per key/value head (e.g., 1 = MHA, 8 = GQA-8).

    Returns:
        float: Estimated total FLOPs for both context and generation phases.
    """

    # Total parameters per decoder layer
    N = 2 * dimensions * num_layers * ((1+1/ratio) * dimensions + ffsize)

    # FLOPs for processing the input tokens (context phase)
    ctx = 2 * N * input_tokens + 4 * num_layers * (input_tokens**2) / ratio * dimensions

    # FLOPs for generating the output tokens (autoregressive decoding phase)
    opt = 2 * N * output_tokens + 2 * num_layers / ratio * dimensions * (2 * output_tokens * input_tokens + output_tokens * (output_tokens - 1))

    return ctx + opt

def flops_encoder_decoder(input_tokens, output_tokens, num_layers, dimensions, ffsize):
    """
    Estimate the FLOPs for an encoder-decoder Transformer model (e.g., T5, BART).

    Parameters:
        input_tokens (int): Number of tokens in the input sequence.
        output_tokens (int): Number of tokens to be generated.
        num_layers (int): Number of layers in both encoder and decoder.
        dimensions (int): Hidden dimension size of the model.
        ffsize (int): Feed-forward network size.

    Returns:
        float: Estimated total FLOPs including encoder, decoder, and cross-attention.
    """
    
    # Total parameters in encoder layers
    N = 2 * dimensions * num_layers * (2 * dimensions + ffsize)

    # Total parameters in decoder layers
    N_dec = 2 * dimensions * num_layers * (3 * dimensions + ffsize)

    # FLOPs for processing the input tokens in the encoder
    ctx = 2 * N * input_tokens + 4 * num_layers * (input_tokens**2) * dimensions

    # FLOPs for cross-attention (decoder attending to encoder outputs)
    crosskv = 4 * num_layers * input_tokens * dimensions * dimensions

    # FLOPs for autoregressive decoding in the decoder
    opt = 2 * N_dec * output_tokens + 2 * num_layers * dimensions * (2 * output_tokens * input_tokens + output_tokens * (output_tokens - 1))

    return ctx + crosskv + opt