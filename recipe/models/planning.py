import torch
from torch import nn
import math
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoConfig, AutoTokenizer
import logging
from typing import Optional, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleEncoder(nn.Module, ABC):
    """Abstract base class for encoding individual sequences."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        # Learnable vector to use when no hidden state is provided
        self.default_hidden = nn.Parameter(torch.randn(1, 1, hidden_size))
    
    def get_hidden_state(self, hidden_state: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        """Get hidden state vector, either provided or learned."""
        if hidden_state is not None:
            return hidden_state.unsqueeze(1)  # [batch_size, 1, hidden_size]
        return self.default_hidden.expand(batch_size, 1, -1)
    
    @abstractmethod
    def forward(self, sequences: torch.Tensor, mask: Optional[torch.Tensor] = None,
               hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

class SampleAggregator(nn.Module, ABC):
    """Abstract base class for aggregating multiple encoded samples."""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, encoded_samples: torch.Tensor) -> torch.Tensor:
        """
        Aggregate multiple encoded samples into a single vector.
        Args:
            encoded_samples: tensor of shape [batch_size, K, hidden_size]
        Returns:
            tensor of shape [batch_size, hidden_size]
        """
        pass

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embeddings from 'Attention is All You Need'.
    """
    def __init__(self, max_seq_length: int, hidden_size: int):
        """
        Initialize position embeddings.
        
        Args:
            max_seq_length: Maximum sequence length
            hidden_size: Size of the embeddings
        """
        super().__init__()
        
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        
        pe = torch.zeros(1, max_seq_length, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_size]
            
        Returns:
            Tensor with position embeddings added
        """
        return x + self.pe[:, :x.size(1)]

class TransformerSampleEncoder(SampleEncoder):
    def __init__(self, hidden_size: int, max_seq_length: int, vocab_size: int, 
                 num_attention_heads: int = 8, pretrained_embeddings: nn.Embedding = None,
                 use_layer_norm: bool = False, num_layers: int = 1):
        super().__init__(hidden_size)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Token embedding (will convert one-hot to hidden_size)
        if pretrained_embeddings is not None:
            # Copy GPT2's embedding table
            self.token_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings.weight.clone(),
                freeze=False  # Allow fine-tuning
            )
        else:
            # Initialize new embedding table
            self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Position embeddings
        self.position_embedding = SinusoidalPositionEmbedding(max_seq_length, hidden_size)
        
        # Transformer encoder with configurable number of layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=4 * hidden_size,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Optional layer norm
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None
        
    def create_attention_mask(self, sequence_mask: Optional[torch.Tensor], batch_size: int, seq_length: int) -> torch.Tensor:
        """Create attention mask that allows hidden state to attend to all valid tokens but not vice versa."""
        if sequence_mask is None:
            return None
        
        # Create attention mask of shape [batch_size, 1 + seq_length, 1 + seq_length]
        attention_mask = torch.zeros((batch_size, 1 + seq_length, 1 + seq_length), 
                                  device=sequence_mask.device, dtype=torch.bool)
        
        # Hidden state (position 0) can attend to valid tokens only and also itself
        attention_mask[:, 0, 1:] = sequence_mask
        attention_mask[:, 0, 0] = True
        
        # Other tokens can attend to each other based on sequence_mask
        attention_mask[:, 1:, 1:] = sequence_mask.unsqueeze(1) & sequence_mask.unsqueeze(2)
        # but they cannot attend to hidden state
        attention_mask[:, 1:, 0] = False
        
        # Repeat mask for each head and fold into batch dimension
        # [batch_size, 1 + seq_length, 1 + seq_length] -> [batch_size * num_heads, 1 + seq_length, 1 + seq_length]
        attention_mask = attention_mask.unsqueeze(1).expand(-1, 8, -1, -1).reshape(-1, 1 + seq_length, 1 + seq_length)
        
        return ~attention_mask  # True means position can be attended to
    
    def forward(self, sequences: torch.Tensor, mask: Optional[torch.Tensor] = None,
               hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = sequences.size(0)
        seq_length = sequences.size(1)
        
        # Get embeddings
        hidden_states = self.token_embedding(sequences)  # [B, T, hidden_size]
        hidden_states = self.position_embedding(hidden_states)
        
        # Get hidden state vector
        hidden_vec = self.get_hidden_state(hidden_state, batch_size)
        
        # Combine hidden vector with sequence embeddings
        combined_states = torch.cat([hidden_vec, hidden_states], dim=1)  # [B, 1+T, hidden_size]
        
        # Create attention mask
        attention_mask = self.create_attention_mask(~mask if mask is not None else None, batch_size, seq_length)
        
        # Apply transformer encoder
        encoded = self.encoder(combined_states, mask=attention_mask)
        
        # Apply layer norm if enabled
        if self.layer_norm is not None:
            encoded = self.layer_norm(encoded)
        
        # Return the hidden state's encoding (first position)
        return encoded[:, 0]  # [B, hidden_size]

class TransformerSampleAggregator(SampleAggregator):
    def __init__(self, hidden_size: int, num_attention_heads: int = 8, 
                 use_layer_norm: bool = False, use_transformer: bool = True,
                 num_layers: int = 1, disable_attention_mask: bool = False,
                 use_cls_token: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_cls_token = use_cls_token
        
        # Only create default_hidden if we're using CLS token
        if use_cls_token:
            self.default_hidden = nn.Parameter(torch.randn(1, 1, hidden_size))
        else:
            self.default_hidden = None
            
        self.disable_attention_mask = disable_attention_mask
        
        # Optional transformer encoder with configurable number of layers
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=4 * hidden_size,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.encoder = None
        
        # Optional layer norm
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None
        
    def get_hidden_state(self, hidden_state: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        if not self.use_cls_token:
            return None
        if hidden_state is not None:
            return hidden_state.unsqueeze(1)  # [batch_size, 1, hidden_size]
        return self.default_hidden.expand(batch_size, 1, -1)
    
    def create_attention_mask(self, batch_size: int, num_samples: int, device: torch.device) -> Optional[torch.Tensor]:
        """Create attention mask for aggregation."""
        if not self.use_cls_token:
            # If not using CLS token, no mask needed (all tokens can attend to all tokens)
            return None
        
        # Create attention mask of shape [batch_size, 1 + num_samples, 1 + num_samples]
        attention_mask = torch.ones((batch_size, 1 + num_samples, 1 + num_samples), 
                                  dtype=torch.bool, device=device)
        
        # Hidden state can attend to all positions
        attention_mask[:, 0, :] = False
        
        # Samples can only attend to other samples, not to hidden state
        attention_mask[:, 1:, 1:] = False
        
        # Repeat mask for each head and fold into batch dimension
        attention_mask = attention_mask.unsqueeze(1).expand(-1, 8, -1, -1).reshape(-1, attention_mask.size(1), attention_mask.size(2))
        
        return attention_mask  # True means position can be attended to
    
    def forward(self, encoded_samples: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        # encoded_samples: [batch_size, K, hidden_size]
        batch_size = encoded_samples.size(0)
        num_samples = encoded_samples.size(1)
        device = encoded_samples.device
        
        if self.use_cls_token:
            # Get hidden state vector and combine with encoded samples
            hidden_vec = self.get_hidden_state(hidden_state, batch_size)
            if hidden_vec is not None:
                hidden_vec = hidden_vec.to(device)
            # Combine hidden vector with encoded samples
            combined_states = torch.cat([hidden_vec, encoded_samples], dim=1)  # [B, 1+K, hidden_size]
        else:
            # Just use the encoded samples
            combined_states = encoded_samples  # [B, K, hidden_size]
        
        # Apply transformer encoder if enabled
        if self.encoder is not None:
            # Create attention mask only if not disabled
            attention_mask = None if self.disable_attention_mask else self.create_attention_mask(batch_size, num_samples, device)
            combined_states = self.encoder(combined_states, mask=attention_mask)
        
        # Apply layer norm if enabled
        if self.layer_norm is not None:
            combined_states = self.layer_norm(combined_states)
        
        # Return either CLS token output or mean pooled output
        if self.use_cls_token:
            return combined_states[:, 0]  # [batch_size, hidden_size]
        else:
            return combined_states.mean(dim=1)  # [batch_size, hidden_size]

class BaseSampleAdapter(nn.Module):
    """Abstract base class for adapters that process samples."""
    def __init__(self, hidden_size, projection_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim or hidden_size
        
        # Down projection and up projection if projection_dim is specified
        if projection_dim and projection_dim != hidden_size:
            self.down_proj = nn.Linear(hidden_size, projection_dim)
            self.up_proj = nn.Linear(projection_dim, hidden_size)
        else:
            self.down_proj = self.up_proj = None
    
    @abstractmethod
    def process_samples(self, samples, hidden_state=None):
        """Process the samples (either tokens or hidden states) through encoder."""
        pass
    
    def forward(self, samples, hidden_state=None):
        """
        Process samples to generate a code vector
        Args:
            samples: tensor of shape either:
                    [batch_size, num_samples, sample_length] for token samples
                    or [batch_size, num_samples, seq_len, hidden_size] for hidden states
            hidden_state: Optional context vector for conditioning
        Returns:
            codes: tensor of shape [batch_size, hidden_size]
        """
        # Process through encoder
        encoded = self.process_samples(samples, hidden_state)
        
        # Reshape to [batch_size, K, projection_dim]
        batch_size = samples.size(0)
        K = samples.size(1)
        encoded = encoded.reshape(batch_size, K, self.projection_dim)
        
        # Aggregate across samples
        codes = self.aggregator(encoded, hidden_state)
        
        # Project back to original dimension if needed
        if self.up_proj is not None:
            codes = self.up_proj(codes)
        
        return codes

class TokenSampleAdapter(BaseSampleAdapter):
    """Adapter that processes token sequences."""
    def __init__(self, hidden_size, vocab_size, 
                 projection_dim=None, pad_token_id=None, pretrained_embeddings=None,
                 use_aggregator_layer_norm=False, disable_aggregator_transformer=False,
                 use_encoder_layer_norm=False, encoder_num_layers=1, aggregator_num_layers=1,
                 disable_attention_mask=False, use_cls_token=False):
        super().__init__(hidden_size, projection_dim)
        
        self.encoder = TransformerSampleEncoder(
            hidden_size=self.projection_dim,
            max_seq_length=500,
            vocab_size=vocab_size,
            num_attention_heads=8,
            pretrained_embeddings=pretrained_embeddings,
            use_layer_norm=use_encoder_layer_norm,
            num_layers=encoder_num_layers
        )
        self.aggregator = TransformerSampleAggregator(
            hidden_size=self.projection_dim,
            num_attention_heads=8,
            use_layer_norm=use_aggregator_layer_norm,
            use_transformer=not disable_aggregator_transformer,
            num_layers=aggregator_num_layers,
            disable_attention_mask=disable_attention_mask,
            use_cls_token=use_cls_token
        )
        self.pad_token_id = pad_token_id
    
    def forward(self, samples, hidden_state=None):
        # print("In forward of TokenSampleAdapter")
        # Process through encoder
        encoded = self.process_samples(samples, hidden_state)
        
        # Reshape to [batch_size, K, projection_dim]
        batch_size = samples.size(0)
        K = samples.size(1)
        encoded = encoded.view(batch_size, K, self.projection_dim)
        
        # Aggregate across samples, passing the same hidden state
        codes = self.aggregator(encoded, hidden_state)
        
        # Project back to original dimension if needed
        if self.up_proj is not None:
            codes = self.up_proj(codes)
        
        return codes
    
    def process_samples(self, samples, hidden_state=None):
        # Create mask for padding tokens
        mask = (samples == self.pad_token_id).reshape(-1, samples.size(-1))
        
        # Process through encoder, passing the hidden state
        # We need to repeat the hidden state for each sample
        if hidden_state is not None:
            batch_size = samples.size(0)
            num_samples = samples.size(1)
            hidden_state = hidden_state.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, hidden_state.size(-1))
        
        # Get encoder output
        encoder_output = self.encoder(
            sequences=samples.reshape(-1, samples.size(-1)),
            mask=mask,
            hidden_state=hidden_state
        )
        
        return encoder_output

class LatentSampleAdapter(BaseSampleAdapter):
    """Adapter that processes hidden state sequences."""
    def __init__(self, hidden_size, continuation_length=20, projection_dim=None,
                 use_aggregator_layer_norm=False, disable_aggregator_transformer=False,
                 encoder_num_layers=1, aggregator_num_layers=1, disable_attention_mask=False,
                 use_cls_token=False):
        super().__init__(hidden_size, projection_dim)
        
        # Initialize encoder for hidden states
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.projection_dim,
            nhead=8,
            dim_feedforward=4 * self.projection_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_num_layers)
        
        self.aggregator = TransformerSampleAggregator(
            hidden_size=self.projection_dim,
            num_attention_heads=8,
            use_layer_norm=use_aggregator_layer_norm,
            use_transformer=not disable_aggregator_transformer,
            num_layers=aggregator_num_layers,
            disable_attention_mask=disable_attention_mask,
            use_cls_token=use_cls_token
        )
        
        # Project input hidden states if needed
        if hidden_size != self.projection_dim:
            self.input_proj = nn.Linear(hidden_size, self.projection_dim)
        else:
            self.input_proj = None
    
    def process_samples(self, samples, hidden_state=None):
        # Project input if needed
        if self.input_proj is not None:
            samples = self.input_proj(samples)
        
        # Reshape to [batch_size * num_samples, seq_length, hidden_size]
        batch_size, num_samples, seq_len, hidden_size = samples.shape
        samples = samples.reshape(-1, seq_len, hidden_size)
        
        # Process through encoder and mean pool
        return self.encoder(samples).mean(dim=1)  # [batch_size * num_samples, hidden_size]

class PretrainedTransformerSampleEncoder(SampleEncoder):
    """Sample encoder that uses a pretrained transformer model."""
    def __init__(self, model_name: str, max_seq_length: int):
        super().__init__()
        # Load pretrained model
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self.max_seq_length = max_seq_length
    
    def forward(self, sequences: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Use pretrained model with mask directly
        outputs = self.model(sequences, attention_mask=mask if mask is not None else None)
        
        # Mean pooling with mask
        if mask is not None:
            # Use mask as is (1s for valid tokens, 0s for padding)
            mask = mask.unsqueeze(-1).expand_as(outputs.last_hidden_state).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = outputs.last_hidden_state.mean(dim=1)
        
        return pooled

class PretrainedTokenSampleAdapter(BaseSampleAdapter):
    """Adapter that uses a pretrained transformer for encoding token sequences."""
    def __init__(self, hidden_size, vocab_size, continuation_length=20, 
                 projection_dim=None, pad_token_id=None, pretrained_model_name="TaylorAI/bge-micro",
                 gpt2_tokenizer=None, use_aggregator_layer_norm=False, 
                 disable_aggregator_transformer=False, use_cls_token=False):
        # Get pretrained model's hidden size
        config = AutoConfig.from_pretrained(pretrained_model_name)
        encoder_hidden_size = config.hidden_size
        
        # Use projection if sizes don't match
        projection_dim = projection_dim or (encoder_hidden_size if encoder_hidden_size != hidden_size else None)
        
        super().__init__(hidden_size, projection_dim)
        
        # Initialize encoder with pretrained model
        self.encoder = PretrainedTransformerSampleEncoder(
            model_name=pretrained_model_name,
            max_seq_length=500
        )
        
        self.aggregator = TransformerSampleAggregator(
            hidden_size=self.projection_dim,
            num_attention_heads=8,
            use_layer_norm=use_aggregator_layer_norm,
            use_transformer=not disable_aggregator_transformer,
            use_cls_token=use_cls_token
        )
        
        # Store both tokenizers
        self.gpt2_tokenizer = gpt2_tokenizer
        self.pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.pad_token_id = pad_token_id
    
    def process_samples(self, samples):
        # First decode the GPT2 tokens to text
        texts = [
            self.gpt2_tokenizer.decode(seq) 
            for seq in samples.view(-1, samples.size(-1))
        ]
        
        # Then encode with the pretrained model's tokenizer
        encoded = self.pretrained_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=500,
            return_tensors='pt'
        ).to(samples.device)
        
        # Use attention mask directly from tokenizer (1 for valid tokens, 0 for padding)
        attention_mask = encoded['attention_mask']
        
        # Process through encoder with mask directly
        return self.encoder(encoded['input_ids'], attention_mask) 

class Planner(ABC):
    """Abstract base class for planners that generate codes for the model."""
    
    def __init__(self, model, adapter, tokenizer, config, device='cuda'):
        self.model = model
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
    
    @abstractmethod
    def forward(
        self,
        prompts: torch.Tensor,
        prompt_masks: torch.Tensor,
        split_index: int,
        num_samples: int,
        continuation_length: int,
        context_hidden: Optional[torch.Tensor] = None,
        debug: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate codes for the given prompts.
        
        Args:
            prompts: Input token ids [batch_size, prompt_length]
            prompt_masks: Attention masks [batch_size, prompt_length]
            split_index: Index at which to split the input
            num_samples: Number of samples to generate
            continuation_length: Length of continuations to generate
            context_hidden: Optional context vector for conditioning
            debug: Whether to print debug information
            
        Returns:
            batch_codes: Generated codes [batch_size, seq_length, hidden_size]
        """
        pass

class BaselinePlanner(Planner):
    """Planner that uses fixed random sequences or states."""
    
    def __init__(self, model, adapter, tokenizer, config, baseline_data, latent_space=False, device='cuda'):
        super().__init__(model, adapter, tokenizer, config, device)
        self.baseline_data = baseline_data  # Either baseline_hiddens or baseline_sequences
        self.latent_space = latent_space
    
    def forward(self, prompts, prompt_masks, split_index, num_samples, continuation_length, context_hidden=None, **kwargs):
        batch_size = prompts.size(0)
        
        if self.latent_space:
            baseline_hiddens = self.baseline_data.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            ).to(self.device)
            return self.adapter(baseline_hiddens, context_hidden)
        else:
            baseline_sequences = self.baseline_data.unsqueeze(0).expand(
                batch_size, -1, -1
            ).to(self.device)
            return self.adapter(baseline_sequences, context_hidden)

class TokenPlanner(Planner):
    """Planner that generates in token space."""
    
    def forward(self, prompts, prompt_masks, split_index, num_samples, continuation_length, context_hidden=None, debug=False, **kwargs):
        # print("In forward of TokenPlanner")
        transformers_logger = logging.getLogger("transformers.generation.utils")
        current_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)

        outputs = self.model.generate(
            prompts,
            attention_mask=prompt_masks,
            max_length=(split_index + 1) + continuation_length,
            num_return_sequences=num_samples,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            top_k=kwargs.get('top_k', 50),
            top_p=kwargs.get('top_p', 0.95),
            codes=None,
            return_dict_in_generate=True,
            output_scores=True,
        )

        continuations = outputs.sequences[:, prompts.size(1):]
        continuations = continuations.reshape(prompts.size(0), num_samples, -1)

        if debug:
            for b in range(prompts.size(0)):
                batch_continuations = []
                for cont in continuations[b]:
                    text = self.tokenizer.decode(cont)
                    batch_continuations.append(text)
                print(f"\nContinuations for batch item {b}:")
                for i, text in enumerate(batch_continuations):
                    print(f"  {i+1}: {text}")

        transformers_logger.setLevel(current_level)
        return self.adapter(continuations, context_hidden)

class LatentPlanner(Planner):
    """Planner that generates in latent space."""
    
    def __init__(self, model, adapter, tokenizer, config, latent_ablation=False, device='cuda'):
        super().__init__(model, adapter, tokenizer, config, device)
        self.latent_ablation = latent_ablation
    
    def forward(self, prompts, prompt_masks, split_index, num_samples, continuation_length, context_hidden=None, debug=False, **kwargs):
        transformers_logger = logging.getLogger("transformers.generation.utils")
        current_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)

        def get_last_hidden_state(outputs):
            last_layer_states = torch.cat([
                states[-1] for states in outputs.hidden_states
            ], dim=1)
            return last_layer_states

        if self.latent_ablation:
            outputs = self.model.generate(
                prompts,
                attention_mask=prompt_masks,
                max_length=(split_index + 1) + 1,
                num_return_sequences=num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                top_k=kwargs.get('top_k', 50),
                top_p=kwargs.get('top_p', 0.95),
                codes=None,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True
            )
            
            last_layer_states = get_last_hidden_state(outputs)
            last_layer_states = last_layer_states[:, prompts.size(1):]
            continuations = last_layer_states.reshape(prompts.size(0), num_samples, 1, self.config.hidden_size)
            continuations = continuations.repeat(1, 1, continuation_length, 1)
        else:
            outputs = self.model.generate(
                prompts,
                attention_mask=prompt_masks,
                max_length=(split_index + 1) + continuation_length,
                num_return_sequences=num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                top_k=kwargs.get('top_k', 50),
                top_p=kwargs.get('top_p', 0.95),
                codes=None,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True
            )
            
            last_layer_states = get_last_hidden_state(outputs)
            last_layer_states = last_layer_states[:, prompts.size(1):]
            continuations = last_layer_states.reshape(prompts.size(0), num_samples, -1, self.config.hidden_size)

        transformers_logger.setLevel(current_level)
        return self.adapter(continuations, context_hidden) 