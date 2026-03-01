import gc
import hashlib
import mlx.core as mx
from mlx_vlm.prompt_utils import apply_chat_template
from typing import List, Dict, Union, Generator, Optional
from mlx_vlm import load, generate, stream_generate, prepare_inputs
from mlx_vlm.generate import generate_step

# Default model parameters
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0

class MLX_VLM:
    """
    A wrapper class for MLX Multimodal Model that handles both streaming and non-streaming inference.

    This class provides a unified interface for generating text responses from images and text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(self, model_path: str):
        """
        Initialize the MLX_VLM model.

        Args:
            model_path (str): Path to the model directory containing model weights and configuration.

        Raises:
            ValueError: If model loading fails.
        """
        try:
            self.model, self.processor = load(model_path, lazy=False, trust_remote_code=True)
            self.config = self.model.config
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

        # System prompt KV cache: reuse prefilled KV states across requests
        self._sys_hash = None        # md5 of last system prompt
        self._sys_snapshot = None    # List of (cls, state, meta_state) per layer
        self._sys_n_tokens = 0       # number of tokens in the cached system prefix

    def _snapshot_cache(self, cache: list) -> list:
        """Snapshot cache states using the state / meta_state protocol.

        All cache types (KVCache, ArraysCache, …) implement .state and .meta_state,
        which is the same API used by mlx_lm's save_prompt_cache / load_prompt_cache.
        MLX arrays are reference-counted and immutable once evaluated; subsequent
        cache updates always create new arrays (KVCache reallocates when full,
        ArraysCache uses mx.concatenate), so holding references to the old arrays
        is safe — they won't be mutated.
        """
        # Force evaluation so arrays are materialized in memory before we ref them
        mx.eval([c.state for c in cache])
        return [(type(c), c.state, c.meta_state) for c in cache]

    def _restore_cache(self, snapshot: list) -> list:
        """Create fresh cache objects restored from snapshot via from_state."""
        return [cls.from_state(state, meta) for cls, state, meta in snapshot]

    def __call__(
        self,
        messages: List[Dict[str, str]],
        images: List[str] = None,
        audios: List[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text response from images and messages.

        Args:
            images (List[str]): List of image paths to process.
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional model parameters (temperature, max_tokens, etc.)

        Returns:
            Union[str, Generator[str, None, None]]:
                - If stream=False: Complete response as string
                - If stream=True: Generator yielding response chunks
        """
        if not images:
            images = None
        if not audios:
            audios = None

        enable_thinking = kwargs.pop("enable_thinking", False)

        # Build full formatted prompt
        full_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        model_params = {
            "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            **kwargs
        }
        # 8-bit KV cache quantization: halves KV memory, reduces attention bandwidth ~15-25%
        model_params.setdefault("kv_bits", 8)

        # --- System prompt KV cache ---
        # For text-only requests with a system message, reuse pre-filled KV states
        # so we only prefill the new user tokens each request instead of the full context.
        use_cache = (
            not images and not audios
            and messages and messages[0].get("role") == "system"
        )

        if use_cache:
            sys_content = messages[0]["content"]
            sys_hash = hashlib.md5(sys_content.encode()).hexdigest()

            if sys_hash != self._sys_hash:
                # Cache miss — prefill the system prompt once and snapshot the KV states.
                # We cannot call apply_chat_template with only a system message (the
                # Qwen3.5 template requires at least one user turn). Instead, construct
                # the system prefix text directly in Qwen/ChatML format and tokenize it,
                # then pass input_ids directly to stream_generate (bypasses prepare_inputs).
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
                sys_prefix_text = f"<|im_start|>system\n{sys_content}<|im_end|>\n"
                sys_ids = tokenizer.encode(sys_prefix_text, add_special_tokens=False)

                from mlx_vlm.models.cache import make_prompt_cache
                sys_cache = make_prompt_cache(self.model.language_model)

                # Use generate_step directly for prefill-only (max_tokens=0).
                # stream_generate crashes with max_tokens=0 because its post-loop
                # "finalize" yield references `token` which is never assigned when
                # the inner loop runs 0 iterations (a bug in the library).
                # generate_step itself handles max_tokens=0 correctly: it runs the
                # full prefill pass then breaks immediately without yielding.
                for _ in generate_step(
                    mx.array([sys_ids]),
                    self.model,
                    None,   # pixel_values
                    None,   # mask
                    max_tokens=0,
                    prompt_cache=sys_cache,
                ):
                    pass  # never executes; prefill happened in generator setup

                self._sys_snapshot = self._snapshot_cache(sys_cache)
                self._sys_n_tokens = len(sys_ids)
                self._sys_hash = sys_hash
                print(f"[model] System prompt KV cached ({self._sys_n_tokens} tokens)")

            # Restore snapshot and get only the NEW tokens to prefill
            prompt_cache = self._restore_cache(self._sys_snapshot)
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            full_ids = tokenizer.encode(full_prompt)
            new_ids = full_ids[self._sys_n_tokens:]
            model_params["input_ids"] = mx.array([new_ids])
            model_params["pixel_values"] = None
            model_params["mask"] = None
            model_params["prompt_cache"] = prompt_cache

            if not stream:
                text = ""
                for r in stream_generate(
                    self.model, self.processor, "",
                    image=None, audio=None,
                    **model_params
                ):
                    text += r.text
                return text
            else:
                return stream_generate(
                    self.model, self.processor, "",
                    image=None, audio=None,
                    **model_params
                )

        # Fallback: full prefill (no system message, or multimodal request)
        if not stream:
            result = generate(
                self.model, self.processor, full_prompt,
                image=images, audio=audios,
                **model_params
            )
            return result.text
        else:
            return stream_generate(
                self.model, self.processor, full_prompt,
                image=images, audio=audios,
                **model_params
            )

    def get_embeddings(
        self,
        prompts: List[str],
        images: Optional[List[str]] = None,
        batch_size: int = 1,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Get embeddings for a list of prompts and optional images, supporting batch processing.
        Args:
            prompts: List of text prompts
            images: Optional list of image paths (must be same length as prompts if provided)
            batch_size: Size of batches for processing
            normalize: Whether to apply L2 normalization to embeddings
        Returns:
            List of embeddings as float arrays
        """
        if images is None:
            images = []

        try:
            # Text-only batch
            if not images:
                # Batch tokenize and pad
                tokenized = [self.processor.tokenizer.encode(self._format_prompt(p, 0), add_special_tokens=True) for p in prompts]
                max_len = max(len(t) for t in tokenized)
                pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
                batch_input_ids = [t + [pad_id] * (max_len - len(t)) for t in tokenized]
                batch_input_ids = mx.array(batch_input_ids)

                # Run in batches
                all_embeddings = []
                try:
                    for i in range(0, len(prompts), batch_size):
                        batch_ids = batch_input_ids[i:i+batch_size]
                        embeddings = self.model.language_model.model(batch_ids)
                        pooled = self._apply_pooling_strategy(embeddings)
                        if normalize:
                            pooled = self._apply_l2_normalization(pooled)
                        all_embeddings.extend(pooled.tolist())

                        # Clean up intermediate arrays
                        del embeddings, pooled
                        mx.clear_cache()

                finally:
                    # Clean up batch arrays
                    del batch_input_ids
                    mx.clear_cache()
                    gc.collect()

                return all_embeddings

            # Image+prompt batch
            if len(images) != len(prompts):
                raise ValueError("If images are provided, must be same length as prompts (one image per prompt)")

            all_embeddings = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_images = images[i:i+batch_size]
                formatted_prompts = [self._format_prompt(p, 1) for p in batch_prompts]

                try:
                    inputs = prepare_inputs(
                        self.processor,
                        batch_images,
                        formatted_prompts,
                        getattr(self.model.config, "image_token_index", None)
                    )
                    input_ids = inputs["input_ids"]
                    pixel_values = inputs.get("pixel_values", None)
                    image_grid_thw = inputs.get("image_grid_thw", None)
                    inputs_embeds = self.model.get_input_embeddings(input_ids, pixel_values, image_grid_thw)
                    embeddings = self.model.language_model.model(None, inputs_embeds=inputs_embeds)
                    pooled = self._apply_pooling_strategy(embeddings)
                    if normalize:
                        pooled = self._apply_l2_normalization(pooled)
                    all_embeddings.extend(pooled.tolist())
                finally:
                    # Clean up batch variables
                    if 'inputs' in locals():
                        del inputs
                    if 'input_ids' in locals():
                        del input_ids
                    if 'pixel_values' in locals():
                        del pixel_values
                    if 'image_grid_thw' in locals():
                        del image_grid_thw
                    if 'inputs_embeds' in locals():
                        del inputs_embeds
                    if 'embeddings' in locals():
                        del embeddings
                    if 'pooled' in locals():
                        del pooled
                    mx.clear_cache()
                    gc.collect()

            return all_embeddings
        except Exception as e:
            # Clean up on error
            mx.clear_cache()
            gc.collect()
            raise

    def _format_prompt(self, prompt: str, n_images: int) -> str:
        """Format a single prompt using the chat template."""
        return apply_chat_template(
            self.processor,
            self.config,
            prompt,
            add_generation_prompt=True,
            num_images=n_images
        )

    def _prepare_single_input(self, formatted_prompt: str, images: List[str]) -> Dict:
        """Prepare inputs for a single prompt-image pair."""
        return prepare_inputs(
            self.processor,
            images,
            formatted_prompt,
            getattr(self.model.config, "image_token_index", None)
        )

    def _get_single_embedding(
        self,
        inputs: Dict,
        normalize: bool = True
    ) -> List[float]:
        """Get embedding for a single processed input."""
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values", None)

        # Extract additional kwargs
        data_kwargs = {
            k: v for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        image_grid_thw = data_kwargs.pop("image_grid_thw", None)

        inputs_embeds = self.model.get_input_embeddings(input_ids, pixel_values, image_grid_thw)
        embeddings = self.model.language_model.model(None, inputs_embeds=inputs_embeds)

        # Apply pooling
        pooled_embedding = self._apply_pooling_strategy(embeddings)

        # Apply normalization if requested
        if normalize:
            pooled_embedding = self._apply_l2_normalization(pooled_embedding)

        return pooled_embedding.tolist()

    def _apply_pooling_strategy(self, embeddings: mx.array) -> mx.array:
        """Apply mean pooling to embeddings."""
        return mx.mean(embeddings, axis=1)

    def _apply_l2_normalization(self, embeddings: mx.array) -> mx.array:
        """Apply L2 normalization to embeddings."""
        l2_norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (l2_norms + 1e-8)
        return embeddings
