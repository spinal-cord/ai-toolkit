/**
 * GPU API response
 */

export interface GpuUtilization {
  gpu: number;
  memory: number;
}

export interface GpuMemory {
  total: number;
  free: number;
  used: number;
}

export interface GpuPower {
  draw: number;
  limit: number;
}

export interface GpuClocks {
  graphics: number;
  memory: number;
}

export interface GpuFan {
  speed: number;
}

export interface GpuInfo {
  index: number;
  name: string;
  driverVersion: string;
  temperature: number;
  utilization: GpuUtilization;
  memory: GpuMemory;
  power: GpuPower;
  clocks: GpuClocks;
  fan: GpuFan;
}

export interface CpuInfo {
  name: string;
  cores: number;
  temperature: number;
  totalMemory: number;
  freeMemory: number;
  availableMemory: number;
  currentLoad: number;
}

export interface GPUApiResponse {
  hasNvidiaSmi: boolean;
  isMac: boolean;
  gpus: GpuInfo[];
  error?: string;
}

/**
 * Training configuration
 */

export interface Wan22TensorTypeConfig {
  rank?: number | null;
  alpha?: number | null;
  full?: boolean;
}

export interface Wan22TensorTypesConfig {
  self_attn?: Wan22TensorTypeConfig;
  cross_attn?: Wan22TensorTypeConfig;
  ffn?: Wan22TensorTypeConfig;
  text_embedding?: Wan22TensorTypeConfig;
  time_embedding?: Wan22TensorTypeConfig;
  head?: Wan22TensorTypeConfig;
}

export interface RankGateConfig {
  enabled?: boolean;
  target_rank_ratio?: number;
  start_step?: number | null;
  end_step?: number | null;
  temperature?: number;
  gamma?: number;
  alpha?: number;
  lambda_mid_max?: number;
  update_every?: number;
  fisher_decay?: number;
  use_first_order?: boolean;
  hardening_window?: number;
  eta_pen?: number;
  final_hardening?: boolean;
}

export interface LayerOverrideConfig {
  tensor_type: string;
  rank: number;
  layer_range: string;
}

export interface NetworkConfig {
  type: string;
  linear: number;
  linear_alpha: number;
  conv: number;
  conv_alpha: number;
  lokr_full_rank: boolean;
  lokr_factor: number;
  network_kwargs: {
    ignore_if_contains: string[];
  };
  transformer_only?: boolean;
  lora_a_init?: string | { method: string; std?: number };
  lora_b_init?: string | { method: string; std?: number };
  high_noise_lora_a_init?: string | { method: string; std?: number };
  high_noise_lora_b_init?: string | { method: string; std?: number };
  low_noise_lora_a_init?: string | { method: string; std?: number };
  low_noise_lora_b_init?: string | { method: string; std?: number };
  // Wan 2.2 tensor type specific configuration
  wan22_tensor_types?: Wan22TensorTypesConfig;
  wan22_enabled_types?: string[];
  // Per-layer rank overrides (global - applies to all experts)
  layer_overrides?: LayerOverrideConfig[];
  // Per-expert per-layer rank overrides (takes precedence over global layer_overrides)
  layer_overrides_high?: LayerOverrideConfig[]; // transformer_1 / high-noise expert
  layer_overrides_low?: LayerOverrideConfig[];  // transformer_2 / low-noise expert
  // Rank gate annealing (SparseForge-inspired)
  rank_gates?: RankGateConfig;
}

export interface SaveConfig {
  dtype: string;
  save_every: number;
  max_step_saves_to_keep: number;
  save_format: string;
  push_to_hub: boolean;
}

export interface DatasetConfig {
  folder_path: string;
  mask_path: string | null;
  mask_min_value: number;
  default_caption: string;
  caption_ext: string;
  caption_dropout_rate: number;
  caption_dropout_rate_t2v?: number;
  shuffle_tokens?: boolean;
  is_reg: boolean;
  network_weight: number;
  cache_latents_to_disk?: boolean;
  cache_latents?: boolean;
  resolution: number[];
  resize_method?: 'bicubic' | 'lanczos';
  controls: string[];
  control_path?: string | null;
  num_frames: number;
  shrink_video_to_frames: boolean;
  do_i2v?: boolean;
  do_t2v?: boolean;
  do_audio?: boolean;
  audio_normalize?: boolean;
  audio_preserve_pitch?: boolean;
  fps?: number;
  flip_x: boolean;
  flip_y: boolean;
  num_repeats?: number;
  control_path_1?: string | null;
  control_path_2?: string | null;
  control_path_3?: string | null;
  auto_frame_count?: boolean;
  // Optical flow caching for spectral_flow loss
  cache_optical_flow_to_disk?: boolean;
  optical_flow_model?: string;
}

export interface EMAConfig {
  use_ema: boolean;
  ema_decay: number;
}

export interface TimestepRangeOverride {
  start_timestep: number;  // absolute start timestep (0-1000)
  end_timestep: number;    // absolute end timestep (0-1000)
  // Loss weight overrides
  flow_weight?: number | null;
  spectral_weight?: number | null;
  spectral_low_weight?: number | null;
  spectral_mid_weight?: number | null;
  spectral_high_weight?: number | null;
  mse_weight?: number | null;
  // Spectral filter overrides
  spectral_low_cutoff?: number | null;
  spectral_high_cutoff?: number | null;
  spectral_lcr_weight?: number | null;
  spectral_temporal_scale?: number | null;
}

export interface TrainConfig {
  batch_size: number;
  bypass_guidance_embedding?: boolean;
  steps: number;
  gradient_accumulation: number;
  train_unet: boolean;
  train_text_encoder: boolean;
  gradient_checkpointing: boolean;
  noise_scheduler: string;
  timestep_type: string;
  content_or_style: string;
  optimizer: string;
  lr_scheduler?: string;
  lr_scheduler_params?: {
    total_iters?: number;
    power?: number;
    lr_end?: number;
    T_0?: number;
    T_mult?: number;
    eta_min?: number;
    step_size?: number;
    gamma?: number;
    factor?: number;
    end_factor?: number;
    start_factor?: number;
    num_warmup_steps?: number;
  };
  // Per-expert LR schedulers for Wan 2.2 14B dual-expert models
  expert_1_lr_scheduler?: string | undefined;
  expert_1_lr_scheduler_params?: {
    total_iters?: number;
    power?: number;
    lr_end?: number;
    T_0?: number;
    T_mult?: number;
    eta_min?: number;
    step_size?: number;
    gamma?: number;
    factor?: number;
    end_factor?: number;
    start_factor?: number;
    num_warmup_steps?: number;
  } | undefined;
  expert_2_lr_scheduler?: string | undefined;
  expert_2_lr_scheduler_params?: {
    total_iters?: number;
    power?: number;
    lr_end?: number;
    T_0?: number;
    T_mult?: number;
    eta_min?: number;
    step_size?: number;
    gamma?: number;
    factor?: number;
    end_factor?: number;
    start_factor?: number;
    num_warmup_steps?: number;
  } | undefined;
  lr: number;
  ema_config?: EMAConfig;
  dtype: string;
  unload_text_encoder: boolean;
  cache_text_embeddings: boolean;
  optimizer_params: {
    weight_decay: number;
  };
  skip_first_sample: boolean;
  force_first_sample: boolean;
  disable_sampling: boolean;
  diff_output_preservation: boolean;
  diff_output_preservation_multiplier: number;
  diff_output_preservation_class: string;
  blank_prompt_preservation?: boolean;
  blank_prompt_preservation_multiplier?: number;
  switch_boundary_every: number;
  loss_type: 'mse' | 'mae' | 'wavelet' | 'spectral' | 'spectral_flow' | 'mse_spectral_flow' | 'stepped' | 'mean_flow' | 'pseudo_huber';
  pseudo_huber_c?: number;
  // Spectral loss config - global weights (fallback for single-expert models)
  spectral_low_weight?: number;
  spectral_mid_weight?: number;
  spectral_high_weight?: number;
  spectral_low_cutoff?: number;
  spectral_high_cutoff?: number;
  spectral_use_phase?: boolean;
  spectral_lcr_weight?: number;
  spectral_transform?: string;
  spectral_temporal_scale?: number;
  // Spectral loss config - per-expert weights (MoE models like Wan 2.2 14B)
  spectral_low_weight_high?: number | null;
  spectral_mid_weight_high?: number | null;
  spectral_high_weight_high?: number | null;
  spectral_low_weight_low?: number | null;
  spectral_mid_weight_low?: number | null;
  spectral_high_weight_low?: number | null;
  // Per-expert frequency cutoffs (optional override)
  spectral_low_cutoff_high?: number | null;
  spectral_high_cutoff_high?: number | null;
  spectral_low_cutoff_low?: number | null;
  spectral_high_cutoff_low?: number | null;
  // Per-expert temporal scale (optional override)
  spectral_temporal_scale_high?: number | null;
  spectral_temporal_scale_low?: number | null;
  // Spectral loss weight for combined losses (spectral_flow, mse_spectral_flow)
  spectral_weight?: number;
  spectral_weight_high?: number | null;
  spectral_weight_low?: number | null;
  // Spectral flow loss config
  spectral_flow_weight?: number;
  spectral_flow_weight_low?: number | null;
  spectral_flow_weight_high?: number | null;
  spectral_flow_max_timestep?: number;
  spectral_flow_reverse_gate?: boolean;
  spectral_flow_motion_weighted?: boolean;
  spectral_flow_adaptive?: boolean;
  spectral_flow_rejection_threshold?: number;
  spectral_flow_max_rejections?: number;
  // MSE + Spectral + Flow loss config
  mse_spectral_flow_mse_weight?: number;
  mse_spectral_flow_mse_weight_low?: number | null;
  mse_spectral_flow_mse_weight_high?: number | null;
  mse_spectral_flow_gradient_projection_enabled?: boolean;
  do_differential_guidance?: boolean;
  differential_guidance_scale?: number;
  audio_loss_multiplier?: number;
  max_loss?: number | null;
  // Per-timestep range loss weight overrides
  // Ranges are in absolute model timesteps (0-1000)
  // Each expert dynamically checks if its current timestep is inside a range
  timestep_range_overrides?: TimestepRangeOverride[];
  // Attention tanh softcapping (Gemma2/Grok-1 style)
  // Prevents attention scores from becoming too extreme, improving training stability
  // Hierarchy: per-type-per-expert → per-type → per-expert → global
  attention_tanh_softcap_enabled?: boolean;
  attention_tanh_softcap_value?: number;  // Global default
  // Per-attention-type overrides (applies to both experts)
  attention_tanh_softcap_value_self_attn?: number | null;
  attention_tanh_softcap_value_cross_attn?: number | null;
  // Per-expert overrides (applies to both attention types)
  attention_tanh_softcap_value_high_noise?: number | null;
  attention_tanh_softcap_value_low_noise?: number | null;
  // Per-type-per-expert overrides (most specific)
  attention_tanh_softcap_value_self_attn_high_noise?: number | null;
  attention_tanh_softcap_value_self_attn_low_noise?: number | null;
  attention_tanh_softcap_value_cross_attn_high_noise?: number | null;
  attention_tanh_softcap_value_cross_attn_low_noise?: number | null;
  // Attention F32 RoPE acceleration
  // Use float32 instead of float64 for rotary embeddings (faster, still stable)
  attention_f32_rope_enabled?: boolean;
  // GELU acceleration for Wan 2.x FeedForward layers
  // Uses tanh.approx.f32 PTX instruction (~2-5% FF speedup)
  gelu_acceleration_enabled?: boolean;
}

export interface QuantizeKwargsConfig {
  exclude: string[];
}

export interface ModelConfig {
  name_or_path: string;
  quantize: boolean;
  quantize_te: boolean;
  qtype: string;
  qtype_te: string;
  quantize_kwargs?: QuantizeKwargsConfig;
  arch: string;
  low_vram: boolean;
  model_kwargs: { [key: string]: any };
  layer_offloading?: boolean;
  layer_offloading_transformer_percent?: number;
  layer_offloading_text_encoder_percent?: number;
  assistant_lora_path?: string;
  unconditional_lora_path?: string;
  te_name_or_path?: string; // Custom text encoder path (local or HF repo)
  compile?: boolean;
  block_compile?: boolean;
  compile_mode?: 'default' | 'max-autotune' | 'fastest';
  compile_fullgraph?: boolean;
  compile_dynamic?: boolean;
  cache_size_limit?: number;
  // Wan transformer eps override (for LayerNorm and attention norms)
  // Official config uses 1e-6 (for fp32 training)
  // For bf16 training, use larger eps like 1e-4 or 1e-5
  // Leave empty to use model's default
  wan_transformer_eps?: number | null;
}

export interface SampleItem {
  prompt: string;
  width?: number;
  height?: number;
  neg?: string;
  seed?: number;
  guidance_scale?: number;
  sample_steps?: number;
  fps?: number;
  num_frames?: number;
  ctrl_img?: string | null;
  ctrl_idx?: number;
  network_multiplier?: number;
  ctrl_img_1?: string | null;
  ctrl_img_2?: string | null;
  ctrl_img_3?: string | null;
  // NAG (Negative Attention Guidance) per-sample override
  nag_scale?: number;
  nag_alpha?: number;
  nag_tau?: number;
}

export interface SampleConfig {
  sampler: string;
  sample_every: number;
  width: number;
  height: number;
  prompts?: string[];
  samples: SampleItem[];
  neg: string;
  seed: number;
  walk_seed: boolean;
  guidance_scale: number;
  sample_steps: number;
  num_frames: number;
  fps: number;
  // NAG (Negative Attention Guidance) parameters - global defaults for all samples
  // nag_scale: 1.0 disables, >1 enables (typical range 1.0–20.0)
  // nag_alpha: blend factor between NAG-guided and original prediction (0.0–2.0)
  // nag_tau:   threshold for similarity-based scaling (typical 1.0–5.0)
  nag_scale?: number;
  nag_alpha?: number;
  nag_tau?: number;
}

export interface LoggingConfig {
  log_every: number;
  use_ui_logger: boolean;
}

export interface SliderConfig {
  guidance_strength?: number;
  anchor_strength?: number;
  positive_prompt?: string;
  negative_prompt?: string;
  target_class?: string;
  anchor_class?: string | null;
}

export interface ProcessConfig {
  type: string;
  sqlite_db_path?: string;
  training_folder: string;
  performance_log_every: number;
  trigger_word: string | null;
  device: string;
  network?: NetworkConfig;
  slider?: SliderConfig;
  save: SaveConfig;
  datasets: DatasetConfig[];
  train: TrainConfig;
  logging: LoggingConfig;
  model: ModelConfig;
  sample: SampleConfig;
}

export interface ConfigObject {
  name: string;
  process: ProcessConfig[];
}

export interface MetaConfig {
  name: string;
  version: string;
}

export interface JobConfig {
  job: string;
  config: ConfigObject;
  meta: MetaConfig;
}

export interface CaptionProcessConfig {
  type: string;
  sqlite_db_path?: string;
  device: string;
  caption: {
    model_name_or_path: string;
    model_name_or_path2?: string;
    dtype: string;
    quantize: boolean;
    qtype: string;
    low_vram: boolean;
    extensions: string[];
    path_to_caption: string;
    recaption: boolean;
    compile?: boolean;
    caption_prompt?: string;
    max_res?: number;
    max_new_tokens?: number;
    fixed_caption?: string;
    caption_extension?: string;
  }
}

export interface CaptionConfigObject {
  name: string;
  process: CaptionProcessConfig[];
}

export interface CaptionJobConfig {
  job: string;
  config: CaptionConfigObject;
}

export interface ConfigDoc {
  title: string | React.ReactNode;
  description: React.ReactNode;
}

export interface SelectOption {
  readonly value: string;
  readonly label: string;
}
export interface GroupedSelectOption {
  readonly label: string;
  readonly options: SelectOption[];
}

export type JobStatus = 'queued' | 'running' | 'stopping' | 'stopped' | 'completed' | 'error';
