'use client';
import { useMemo, useState } from 'react';
import {
  modelArchs,
  ModelArch,
  groupedModelOptions,
  quantizationOptions,
  defaultQtype,
  jobTypeOptions,
  SampleTags,
} from './options';
import { defaultCompileOptions, defaultDatasetConfig } from './jobConfig';
import { GroupedSelectOption, JobConfig, SelectOption } from '@/types';
import { objectCopy, tagsToObj, objToTags } from '@/utils/basic';
import {
  TextInput,
  TextAreaInput,
  SelectInput,
  Checkbox,
  FormGroup,
  NumberInput,
  SliderInput,
  CreatableSelectInput,
} from '@/components/formInputs';
import Card from '@/components/Card';
import { X, Copy, Wand2, SquareDashed } from 'lucide-react';
import { openUpsamplePromptsModal, toAspectRatio } from '@/components/UpsamplePromptsModal';
import { openPromptBoxEditor } from '@/components/PromptBoxEditorModal';
import AddSingleImageModal, { openAddImageModal } from '@/components/AddSingleImageModal';
import SampleControlImage from '@/components/SampleControlImage';
import { FlipHorizontal2, FlipVertical2 } from 'lucide-react';
import { handleModelArchChange } from './utils';
import { IoFlaskSharp } from 'react-icons/io5';
import { isMac } from '@/helpers/basic';

type Props = {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  datasetOptions: any;
  isLoading?: boolean;
};

const isDev = process.env.NODE_ENV === 'development';

// ============================================================================
// Rank Gate Annealing Configuration Section
// ============================================================================
function RankGateConfigSection({ jobConfig, setJobConfig }: { jobConfig: JobConfig; setJobConfig: (value: any, key: string) => void }) {
  const [collapsed, setCollapsed] = useState(true);
  const rg = jobConfig.config.process[0].network?.rank_gates || {};
  const enabled = rg.enabled !== false; // defaults to true

  const setRg = (key: string, value: any) => {
    setJobConfig({ ...rg, [key]: value }, 'config.process[0].network.rank_gates');
  };

  return (
    <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid var(--border)' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          padding: '4px 0',
        }}
        onClick={() => setCollapsed(!collapsed)}
      >
        <div>
          <div style={{ fontWeight: 600, color: 'var(--accent)' }}>Rank Gate Annealing (SparseForge-inspired)</div>
          <div style={{ fontSize: '11px', color: 'var(--muted)' }}>
            Soft, curvature-aware rank gating. Enabled by default. Prevents rank collapse during training.
          </div>
        </div>
        <span style={{ fontSize: '16px', color: 'var(--muted)' }}>{collapsed ? '▸' : '▾'}</span>
      </div>

      {!collapsed && (
        <div style={{ marginTop: '12px' }}>
          {/* Enable/Disable */}
          <Checkbox
            label="Enable Rank Gates"
            docKey="config.process[0].network.rank_gates.enabled"
            checked={enabled}
            onChange={value => setRg('enabled', value)}
          />

          {enabled && (
            <>
              {/* Target Rank Ratio */}
              <div style={{ marginTop: '12px' }}>
                <div style={{ fontSize: '12px', fontWeight: 500, marginBottom: '4px', color: 'var(--accent)' }}>Pruning Targets</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <NumberInput
                    label="Target Rank Ratio"
                    docKey="config.process[0].network.rank_gates.target_rank_ratio"
                    value={rg.target_rank_ratio ?? 0.3}
                    onChange={value => setRg('target_rank_ratio', value)}
                    placeholder="0.3 = keep 30% of ranks (aggressive default)"
                    min={0.1}
                    max={1.0}
                    step={0.05}
                  />
                  <NumberInput
                    label="Update Every Steps"
                    docKey="config.process[0].network.rank_gates.update_every"
                    value={rg.update_every ?? 15}
                    onChange={value => setRg('update_every', value)}
                    placeholder="Update gates every N steps (default: 15)"
                    min={1}
                    max={1000}
                  />
                </div>
              </div>

              {/* Annealing Timing */}
              <div style={{ marginTop: '12px' }}>
                <div style={{ fontSize: '12px', fontWeight: 500, marginBottom: '4px', color: 'var(--accent)' }}>Annealing Timing (leave empty for auto)</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <NumberInput
                    label="Start Step"
                    docKey="config.process[0].network.rank_gates.start_step"
                    value={rg.start_step ?? ''}
                    onChange={value => setRg('start_step', value === '' ? null : value)}
                    placeholder="Auto: 5% of total"
                    min={0}
                  />
                  <NumberInput
                    label="End Step"
                    docKey="config.process[0].network.rank_gates.end_step"
                    value={rg.end_step ?? ''}
                    onChange={value => setRg('end_step', value === '' ? null : value)}
                    placeholder="Auto: 75% of total"
                    min={0}
                  />
                </div>
              </div>

              {/* Temperature & Decay */}
              <div style={{ marginTop: '12px' }}>
                <div style={{ fontSize: '12px', fontWeight: 500, marginBottom: '4px', color: 'var(--accent)' }}>Temperature & Decay</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
                  <NumberInput
                    label="Temperature"
                    docKey="config.process[0].network.rank_gates.temperature"
                    value={rg.temperature ?? 1.0}
                    onChange={value => setRg('temperature', value)}
                    placeholder="Initial sigmoid temperature"
                    min={0.1}
                    max={10}
                    step={0.1}
                  />
                  <NumberInput
                    label="Gamma (Temp Decay)"
                    docKey="config.process[0].network.rank_gates.gamma"
                    value={rg.gamma ?? 0.95}
                    onChange={value => setRg('gamma', value)}
                    placeholder="T ← γT per update (default: 0.95)"
                    min={0.9}
                    max={0.999}
                    step={0.001}
                  />
                  <NumberInput
                    label="Alpha (Gate EMA)"
                    docKey="config.process[0].network.rank_gates.alpha"
                    value={rg.alpha ?? 0.1}
                    onChange={value => setRg('alpha', value)}
                    placeholder="Gate update rate (default: 0.1)"
                    min={0.001}
                    max={0.5}
                    step={0.001}
                  />
                </div>
              </div>

              {/* Binary Preference */}
              <div style={{ marginTop: '12px' }}>
                <div style={{ fontSize: '12px', fontWeight: 500, marginBottom: '4px', color: 'var(--accent)' }}>Binary Preference Penalty (L_mid)</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <NumberInput
                    label="Lambda Mid Max"
                    docKey="config.process[0].network.rank_gates.lambda_mid_max"
                    value={rg.lambda_mid_max ?? 0.01}
                    onChange={value => setRg('lambda_mid_max', value)}
                    placeholder="Max penalty strength (default: 0.01)"
                    min={0}
                    max={0.1}
                    step={0.001}
                  />
                  <NumberInput
                    label="Eta Penalty"
                    docKey="config.process[0].network.rank_gates.eta_pen"
                    value={rg.eta_pen ?? 0.01}
                    onChange={value => setRg('eta_pen', value)}
                    placeholder="Mid-preference nudge"
                    min={0}
                    max={1}
                    step={0.001}
                  />
                </div>
              </div>

              {/* Fisher & Scoring */}
              <div style={{ marginTop: '12px' }}>
                <div style={{ fontSize: '12px', fontWeight: 500, marginBottom: '4px', color: 'var(--accent)' }}>Fisher & Scoring</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <NumberInput
                    label="Fisher Decay"
                    docKey="config.process[0].network.rank_gates.fisher_decay"
                    value={rg.fisher_decay ?? 0.999}
                    onChange={value => setRg('fisher_decay', value)}
                    placeholder="EMA decay for Fisher"
                    min={0.9}
                    max={0.9999}
                    step={0.0001}
                  />
                  <Checkbox
                    label="Use First-Order Term |g·w|"
                    docKey="config.process[0].network.rank_gates.use_first_order"
                    checked={rg.use_first_order !== false}
                    onChange={value => setRg('use_first_order', value)}
                  />
                </div>
              </div>

              {/* Hardening */}
              <div style={{ marginTop: '12px' }}>
                <div style={{ fontSize: '12px', fontWeight: 500, marginBottom: '4px', color: 'var(--accent)' }}>Final Hardening</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <NumberInput
                    label="Hardening Window"
                    docKey="config.process[0].network.rank_gates.hardening_window"
                    value={rg.hardening_window ?? 500}
                    onChange={value => setRg('hardening_window', value)}
                    placeholder="Steps for soft→hard transition"
                    min={0}
                    max={10000}
                  />
                  <Checkbox
                    label="Final Hardening (Binarize)"
                    docKey="config.process[0].network.rank_gates.final_hardening"
                    checked={rg.final_hardening !== false}
                    onChange={value => setRg('final_hardening', value)}
                  />
                </div>
              </div>

              {/* Quick Presets */}
              <div style={{ marginTop: '12px', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                <button
                  type="button"
                  onClick={() => {
                    setRg('target_rank_ratio', 0.8);
                    setRg('lambda_mid_max', 0.002);
                    setRg('alpha', 0.02);
                    setRg('gamma', 0.98);
                    setRg('update_every', 40);
                  }}
                  style={{
                    padding: '4px 8px',
                    fontSize: '10px',
                    cursor: 'pointer',
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text)',
                    border: '1px solid var(--border)',
                    borderRadius: '4px',
                  }}
                >
                  Conservative (80% keep)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setRg('target_rank_ratio', 0.5);
                    setRg('lambda_mid_max', 0.005);
                    setRg('alpha', 0.05);
                    setRg('gamma', 0.97);
                    setRg('update_every', 25);
                  }}
                  style={{
                    padding: '4px 8px',
                    fontSize: '10px',
                    cursor: 'pointer',
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text)',
                    border: '1px solid var(--border)',
                    borderRadius: '4px',
                  }}
                >
                  Moderate (50% keep)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setRg('target_rank_ratio', 0.3);
                    setRg('lambda_mid_max', 0.01);
                    setRg('alpha', 0.1);
                    setRg('gamma', 0.95);
                    setRg('update_every', 15);
                  }}
                  style={{
                    padding: '4px 8px',
                    fontSize: '10px',
                    cursor: 'pointer',
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text)',
                    border: '1px solid var(--border)',
                    borderRadius: '4px',
                  }}
                >
                  Aggressive Default (30% keep)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setRg('target_rank_ratio', 0.2);
                    setRg('lambda_mid_max', 0.015);
                    setRg('alpha', 0.15);
                    setRg('gamma', 0.94);
                    setRg('update_every', 12);
                  }}
                  style={{
                    padding: '4px 8px',
                    fontSize: '10px',
                    cursor: 'pointer',
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text)',
                    border: '1px solid var(--border)',
                    borderRadius: '4px',
                  }}
                >
                  Very Aggressive (20% keep)
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

const loraInitOptions: SelectOption[] = [
  { value: 'gaussian_random', label: 'Gaussian Random (std=1/√rank)' },
  { value: 'zeros', label: 'Zeros (Default B)' },
  { value: 'kaiming_uniform', label: 'Kaiming Uniform' },
  { value: 'kaiming_normal', label: 'Kaiming Normal' },
  { value: 'xavier_uniform', label: 'Xavier Uniform' },
  { value: 'xavier_normal', label: 'Xavier Normal' },
  { value: 'normal', label: 'Normal (std=0.01)' },
  { value: 'small_noise', label: 'Small Noise (std=0.001)' },
];

// Wan 2.2 Tensor Types Configuration
const WAN22_TENSOR_TYPES: Record<string, { name: string; maxRank: number; description: string }> = {
  self_attn: { name: 'Self Attention', maxRank: 5120, description: 'attn1 q/k/v/o projections' },
  'self_attn.q': { name: 'Self Attention Q', maxRank: 5120, description: 'attn1 Q projection' },
  'self_attn.k': { name: 'Self Attention K', maxRank: 5120, description: 'attn1 K projection' },
  'self_attn.v': { name: 'Self Attention V', maxRank: 5120, description: 'attn1 V projection' },
  'self_attn.o': { name: 'Self Attention O', maxRank: 5120, description: 'attn1 output projection' },
  cross_attn: { name: 'Cross Attention', maxRank: 5120, description: 'attn2 q/k/v/o projections' },
  'cross_attn.q': { name: 'Cross Attention Q', maxRank: 5120, description: 'attn2 Q projection' },
  'cross_attn.k': { name: 'Cross Attention K', maxRank: 5120, description: 'attn2 K projection' },
  'cross_attn.v': { name: 'Cross Attention V', maxRank: 5120, description: 'attn2 V projection' },
  'cross_attn.o': { name: 'Cross Attention O', maxRank: 5120, description: 'attn2 output projection' },
  ffn: { name: 'Feed Forward', maxRank: 5120, description: 'FFN projections' },
  'ffn.0': { name: 'FFN Up', maxRank: 5120, description: 'FFN up projection (ffn.up)' },
  'ffn.2': { name: 'FFN Down', maxRank: 5120, description: 'FFN down projection (ffn.down)' },
  text_embedding: { name: 'Text Embedding', maxRank: 4096, description: 'text embedding linear layers' },
  'text_embedding.1': { name: 'Text Embedding L1', maxRank: 4096, description: 'text embedding linear_1' },
  'text_embedding.2': { name: 'Text Embedding L2', maxRank: 4096, description: 'text embedding linear_2' },
  time_embedding: { name: 'Time Embedding', maxRank: 256, description: 'time embedding linear layers' },
  'time_embedding.1': { name: 'Time Embedding L1', maxRank: 256, description: 'time embedding linear_1' },
  'time_embedding.2': { name: 'Time Embedding L2', maxRank: 256, description: 'time embedding linear_2' },
  head: { name: 'Output Head', maxRank: 64, description: 'output projection' },
  patch_embedding: { name: 'Patch Embedding', maxRank: -1, description: 'patch embedding (full weight)' },
  modulation: { name: 'Modulation', maxRank: -1, description: 'modulation parameters (full weight)' },
  norm: { name: 'Norm', maxRank: -1, description: 'normalization parameters (full weight)' },
  time_projection: { name: 'Time Projection', maxRank: -1, description: 'time projection (full weight)' },
};

// Aliases mapping - these are resolved to canonical tensor types
// Matches backend WAN22_TENSOR_TYPE_ALIASES
const WAN22_TENSOR_TYPE_ALIASES: Record<string, string> = {
  // FFN aliases
  'ffn.up': 'ffn.0',
  'ffn.down': 'ffn.2',
  'ffn.net.0.proj': 'ffn.0',
  'ffn.net.2': 'ffn.2',
  // Text embedding aliases (full layer paths)
  'condition_embedder.text_embedder.linear_1': 'text_embedding.1',
  'condition_embedder.text_embedder.linear_2': 'text_embedding.2',
  'text_embedder.linear_1': 'text_embedding.1',
  'text_embedder.linear_2': 'text_embedding.2',
  // Time embedding aliases (full layer paths)
  'condition_embedder.time_embedder.linear_1': 'time_embedding.1',
  'condition_embedder.time_embedder.linear_2': 'time_embedding.2',
  'time_embedder.linear_1': 'time_embedding.1',
  'time_embedder.linear_2': 'time_embedding.2',
  // Head aliases
  'proj_out': 'head',
  // Patch embedding aliases
  'patch_embed': 'patch_embedding',
  // Modulation aliases
  'scale_shift_table': 'modulation',
  'mod': 'modulation',
  // Time projection aliases
  'condition_embedder.time_proj': 'time_projection',
  'time_proj': 'time_projection',
  // Self attention sub-type aliases
  'attn1.to_q': 'self_attn.q',
  'attn1.to_k': 'self_attn.k',
  'attn1.to_v': 'self_attn.v',
  'attn1.to_out.0': 'self_attn.o',
  // Cross attention sub-type aliases
  'attn2.to_q': 'cross_attn.q',
  'attn2.to_k': 'cross_attn.k',
  'attn2.to_v': 'cross_attn.v',
  'attn2.to_out.0': 'cross_attn.o',
};

// Resolve a tensor type to its canonical name
const resolveTensorType = (typeKey: string): string => {
  if (WAN22_TENSOR_TYPES[typeKey]) return typeKey;
  if (WAN22_TENSOR_TYPE_ALIASES[typeKey]) return WAN22_TENSOR_TYPE_ALIASES[typeKey];
  return typeKey; // Return as-is (will be validated later)
};

// Helper to check if a tensor type is a single tensor (not per-block)
const isSingleTensorType = (typeKey: string): boolean => {
  const canonical = resolveTensorType(typeKey);
  const singleTensorTypes = [
    'patch_embedding', 'modulation', 'norm', 'time_projection', 'head',
    'text_embedding', 'text_embedding.1', 'text_embedding.2',
    'time_embedding', 'time_embedding.1', 'time_embedding.2'
  ];
  return singleTensorTypes.includes(canonical);
};

// Helper to get max layers for a tensor type
const getMaxLayers = (typeKey: string): number => {
  const canonical = resolveTensorType(typeKey);
  const typeInfo = WAN22_TENSOR_TYPES[canonical];
  if (!typeInfo || typeInfo.maxRank === -1) return 0;
  const name = typeInfo.name || '';
  if (name.includes('Attn') || name.includes('FFN')) return 40; // Wan 2.2 has 40 transformer blocks (0-39)
  return 0;
};

const schedulerOptions: SelectOption[] = [
  { value: 'euler', label: 'Euler' },
  { value: 'ddim', label: 'DDIM' },
  { value: 'unipc', label: 'UniPC' },
  { value: 'flowmatch', label: 'FlowMatch' },
  { value: 'custom_flowmatch', label: 'Custom FlowMatch' },
  { value: 'ddpm', label: 'DDPM' },
  { value: 'pndm', label: 'PNDM' },
  { value: 'dpm', label: 'DPM' },
];

// Optimizers that have their own adaptive learning rate and don't need external LR schedulers
const NO_SCHEDULER_OPTIMIZERS = ['prodigyopt', 'prodigy8bit', 'automagic', 'automagic2', 'automagic3', 'dadaptation', 'dadaptationlion'];

// Default parameters for each scheduler type
const SCHEDULER_DEFAULT_PARAMS: Record<string, Record<string, any>> = {
  none: {},
  cosine: {},
  cosine_with_restarts: { T_mult: 2, eta_min: 0 },
  step: { step_size: 100, gamma: 0.1 },
  polynomial: { power: 0.8, lr_end: 0 },
  constant: { factor: 1.0, step_size: 100, end_factor: 1.0 },
  linear: { start_factor: 1.0, end_factor: 0.0 },
  constant_with_warmup: { num_warmup_steps: 1000 },
};

export default function SimpleJob({
  jobConfig,
  setJobConfig,
  handleSubmit,
  status,
  runId,
  gpuIDs,
  setGpuIDs,
  gpuList,
  datasetOptions,
  isLoading,
}: Props) {
  const modelArch = useMemo(() => {
    return modelArchs.find(a => a.name === jobConfig.config.process[0].model.arch) as ModelArch;
  }, [jobConfig.config.process[0].model.arch]);

  const jobType = useMemo(() => {
    return jobTypeOptions.find(j => j.value === jobConfig.config.process[0].type);
  }, [jobConfig.config.process[0].type]);

  const disableSections = useMemo(() => {
    let sections: string[] = [];
    if (modelArch?.disableSections) {
      sections = sections.concat(modelArch.disableSections);
    }
    if (jobType?.disableSections) {
      sections = sections.concat(jobType.disableSections);
    }
    return sections;
  }, [modelArch, jobType]);

  const isVideoModel = !!(modelArch?.group === 'video');
  const isAudioModel = !!(modelArch?.group === 'audio');

  const taggedSampleArr: Record<string, any>[] | null = useMemo(() => {
    if (!modelArch) return null;
    if (!modelArch.sampleTags) return null;
    if (!jobConfig.config.process[0].sample.samples) return null;
    let sampleArr: any[] = [];
    for (let i = 0; i < jobConfig.config.process[0].sample.samples.length; i++) {
      const taggedPrompt = jobConfig.config.process[0].sample.samples[i].prompt;
      const tagsObj = tagsToObj(taggedPrompt);
      sampleArr.push(tagsObj);
    }
    return sampleArr;
  }, [modelArch, jobConfig.config.process[0].sample.samples]);

  const modelArchTagSections: SampleTags[] | null = useMemo(() => {
    if (!modelArch?.sampleTags) return null;
    const maxPerGroup = 5;
    let sections: SampleTags[] = [];
    let subSection: SampleTags = {};
    for (const [tagKey, tag] of Object.entries(modelArch.sampleTags)) {
      if ((tag.full && Object.keys(subSection).length > 0) || Object.keys(subSection).length >= maxPerGroup) {
        // reset the sub section build if the next tag is full or max per group is reached
        sections.push(subSection);
        subSection = {};
      }
      subSection[tagKey] = tag;
      if (tag.full) {
        // if the tag is full, push the section immediately and reset the sub section build
        sections.push(subSection);
        subSection = {};
      }
    }
    if (Object.keys(subSection).length > 0) {
      sections.push(subSection);
    }
    return sections.length > 0 ? sections : null;
  }, [modelArch]);

  const numTopCards = useMemo(() => {
    let count = 4; // job settings, model config, target config, save config
    if (modelArch?.additionalSections?.includes('model.multistage')) {
      count += 1; // add multistage card
    }
    if (!disableSections.includes('model.quantize')) {
      count += 1; // add quantization card
    }
    if (!disableSections.includes('slider')) {
      count += 1; // add slider card
    }
    return count;
  }, [modelArch, disableSections]);

  let topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-4 gap-6';

  if (numTopCards == 5) {
    topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6';
  }
  if (numTopCards == 6) {
    topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 2xl:grid-cols-6 gap-6';
  }

  const numTrainingCols = useMemo(() => {
    let count = 4;
    if (!disableSections.includes('train.diff_output_preservation')) {
      count += 1;
    }
    return count;
  }, [disableSections]);

  let trainingBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';

  if (numTrainingCols == 5) {
    trainingBarClass = 'grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6';
  }

  const transformerQuantizationOptions: GroupedSelectOption[] | SelectOption[] = useMemo(() => {
    const hasARA = modelArch?.accuracyRecoveryAdapters && Object.keys(modelArch.accuracyRecoveryAdapters).length > 0;
    if (!hasARA) {
      return quantizationOptions;
    }
    let newQuantizationOptions = [
      {
        label: 'Standard',
        options: [quantizationOptions[0], quantizationOptions[1]],
      },
    ];

    // add ARAs if they exist for the model
    let ARAs: SelectOption[] = [];
    if (modelArch.accuracyRecoveryAdapters) {
      for (const [label, value] of Object.entries(modelArch.accuracyRecoveryAdapters)) {
        ARAs.push({ value, label });
      }
    }
    if (ARAs.length > 0) {
      newQuantizationOptions.push({
        label: 'Accuracy Recovery Adapters',
        options: ARAs,
      });
    }

    let additionalQuantizationOptions: SelectOption[] = [];
    // add the quantization options if they are not already included
    for (let i = 2; i < quantizationOptions.length; i++) {
      const option = quantizationOptions[i];
      additionalQuantizationOptions.push(option);
    }
    if (additionalQuantizationOptions.length > 0) {
      newQuantizationOptions.push({
        label: 'Additional Quantization Options',
        options: additionalQuantizationOptions,
      });
    }
    return newQuantizationOptions;
  }, [modelArch]);

  const showGPUSelect = !isMac();

  let numDatasetCols = 4;
  let numSampleTopCols = 4;
  let datasetStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';
  let sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';
  if (isVideoModel) {
    numSampleTopCols += 1;
  }
  if (isAudioModel) {
    numDatasetCols -= 1;
    numSampleTopCols -= 1;
  }
  if (numDatasetCols == 3) {
    datasetStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6';
  }
  if (numSampleTopCols == 5) {
    sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6';
  }
  if (numSampleTopCols == 3) {
    sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6';
  }
  return (
    <>
      <form
        onSubmit={handleSubmit}
        className={`space-y-8 relative ${isLoading ? 'pointer-events-none opacity-50' : ''}`}
      >
        {isLoading && (
          <div className="absolute inset-0 z-50 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-gray-400 border-t-blue-500" />
              <span className="text-sm text-gray-400">Loading...</span>
            </div>
          </div>
        )}
        <div className={topBarClass}>
          <Card title="Job">
            <TextInput
              label="Training Name"
              value={jobConfig.config.name}
              docKey="config.name"
              onChange={value => setJobConfig(value, 'config.name')}
              placeholder="Enter training name"
              disabled={runId !== null}
              required
            />
            {showGPUSelect && (
              <SelectInput
                label="GPU ID"
                value={`${gpuIDs}`}
                docKey="gpuids"
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            )}
            {disableSections.includes('trigger_word') ? null : (
              <TextInput
                label="Trigger Word"
                value={jobConfig.config.process[0].trigger_word || ''}
                docKey="config.process[0].trigger_word"
                onChange={(value: string | null) => {
                  if (value?.trim() === '') {
                    value = null;
                  }
                  setJobConfig(value, 'config.process[0].trigger_word');
                }}
                placeholder=""
                required
              />
            )}
          </Card>

          {/* Model Configuration Section */}
          <Card title="Model">
            <SelectInput
              label="Model Architecture"
              value={jobConfig.config.process[0].model.arch}
              onChange={value => {
                handleModelArchChange(jobConfig.config.process[0].model.arch, value, jobConfig, setJobConfig);
              }}
              options={groupedModelOptions}
            />
            <TextInput
              label="Name or Path"
              value={jobConfig.config.process[0].model.name_or_path}
              docKey="config.process[0].model.name_or_path"
              onChange={(value: string | null) => {
                if (value?.trim() === '') {
                  value = null;
                }
                setJobConfig(value, 'config.process[0].model.name_or_path');
              }}
              placeholder=""
              required
            />
            {modelArch?.additionalSections?.includes('model.assistant_lora_path') && (
              <TextInput
                label="Training Adapter Path"
                value={jobConfig.config.process[0].model.assistant_lora_path ?? ''}
                docKey="config.process[0].model.assistant_lora_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.assistant_lora_path');
                }}
                placeholder=""
              />
            )}
            {modelArch?.additionalSections?.includes('model.unconditional_lora_path') && (
              <TextInput
                label="Unconditional Adapter Path"
                value={jobConfig.config.process[0].model.unconditional_lora_path ?? ''}
                docKey="config.process[0].model.unconditional_lora_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.unconditional_lora_path');
                }}
                placeholder=""
              />
            )}
            {modelArch?.additionalSections?.includes('model.te_name_or_path') && (
              <TextInput
                label="Text Encoder Path"
                value={jobConfig.config.process[0].model.te_name_or_path ?? ''}
                docKey="config.process[0].model.te_name_or_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.te_name_or_path');
                }}
                placeholder="HF repo ID or local path"
              />
            )}
            {modelArch?.additionalSections?.includes('model.low_vram') && (
              <FormGroup label="Options">
                <Checkbox
                  label="Low VRAM"
                  checked={jobConfig.config.process[0].model.low_vram}
                  onChange={value => setJobConfig(value, 'config.process[0].model.low_vram')}
                />
              </FormGroup>
            )}
            {modelArch?.additionalSections?.includes('model.model_kwargs.kv_cache') && (
              <Checkbox
                label="KV Cache"
                docKey="model.model_kwargs.kv_cache"
                checked={jobConfig.config.process[0].model.model_kwargs.kv_cache || false}
                onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.kv_cache')}
              />
            )}
            {modelArch?.additionalSections?.includes('model.qie.match_target_res') && (
              <Checkbox
                label="Match Target Res"
                docKey="model.qie.match_target_res"
                checked={jobConfig.config.process[0].model.model_kwargs.match_target_res}
                onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.match_target_res')}
              />
            )}
            {modelArch?.additionalSections?.includes('model.layer_offloading') && !isMac() && (
              <>
                <Checkbox
                  label={
                    <>
                      Layer Offloading <IoFlaskSharp className="inline text-yellow-500" name="Experimental" />{' '}
                    </>
                  }
                  checked={jobConfig.config.process[0].model.layer_offloading || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.layer_offloading')}
                  docKey="model.layer_offloading"
                />
                {jobConfig.config.process[0].model.layer_offloading && (
                  <div className="pt-2">
                    <SliderInput
                      label="Transformer Offload %"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_transformer_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_transformer_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                    <SliderInput
                      label="Text Encoder Offload %"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_text_encoder_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_text_encoder_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                  </div>
                )}
              </>
            )}
          </Card>
          {disableSections.includes('model.quantize') ? null : (
            <Card title="Quantize / Compile">
              <SelectInput
                label="Transformer"
                value={jobConfig.config.process[0].model.quantize ? jobConfig.config.process[0].model.qtype : ''}
                onChange={value => {
                  if (value === '') {
                    setJobConfig(false, 'config.process[0].model.quantize');
                    value = defaultQtype;
                  } else {
                    setJobConfig(true, 'config.process[0].model.quantize');
                  }
                  setJobConfig(value, 'config.process[0].model.qtype');
                }}
                options={transformerQuantizationOptions}
              />
              {!disableSections.includes('model.quantize_te') && (
                <SelectInput
                  label="Text Encoder"
                  value={
                    jobConfig.config.process[0].model.quantize_te ? jobConfig.config.process[0].model.qtype_te : ''
                  }
                  onChange={value => {
                    if (value === '') {
                      setJobConfig(false, 'config.process[0].model.quantize_te');
                      value = defaultQtype;
                    } else {
                      setJobConfig(true, 'config.process[0].model.quantize_te');
                    }
                    setJobConfig(value, 'config.process[0].model.qtype_te');
                  }}
                  options={quantizationOptions}
                />
              )}
              <FormGroup label="Compile Options">
                <></>
              </FormGroup>
              <Checkbox
                label="Compile Model"
                checked={jobConfig.config.process[0].model.compile || false}
                onChange={value => {
                  setJobConfig(value, 'config.process[0].model.compile');
                  if (value) {
                    for (const key in defaultCompileOptions) {
                      setJobConfig((defaultCompileOptions as any)[key], `config.process[0].model.${key}`);
                    }
                  } else {
                    for (const key in defaultCompileOptions) {
                      setJobConfig(undefined, `config.process[0].model.${key}`);
                    }
                  }
                }}
              />
            </Card>
          )}
          {modelArch?.additionalSections?.includes('model.multistage') && (
            <Card title="Multistage">
              <FormGroup label="Stages to Train" docKey={'model.multistage'}>
                <Checkbox
                  label="High Noise"
                  checked={jobConfig.config.process[0].model.model_kwargs?.train_high_noise || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_high_noise')}
                />
                <Checkbox
                  label="Low Noise"
                  checked={jobConfig.config.process[0].model.model_kwargs?.train_low_noise || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_low_noise')}
                />
              </FormGroup>
              <NumberInput
                label="Switch Every"
                value={jobConfig.config.process[0].train.switch_boundary_every}
                onChange={value => setJobConfig(value, 'config.process[0].train.switch_boundary_every')}
                placeholder="eg. 1"
                docKey={'train.switch_boundary_every'}
                min={1}
                required
              />
            </Card>
          )}
          <Card title="Target">
            <SelectInput
              label="Target Type"
              value={jobConfig.config.process[0].network?.type ?? 'lora'}
              onChange={value => setJobConfig(value, 'config.process[0].network.type')}
              options={[
                { value: 'lora', label: 'LoRA' },
                { value: 'lokr', label: 'LoKr' },
              ]}
            />
            {jobConfig.config.process[0].network?.type == 'lokr' && (
              <SelectInput
                label="LoKr Factor"
                value={`${jobConfig.config.process[0].network?.lokr_factor ?? -1}`}
                onChange={value => setJobConfig(parseInt(value), 'config.process[0].network.lokr_factor')}
                options={[
                  { value: '-1', label: 'Auto' },
                  { value: '4', label: '4' },
                  { value: '8', label: '8' },
                  { value: '16', label: '16' },
                  { value: '32', label: '32' },
                ]}
              />
            )}
            {jobConfig.config.process[0].network?.type == 'lora' && (
              <>
                <NumberInput
                  label="Linear Rank"
                  value={jobConfig.config.process[0].network.linear}
                  onChange={value => {
                    console.log('onChange', value);
                    setJobConfig(value, 'config.process[0].network.linear');
                    setJobConfig(value, 'config.process[0].network.linear_alpha');
                  }}
                  placeholder="eg. 16"
                  min={0}
                  max={1024}
                  required
                />
                {disableSections.includes('network.conv') ? null : (
                  <NumberInput
                    label="Conv Rank"
                    value={jobConfig.config.process[0].network.conv}
                    onChange={value => {
                      console.log('onChange', value);
                      setJobConfig(value, 'config.process[0].network.conv');
                      setJobConfig(value, 'config.process[0].network.conv_alpha');
                    }}
                    placeholder="eg. 16"
                    min={0}
                    max={1024}
                  />
                )}
                {(() => {
                  const getInitConfig = (key: string) => {
                    const val = jobConfig.config.process[0].network?.[key];
                    if (!val) return { method: 'gaussian_random', std: undefined };
                    if (typeof val === 'string') return { method: val, std: undefined };
                    return val;
                  };
                  const setInitConfig = (key: string, method: string, std?: number) => {
                    if (method === 'gaussian_random' && std !== undefined) {
                      setJobConfig({ method, std }, key);
                    } else if (method === 'gaussian_random') {
                      setJobConfig({ method }, key);
                    } else {
                      setJobConfig(method, key);
                    }
                  };
                  const getMethod = (val: any) => {
                    if (!val) return 'gaussian_random';
                    if (typeof val === 'string') return val;
                    return val.method;
                  };
                  const getStd = (val: any) => {
                    if (!val || typeof val === 'string') return undefined;
                    return val.std;
                  };

                  // Hide general LoRA init selectors for multistage (per-expert) models
                  const isPerExpert =
                    modelArch?.additionalSections?.includes('model.multistage');

                  return (
                    <>
                      {!isPerExpert && (
                        <>
                          <SelectInput
                            label="LoRA A Matrix Init"
                            docKey="config.process[0].network.lora_a_init"
                            value={getMethod(jobConfig.config.process[0].network?.lora_a_init) || 'gaussian_random'}
                            onChange={value => {
                              const currentStd = getStd(jobConfig.config.process[0].network?.lora_a_init);
                              setInitConfig('config.process[0].network.lora_a_init', value, value === 'gaussian_random' ? currentStd : undefined);
                            }}
                            options={loraInitOptions}
                          />
                          {getMethod(jobConfig.config.process[0].network?.lora_a_init) === 'gaussian_random' && (
                            <NumberInput
                              label="LoRA A Init Std"
                              docKey="config.process[0].network.lora_a_init_std"
                              value={getStd(jobConfig.config.process[0].network?.lora_a_init) ?? ''}
                              onChange={value => {
                                const method = getMethod(jobConfig.config.process[0].network?.lora_a_init);
                                setInitConfig('config.process[0].network.lora_a_init', method, value);
                              }}
                              placeholder="Default: 1/√rank"
                              min={0}
                              max={10}
                              step={0.01}
                            />
                          )}
                          <SelectInput
                            label="LoRA B Matrix Init"
                            docKey="config.process[0].network.lora_b_init"
                            value={getMethod(jobConfig.config.process[0].network?.lora_b_init) || 'zeros'}
                            onChange={value => {
                              const currentStd = getStd(jobConfig.config.process[0].network?.lora_b_init);
                              setInitConfig('config.process[0].network.lora_b_init', value, value === 'gaussian_random' ? currentStd : undefined);
                            }}
                            options={loraInitOptions}
                          />
                          {getMethod(jobConfig.config.process[0].network?.lora_b_init) === 'gaussian_random' && (
                            <NumberInput
                              label="LoRA B Init Std"
                              docKey="config.process[0].network.lora_b_init_std"
                              value={getStd(jobConfig.config.process[0].network?.lora_b_init) ?? ''}
                              onChange={value => {
                                const method = getMethod(jobConfig.config.process[0].network?.lora_b_init);
                                setInitConfig('config.process[0].network.lora_b_init', method, value);
                              }}
                              placeholder="Default: 1/√rank"
                              min={0}
                              max={10}
                              step={0.01}
                            />
                          )}
                        </>
                      )}
                      {modelArch?.additionalSections?.includes('model.multistage') && (
                        <>
                          {jobConfig.config.process[0].model.model_kwargs?.train_high_noise && (
                            <>
                              <SelectInput
                                label="High Noise LoRA A Init"
                                docKey="config.process[0].network.high_noise_lora_a_init"
                                value={getMethod(jobConfig.config.process[0].network?.high_noise_lora_a_init) || getMethod(jobConfig.config.process[0].network?.lora_a_init) || 'gaussian_random'}
                                onChange={value => {
                                  const currentStd = getStd(jobConfig.config.process[0].network?.high_noise_lora_a_init);
                                  setInitConfig('config.process[0].network.high_noise_lora_a_init', value, value === 'gaussian_random' ? currentStd : undefined);
                                }}
                                options={loraInitOptions}
                              />
                              {getMethod(jobConfig.config.process[0].network?.high_noise_lora_a_init) === 'gaussian_random' && (
                                <NumberInput
                                  label="High Noise LoRA A Init Std"
                                  docKey="config.process[0].network.high_noise_lora_a_init_std"
                                  value={getStd(jobConfig.config.process[0].network?.high_noise_lora_a_init) ?? ''}
                                  onChange={value => {
                                    const method = getMethod(jobConfig.config.process[0].network?.high_noise_lora_a_init);
                                    setInitConfig('config.process[0].network.high_noise_lora_a_init', method, value);
                                  }}
                                  placeholder="Default: 1/√rank"
                                  min={0}
                                  max={10}
                                  step={0.01}
                                />
                              )}
                              <SelectInput
                                label="High Noise LoRA B Init"
                                docKey="config.process[0].network.high_noise_lora_b_init"
                                value={getMethod(jobConfig.config.process[0].network?.high_noise_lora_b_init) || getMethod(jobConfig.config.process[0].network?.lora_b_init) || 'zeros'}
                                onChange={value => {
                                  const currentStd = getStd(jobConfig.config.process[0].network?.high_noise_lora_b_init);
                                  setInitConfig('config.process[0].network.high_noise_lora_b_init', value, value === 'gaussian_random' ? currentStd : undefined);
                                }}
                                options={loraInitOptions}
                              />
                              {getMethod(jobConfig.config.process[0].network?.high_noise_lora_b_init) === 'gaussian_random' && (
                                <NumberInput
                                  label="High Noise LoRA B Init Std"
                                  docKey="config.process[0].network.high_noise_lora_b_init_std"
                                  value={getStd(jobConfig.config.process[0].network?.high_noise_lora_b_init) ?? ''}
                                  onChange={value => {
                                    const method = getMethod(jobConfig.config.process[0].network?.high_noise_lora_b_init);
                                    setInitConfig('config.process[0].network.high_noise_lora_b_init', method, value);
                                  }}
                                  placeholder="Default: 1/√rank"
                                  min={0}
                                  max={10}
                                  step={0.01}
                                />
                              )}
                            </>
                          )}
                          {jobConfig.config.process[0].model.model_kwargs?.train_low_noise && (
                            <>
                              <SelectInput
                                label="Low Noise LoRA A Init"
                                docKey="config.process[0].network.low_noise_lora_a_init"
                                value={getMethod(jobConfig.config.process[0].network?.low_noise_lora_a_init) || getMethod(jobConfig.config.process[0].network?.lora_a_init) || 'gaussian_random'}
                                onChange={value => {
                                  const currentStd = getStd(jobConfig.config.process[0].network?.low_noise_lora_a_init);
                                  setInitConfig('config.process[0].network.low_noise_lora_a_init', value, value === 'gaussian_random' ? currentStd : undefined);
                                }}
                                options={loraInitOptions}
                              />
                              {getMethod(jobConfig.config.process[0].network?.low_noise_lora_a_init) === 'gaussian_random' && (
                                <NumberInput
                                  label="Low Noise LoRA A Init Std"
                                  docKey="config.process[0].network.low_noise_lora_a_init_std"
                                  value={getStd(jobConfig.config.process[0].network?.low_noise_lora_a_init) ?? ''}
                                  onChange={value => {
                                    const method = getMethod(jobConfig.config.process[0].network?.low_noise_lora_a_init);
                                    setInitConfig('config.process[0].network.low_noise_lora_a_init', method, value);
                                  }}
                                  placeholder="Default: 1/√rank"
                                  min={0}
                                  max={10}
                                  step={0.01}
                                />
                              )}
                              <SelectInput
                                label="Low Noise LoRA B Init"
                                docKey="config.process[0].network.low_noise_lora_b_init"
                                value={getMethod(jobConfig.config.process[0].network?.low_noise_lora_b_init) || getMethod(jobConfig.config.process[0].network?.lora_b_init) || 'zeros'}
                                onChange={value => {
                                  const currentStd = getStd(jobConfig.config.process[0].network?.low_noise_lora_b_init);
                                  setInitConfig('config.process[0].network.low_noise_lora_b_init', value, value === 'gaussian_random' ? currentStd : undefined);
                                }}
                                options={loraInitOptions}
                              />
                              {getMethod(jobConfig.config.process[0].network?.low_noise_lora_b_init) === 'gaussian_random' && (
                                <NumberInput
                                  label="Low Noise LoRA B Init Std"
                                  docKey="config.process[0].network.low_noise_lora_b_init_std"
                                  value={getStd(jobConfig.config.process[0].network?.low_noise_lora_b_init) ?? ''}
                                  onChange={value => {
                                    const method = getMethod(jobConfig.config.process[0].network?.low_noise_lora_b_init);
                                    setInitConfig('config.process[0].network.low_noise_lora_b_init', method, value);
                                  }}
                                  placeholder="Default: 1/√rank"
                                  min={0}
                                  max={10}
                                  step={0.01}
                                />
                              )}
                            </>
                          )}
                        </>
                      )}
                    </>
                  );
                })()}
              </>
            )}
            {/* Wan 2.2 Tensor-Type-Specific LoRA Configuration */}
            {modelArch?.name?.includes('wan22') && jobConfig.config.process[0].network?.type == 'lora' && (
              <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid var(--border)' }}>
                <div style={{ fontWeight: 600, marginBottom: '8px', color: 'var(--accent)' }}>Wan 2.2 Tensor-Type-Specific Ranks</div>
                <div style={{ fontSize: '11px', color: 'var(--muted)', marginBottom: '12px' }}>
                  Control which tensor types are trained and their individual ranks. Leave a rank empty to use the Linear Rank above for that type.
                  Set rank to null/0 or uncheck "Train" to skip that type. Check "Full" for full weight training (no LoRA).
                </div>
                
                {/* Wan 2.2 Tensor Types Grid */}
                <div style={{ display: 'grid', gap: '8px' }}>
                  {Object.entries(WAN22_TENSOR_TYPES).map(([typeKey, typeInfo]) => {
                    const wan22Types = jobConfig.config.process[0].network?.wan22_tensor_types || {};
                    const typeConfig = wan22Types[typeKey] || {};
                    const rank = typeConfig.rank !== undefined ? typeConfig.rank : '';
                    const full = typeConfig.full || false;
                    const maxRank = typeInfo.maxRank;
                    
                    return (
                      <div key={typeKey} style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: '8px',
                        padding: '8px',
                        background: 'var(--bg-secondary)',
                        borderRadius: '6px'
                      }}>
                        <div style={{ flex: '0 0 120px', fontWeight: 500, fontSize: '13px' }}>
                          {typeInfo.name}
                          <div style={{ fontSize: '10px', color: 'var(--muted)' }}>{typeInfo.description}</div>
                        </div>
                        <div style={{ flex: '0 0 110px' }}>
                          <NumberInput
                            label={null}
                            value={rank === null || rank === 0 ? '' : (rank !== undefined ? rank : '')}
                            onChange={value => {
                              let newRank = value;
                              if (value === '' || value === null || value === undefined) {
                                newRank = null;  // empty = use linear rank
                              } else if (value === 0) {
                                newRank = 0;  // 0 = skip this type
                              } else {
                                newRank = parseInt(value);
                              }
                              const newTypes = { ...wan22Types, [typeKey]: { ...typeConfig, rank: newRank, alpha: newRank } };
                              setJobConfig(newTypes, 'config.process[0].network.wan22_tensor_types');
                            }}
                            placeholder="rank"
                            min={0}
                            max={maxRank}
                            style={{ height: '28px' }}
                          />
                        </div>
                        <div style={{ flex: '0 0 140px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <button
                            onClick={() => {
                              const newTypes = { ...wan22Types, [typeKey]: { ...typeConfig, rank: maxRank, alpha: maxRank, full: false } };
                              setJobConfig(newTypes, 'config.process[0].network.wan22_tensor_types');
                            }}
                            style={{
                              padding: '2px 6px',
                              fontSize: '10px',
                              cursor: 'pointer',
                              background: 'var(--accent)',
                              color: 'white',
                              border: 'none',
                              borderRadius: '4px'
                            }}
                          >
                            Max ({maxRank})
                          </button>
                          <button
                            onClick={() => {
                              const newTypes = { ...wan22Types, [typeKey]: { ...typeConfig, rank: null, alpha: null, full: false } };
                              setJobConfig(newTypes, 'config.process[0].network.wan22_tensor_types');
                            }}
                            style={{
                              padding: '2px 6px',
                              fontSize: '10px',
                              cursor: 'pointer',
                              background: 'var(--bg-tertiary)',
                              color: 'var(--text)',
                              border: '1px solid var(--border)',
                              borderRadius: '4px'
                            }}
                          >
                            Default
                          </button>
                        </div>
                        <div style={{ flex: '0 0 50px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <input
                            type="checkbox"
                            checked={full}
                            onChange={e => {
                              const newTypes = { ...wan22Types, [typeKey]: { ...typeConfig, full: e.target.checked } };
                              setJobConfig(newTypes, 'config.process[0].network.wan22_tensor_types');
                            }}
                            title="Full weight training (no LoRA)"
                          />
                          <label style={{ fontSize: '10px', color: 'var(--muted)' }}>Full</label>
                        </div>
                      </div>
                    );
                  })}
                </div>
                
                {/* Clear Wan 2.2 Config Button */}
                <button
                  onClick={() => {
                    setJobConfig(null, 'config.process[0].network.wan22_tensor_types');
                    setJobConfig(null, 'config.process[0].network.wan22_enabled_types');
                  }}
                  style={{
                    marginTop: '8px',
                    padding: '4px 8px',
                    fontSize: '10px',
                    cursor: 'pointer',
                    background: 'transparent',
                    color: 'var(--muted)',
                    border: '1px dashed var(--border)',
                    borderRadius: '4px',
                    width: '100%'
                  }}
                >
                  Clear Wan 2.2 Tensor Type Config (use Linear Rank for all)
                </button>
              </div>
            )}
            {/* Per-Layer Rank Overrides */}
            {jobConfig.config.process[0].network?.type == 'lora' && (
              <>
                <style jsx>{`input[type="number"]::-webkit-inner-spin-button,
                  input[type="number"]::-webkit-outer-spin-button {
                    -webkit-appearance: none;
                    margin: 0;
                  }
                  input[type="number"] {
                    -moz-appearance: textfield;
                  }`}</style>
                <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid var(--border)' }}>
                  <div style={{ fontWeight: 600, marginBottom: '8px', color: 'var(--accent)' }}>Per-Layer Rank Overrides</div>
                  <div style={{ fontSize: '11px', color: 'var(--muted)', marginBottom: '12px' }}>
                    Apply specific ranks to individual layers or tensor types. Useful for fine-tuning specific blocks (e.g., layers 0-15).
                    These settings override any global or Wan 2.2 tensor-type specific configurations.
                  </div>

                  {/* Helper components */}
                  {(() => {
                    const validateLayerRange = (rangeStr: string): boolean => {
                      if (!rangeStr || rangeStr.trim() === '') return false;
                      const segments = rangeStr.split(',').map(s => s.trim()).filter(Boolean);
                      for (const seg of segments) {
                        if (!/^(\d+)(?:[.]{1,3}\d+|\-\d+)$/.test(seg) && !/^\d+$/.test(seg)) {
                          return false;
                        }
                      }
                      return true;
                    };

                    const TensorOverrideRow = ({
                      override,
                      index,
                      configKey,
                      accentColor
                    }: {
                      override: { tensor_type: string; rank: number; layer_range: string };
                      index: number;
                      configKey: string;
                      accentColor?: string;
                    }) => {
                      // Resolve aliases when loading from config (e.g., 'ffn.up' -> 'ffn.0')
                      const canonicalType = resolveTensorType(override.tensor_type);
                      const isSingleTensor = isSingleTensorType(canonicalType);
                      const isDisabled = override.rank <= 0; // Tensor excluded from training
                      const isRangeValid = isSingleTensor || validateLayerRange(override.layer_range);
                      const maxLayers = getMaxLayers(canonicalType);

                      const updateOverride = (field: string, value: any) => {
                        const current = (jobConfig.config.process[0].network as any)?.[configKey] || [];
                        const newOverrides = [...current];
                        newOverrides[index] = { ...override, [field]: value };
                        setJobConfig(newOverrides, `config.process[0].network.${configKey}`);
                      };

                      const removeOverride = () => {
                        const current = (jobConfig.config.process[0].network as any)?.[configKey] || [];
                        const newOverrides = current.filter((_, idx) => idx !== index);
                        setJobConfig(newOverrides, `config.process[0].network.${configKey}`);
                      };

                      return (
                        <div key={index} style={{
                          display: 'grid',
                          gridTemplateColumns: '1fr 1fr 1fr auto',
                          gap: '8px',
                          padding: '8px',
                          background: isDisabled ? '#1a1a2e' : 'var(--bg-secondary)',
                          borderRadius: '6px',
                          alignItems: 'center',
                          opacity: isDisabled ? 0.7 : 1,
                          border: isDisabled ? '1px dashed #ef4444' : '1px solid transparent'
                        }}>
                          <SelectInput
                            label={null}
                            value={canonicalType} // Use resolved canonical type for display
                            onChange={(value) => updateOverride('tensor_type', value)}
                            options={Object.entries(WAN22_TENSOR_TYPES).map(([typeKey, typeInfo]) => ({
                              value: typeKey,
                              label: typeInfo.name
                            }))}
                            style={{ height: '28px' }}
                          />
                          <NumberInput
                            label={null}
                            value={override.rank}
                            onChange={(value) => updateOverride('rank', value)}
                            placeholder="rank"
                            style={{
                              height: '28px',
                              border: isDisabled ? '1px solid #ef4444' : undefined,
                              background: isDisabled ? '#2d1b1b' : undefined
                            }}
                          />
                          <input
                            type="text"
                            value={isSingleTensor ? '' : override.layer_range}
                            onChange={(e) => !isSingleTensor && updateOverride('layer_range', e.target.value)}
                            placeholder={isSingleTensor ? "N/A (single tensor)" : "0-2, 1-4,5-6, or 0,1,2"}
                            disabled={isSingleTensor || isDisabled}
                            style={{
                              height: '28px',
                              background: isSingleTensor ? '#111827' : 'var(--bg-tertiary)',
                              border: `1px solid ${isRangeValid ? 'var(--border)' : '#ef4444'}`,
                              borderRadius: '4px',
                              color: isSingleTensor ? '#4b5563' : 'var(--text)',
                              padding: '0 8px',
                              cursor: isSingleTensor ? 'not-allowed' : 'text'
                            }}
                          />
                          <button
                            onClick={removeOverride}
                            style={{
                              padding: '4px 8px',
                              fontSize: '10px',
                              cursor: 'pointer',
                              background: 'var(--bg-tertiary)',
                              color: 'var(--text)',
                              border: '1px solid var(--border)',
                              borderRadius: '4px'
                            }}
                          >
                            <X className="w-4 h-4" />
                          </button>
                          <div style={{ gridColumn: '1 / -1', fontSize: '10px', color: isDisabled ? '#ef4444' : (isRangeValid ? 'var(--muted)' : '#ef4444'), marginTop: '4px' }}>
                            {isDisabled
                              ? "⚠ Tensor EXCLUDED from LoRA training (rank ≤ 0). Set rank > 0 to enable."
                              : isSingleTensor
                                ? "Single tensor (no layers). Layer range is ignored."
                                : !isRangeValid && override.layer_range.trim() !== ''
                                  ? "Invalid format. Use: 0-2, 1-4,5-6, or 0,1,2"
                                  : maxLayers > 0
                                    ? `Available layers: 0 to ${maxLayers - 1} (e.g., 0-${maxLayers - 1})`
                                    : "Specify layer range like 0-2, 1-4,5-6, or 0,1,2"}
                          </div>
                        </div>
                      );
                    };

                    const OverrideSection = ({
                      title,
                      description,
                      configKey,
                      accentColor,
                      borderColor,
                      exampleOverride
                    }: {
                      title: string;
                      description: string;
                      configKey: string;
                      accentColor?: string;
                      borderColor?: string;
                      exampleOverride: { tensor_type: string; rank: number; layer_range: string };
                    }) => {
                      const overrides = (jobConfig.config.process[0].network as any)?.[configKey] || [];

                      return (
                        <div style={{
                          marginBottom: '12px',
                          padding: '10px',
                          background: 'var(--bg-tertiary)',
                          borderRadius: '6px',
                          border: `1px solid ${borderColor || 'var(--border)'}`
                        }}>
                          <div style={{ fontWeight: 600, marginBottom: '4px', color: accentColor || 'var(--accent)', fontSize: '12px' }}>
                            {title}
                          </div>
                          <div style={{ fontSize: '10px', color: 'var(--muted)', marginBottom: '8px' }}>
                            {description}
                          </div>
                          {overrides.length > 0 && (
                            <div style={{ display: 'grid', gap: '8px', marginBottom: '8px' }}>
                              {overrides.map((override, i) => (
                                <TensorOverrideRow
                                  key={i}
                                  override={override}
                                  index={i}
                                  configKey={configKey}
                                  accentColor={accentColor}
                                />
                              ))}
                            </div>
                          )}
                          <button
                            onClick={() => {
                              const current = (jobConfig.config.process[0].network as any)?.[configKey] || [];
                              const newOverrides = [...current, exampleOverride];
                              setJobConfig(newOverrides, `config.process[0].network.${configKey}`);
                            }}
                            style={{
                              padding: '4px 8px',
                              fontSize: '10px',
                              cursor: 'pointer',
                              background: accentColor || 'var(--accent)',
                              color: 'white',
                              border: 'none',
                              borderRadius: '4px',
                              marginRight: '6px'
                            }}
                          >
                            Add Override
                          </button>
                          <button
                            onClick={() => setJobConfig([], `config.process[0].network.${configKey}`)}
                            style={{
                              padding: '4px 8px',
                              fontSize: '10px',
                              cursor: 'pointer',
                              background: 'transparent',
                              color: 'var(--muted)',
                              border: '1px dashed var(--border)',
                              borderRadius: '4px'
                            }}
                          >
                            Clear
                          </button>
                        </div>
                      );
                    };

                    return (
                      <>
                        {/* Global Overrides */}
                        <OverrideSection
                          title="Global Overrides (all experts)"
                          description="Applies to all transformer experts. Can be overridden by per-expert settings below."
                          configKey="layer_overrides"
                          exampleOverride={{ tensor_type: 'self_attn.v', rank: 64, layer_range: '0-2' }}
                        />

                        {/* High-Noise Expert Overrides */}
                        <OverrideSection
                          title="High-Noise Expert Overrides (transformer_1)"
                          description="Overrides global settings for the high-noise expert (transformer_1). Empty = uses global overrides."
                          configKey="layer_overrides_high"
                          accentColor="#f59e0b"
                          borderColor="#f59e0b33"
                          exampleOverride={{ tensor_type: 'self_attn.q', rank: 128, layer_range: '0-10' }}
                        />

                        {/* Low-Noise Expert Overrides */}
                        <OverrideSection
                          title="Low-Noise Expert Overrides (transformer_2)"
                          description="Overrides global settings for the low-noise expert (transformer_2). Empty = uses global overrides."
                          configKey="layer_overrides_low"
                          accentColor="#06b6d4"
                          borderColor="#06b6d433"
                          exampleOverride={{ tensor_type: 'self_attn.k', rank: 96, layer_range: '20-39' }}
                        />
                      </>
                    );
                  })()}
                </div>
              </>
            )}
            {/* Rank Gate Annealing (SparseForge-inspired) */}
            {jobConfig.config.process[0].network?.type == 'lora' && (
              <RankGateConfigSection
                jobConfig={jobConfig}
                setJobConfig={setJobConfig}
              />
            )}
          </Card>
          {!disableSections.includes('slider') && (
            <Card title="Slider">
              <TextInput
                label="Target Class"
                className=""
                value={jobConfig.config.process[0].slider?.target_class ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.target_class')}
                placeholder="eg. person"
              />
              <TextInput
                label="Positive Prompt"
                className=""
                value={jobConfig.config.process[0].slider?.positive_prompt ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.positive_prompt')}
                placeholder="eg. person who is happy"
              />
              <TextInput
                label="Negative Prompt"
                className=""
                value={jobConfig.config.process[0].slider?.negative_prompt ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.negative_prompt')}
                placeholder="eg. person who is sad"
              />
              <TextInput
                label="Anchor Class"
                className=""
                value={jobConfig.config.process[0].slider?.anchor_class ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.anchor_class')}
                placeholder=""
              />
            </Card>
          )}
          <Card title="Save">
            <SelectInput
              label="Data Type"
              value={jobConfig.config.process[0].save.dtype}
              onChange={value => setJobConfig(value, 'config.process[0].save.dtype')}
              options={[
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'fp32', label: 'FP32' },
              ]}
            />
            <NumberInput
              label="Save Every"
              value={jobConfig.config.process[0].save.save_every}
              onChange={value => setJobConfig(value, 'config.process[0].save.save_every')}
              placeholder="eg. 250"
              min={1}
              required
            />
            <NumberInput
              label="Max Step Saves to Keep"
              value={jobConfig.config.process[0].save.max_step_saves_to_keep}
              onChange={value => setJobConfig(value, 'config.process[0].save.max_step_saves_to_keep')}
              placeholder="eg. 4"
              min={1}
              required
            />
          </Card>
        </div>
        <div>
          <Card title="Training">
            <div className={trainingBarClass}>
              <div>
                <NumberInput
                  label="Batch Size"
                  value={jobConfig.config.process[0].train.batch_size}
                  onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                  placeholder="eg. 4"
                  min={1}
                  required
                />
                <NumberInput
                  label="Gradient Accumulation"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.gradient_accumulation}
                  onChange={value => setJobConfig(value, 'config.process[0].train.gradient_accumulation')}
                  placeholder="eg. 1"
                  min={1}
                  required
                />
                <NumberInput
                  label="Steps"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.steps}
                  onChange={value => setJobConfig(value, 'config.process[0].train.steps')}
                  placeholder="eg. 2000"
                  min={1}
                  required
                />
                <SelectInput
                  label="Noise Scheduler"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.noise_scheduler}
                  onChange={value => setJobConfig(value, 'config.process[0].train.noise_scheduler')}
                  options={schedulerOptions}
                />
              </div>
              <div>
                <SelectInput
                  label="Optimizer"
                  value={jobConfig.config.process[0].train.optimizer}
                  onChange={value => {
                    setJobConfig(value, 'config.process[0].train.optimizer');
                    const isNoScheduler = NO_SCHEDULER_OPTIMIZERS.includes(value);
                    if (isNoScheduler) {
                      // Lock to 'none' and clear params for optimizers that don't need schedulers
                      setJobConfig('none', 'config.process[0].train.lr_scheduler');
                      setJobConfig({}, 'config.process[0].train.lr_scheduler_params');
                    } else {
                      // Reset to 'none' with defaults when switching away from no-scheduler optimizer
                      const currentScheduler = jobConfig.config.process[0].train.lr_scheduler;
                      if (currentScheduler && currentScheduler !== 'none') {
                        // Keep current scheduler but reset its params to defaults
                        const defaults = SCHEDULER_DEFAULT_PARAMS[currentScheduler] || {};
                        setJobConfig(defaults, 'config.process[0].train.lr_scheduler_params');
                      }
                    }
                  }}
                  docKey="optimizer"
                  options={[
                    { value: 'adam', label: 'Adam' },
                    { value: 'adamw', label: 'AdamW' },
                    { value: 'adamw_fused', label: 'AdamW Fused' },
                    { value: 'adamw8bit', label: 'AdamW8Bit' },
                    { value: 'adamw_fp8', label: 'AdamW FP8' },
                    { value: 'adamw_bf16', label: 'AdamW BF16' },
                    { value: 'adam8', label: 'Adam8Bit' },
                    { value: 'adamw8', label: 'AdamW8Bit' },
                    { value: 'lion', label: 'Lion' },
                    { value: 'lion8bit', label: 'Lion8Bit' },
                    { value: 'adagrad', label: 'Adagrad' },
                    { value: 'adafactor', label: 'Adafactor' },
                    { value: 'dadaptation', label: 'DAdaptAdam' },
                    { value: 'dadaptationlion', label: 'DAdaptLion' },
                    { value: 'prodigyopt', label: 'Prodigy' },
                    { value: 'prodigy8bit', label: 'Prodigy8Bit' },
                    { value: 'automagic', label: 'Automagic' },
                    { value: 'automagic2', label: 'Automagic v2' },
                    { value: 'automagic3', label: 'Automagic v3' },
                  ]}
                />
                <SelectInput
                  label="LR Scheduler"
                  value={jobConfig.config.process[0].train.lr_scheduler}
                  onChange={value => {
                    setJobConfig(value, 'config.process[0].train.lr_scheduler');
                    // Reset params to defaults for the selected scheduler
                    const defaults = SCHEDULER_DEFAULT_PARAMS[value] || {};
                    setJobConfig(defaults, 'config.process[0].train.lr_scheduler_params');
                  }}
                  docKey="lr_scheduler"
                  options={[
                    { value: 'none', label: 'None' },
                    { value: 'cosine', label: 'Cosine' },
                    { value: 'cosine_with_restarts', label: 'Cosine with Restarts' },
                    { value: 'step', label: 'Step' },
                    { value: 'polynomial', label: 'Polynomial' },
                    { value: 'constant', label: 'Constant' },
                    { value: 'linear', label: 'Linear' },
                    { value: 'constant_with_warmup', label: 'Constant with Warmup' },
                  ]}
                  disabled={NO_SCHEDULER_OPTIMIZERS.includes(jobConfig.config.process[0].train.optimizer)}
                />
                {jobConfig.config.process[0].train.lr_scheduler === 'cosine' && (
                  <NumberInput
                    label="Total Iters (T_max)"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.lr_scheduler_params?.total_iters}
                    onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.total_iters')}
                    placeholder="Defaults to training steps"
                    min={1}
                  />
                )}
                {jobConfig.config.process[0].train.lr_scheduler === 'cosine_with_restarts' && (
                  <>
                    <NumberInput
                      label="T_0 (Initial Period)"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.T_0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.T_0')}
                      placeholder="Defaults to training steps"
                      min={1}
                    />
                    <NumberInput
                      label="T_mult"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.T_mult}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.T_mult')}
                      placeholder="eg. 2"
                      min={1}
                    />
                    <NumberInput
                      label="Eta Min"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.eta_min}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.eta_min')}
                      placeholder="eg. 0.0"
                      min={0}
                    />
                  </>
                )}
                {jobConfig.config.process[0].train.lr_scheduler === 'step' && (
                  <>
                    <NumberInput
                      label="Step Size"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.step_size}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.step_size')}
                      placeholder="eg. 100"
                      min={1}
                    />
                    <NumberInput
                      label="Gamma"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.gamma}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.gamma')}
                      placeholder="eg. 0.1"
                      min={0}
                      max={1}
                    />
                  </>
                )}
                {jobConfig.config.process[0].train.lr_scheduler === 'polynomial' && (
                  <>
                    <NumberInput
                      label="Power"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.power}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.power')}
                      placeholder="eg. 0.8"
                      min={0.01}
                      max={10}
                    />
                    <NumberInput
                      label="LR End"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.lr_end}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.lr_end')}
                      placeholder="eg. 0.0"
                      min={0}
                    />
                  </>
                )}
                {jobConfig.config.process[0].train.lr_scheduler === 'constant' && (
                  <>
                    <NumberInput
                      label="Factor"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.factor}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.factor')}
                      placeholder="eg. 1.0"
                      min={0}
                    />
                    <NumberInput
                      label="Step Size"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.step_size}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.step_size')}
                      placeholder="eg. 100"
                      min={1}
                    />
                    <NumberInput
                      label="End Factor"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.end_factor}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.end_factor')}
                      placeholder="eg. 1.0"
                      min={0}
                    />
                  </>
                )}
                {jobConfig.config.process[0].train.lr_scheduler === 'linear' && (
                  <>
                    <NumberInput
                      label="Start Factor"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.start_factor}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.start_factor')}
                      placeholder="eg. 1.0"
                      min={0}
                    />
                    <NumberInput
                      label="End Factor"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.end_factor}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.end_factor')}
                      placeholder="eg. 0.0"
                      min={0}
                    />
                    <NumberInput
                      label="Total Iters"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.lr_scheduler_params?.total_iters}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.total_iters')}
                      placeholder="Defaults to training steps"
                      min={1}
                    />
                  </>
                )}
                {jobConfig.config.process[0].train.lr_scheduler === 'constant_with_warmup' && (
                  <NumberInput
                    label="Warmup Steps"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.lr_scheduler_params?.num_warmup_steps}
                    onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler_params.num_warmup_steps')}
                    placeholder="Defaults to 1000"
                    min={0}
                  />
                )}

                {/* Per-Expert LR Schedulers for Wan 2.2 14B Dual-Expert Models */}
                {modelArch?.additionalSections?.includes('model.multistage') &&
                  jobConfig.config.process[0].model.model_kwargs?.train_high_noise &&
                  jobConfig.config.process[0].model.model_kwargs?.train_low_noise && (
                  <div className="pt-3 pb-2 border-t border-gray-700">
                    <p className="text-xs text-blue-400 mb-2 font-medium">
                      Per-Expert LR Schedulers (Wan 2.2 14B Dual-Expert)
                    </p>
                    <p className="text-[10px] text-gray-500 mb-2">
                      Each expert gets its own scheduler tracking expert-specific steps. Leave empty to use global scheduler for each expert.
                    </p>

                    {/* Expert 1 (High-Noise) Scheduler */}
                    <div className="mb-2">
                      <p className="text-[10px] text-green-400 mb-1 font-medium">High-Noise Expert (Transformer 1)</p>
                      <SelectInput
                        label="Expert 1 LR Scheduler"
                        value={jobConfig.config.process[0].train.expert_1_lr_scheduler || 'Use global scheduler'}
                        onChange={value => {
                          if (value === 'Use global scheduler') {
                            setJobConfig(undefined, 'config.process[0].train.expert_1_lr_scheduler');
                            setJobConfig(undefined, 'config.process[0].train.expert_1_lr_scheduler_params');
                          } else {
                            setJobConfig(value, 'config.process[0].train.expert_1_lr_scheduler');
                            const defaults = SCHEDULER_DEFAULT_PARAMS[value] || {};
                            setJobConfig(defaults, 'config.process[0].train.expert_1_lr_scheduler_params');
                          }
                        }}
                        options={[
                          { value: 'Use global scheduler', label: 'Use global scheduler' },
                          { value: 'none', label: 'None (constant LR)' },
                          { value: 'cosine', label: 'Cosine' },
                          { value: 'cosine_with_restarts', label: 'Cosine with Restarts' },
                          { value: 'step', label: 'Step' },
                          { value: 'polynomial', label: 'Polynomial' },
                          { value: 'constant', label: 'Constant' },
                          { value: 'linear', label: 'Linear' },
                          { value: 'constant_with_warmup', label: 'Constant with Warmup' },
                        ]}
                      />
                    </div>

                    {/* Expert 1 Scheduler Params */}
                    {jobConfig.config.process[0].train.expert_1_lr_scheduler === 'cosine' && (
                      <NumberInput
                        label="Expert 1 Total Iters"
                        className="pt-1"
                        value={jobConfig.config.process[0].train.expert_1_lr_scheduler_params?.total_iters ?? null}
                        onChange={value => setJobConfig(value, 'config.process[0].train.expert_1_lr_scheduler_params.total_iters')}
                        placeholder="Defaults to training steps"
                        min={1}
                      />
                    )}
                    {jobConfig.config.process[0].train.expert_1_lr_scheduler === 'polynomial' && (
                      <>
                        <NumberInput
                          label="Expert 1 Power"
                          className="pt-1"
                          value={jobConfig.config.process[0].train.expert_1_lr_scheduler_params?.power ?? null}
                          onChange={value => setJobConfig(value, 'config.process[0].train.expert_1_lr_scheduler_params.power')}
                          placeholder="eg. 0.8"
                          min={0.01}
                          max={10}
                        />
                        <NumberInput
                          label="Expert 1 LR End"
                          className="pt-1"
                          value={jobConfig.config.process[0].train.expert_1_lr_scheduler_params?.lr_end ?? null}
                          onChange={value => setJobConfig(value, 'config.process[0].train.expert_1_lr_scheduler_params.lr_end')}
                          placeholder="eg. 0.0"
                          min={0}
                        />
                      </>
                    )}

                    {/* Expert 2 (Low-Noise) Scheduler */}
                    <div className="mb-2 mt-2">
                      <p className="text-[10px] text-purple-400 mb-1 font-medium">Low-Noise Expert (Transformer 2)</p>
                      <SelectInput
                        label="Expert 2 LR Scheduler"
                        value={jobConfig.config.process[0].train.expert_2_lr_scheduler || 'Use global scheduler'}
                        onChange={value => {
                          if (value === 'Use global scheduler') {
                            setJobConfig(undefined, 'config.process[0].train.expert_2_lr_scheduler');
                            setJobConfig(undefined, 'config.process[0].train.expert_2_lr_scheduler_params');
                          } else {
                            setJobConfig(value, 'config.process[0].train.expert_2_lr_scheduler');
                            const defaults = SCHEDULER_DEFAULT_PARAMS[value] || {};
                            setJobConfig(defaults, 'config.process[0].train.expert_2_lr_scheduler_params');
                          }
                        }}
                        options={[
                          { value: 'Use global scheduler', label: 'Use global scheduler' },
                          { value: 'none', label: 'None (constant LR)' },
                          { value: 'cosine', label: 'Cosine' },
                          { value: 'cosine_with_restarts', label: 'Cosine with Restarts' },
                          { value: 'step', label: 'Step' },
                          { value: 'polynomial', label: 'Polynomial' },
                          { value: 'constant', label: 'Constant' },
                          { value: 'linear', label: 'Linear' },
                          { value: 'constant_with_warmup', label: 'Constant with Warmup' },
                        ]}
                      />
                    </div>

                    {/* Expert 2 Scheduler Params */}
                    {jobConfig.config.process[0].train.expert_2_lr_scheduler === 'cosine' && (
                      <NumberInput
                        label="Expert 2 Total Iters"
                        className="pt-1"
                        value={jobConfig.config.process[0].train.expert_2_lr_scheduler_params?.total_iters ?? null}
                        onChange={value => setJobConfig(value, 'config.process[0].train.expert_2_lr_scheduler_params.total_iters')}
                        placeholder="Defaults to training steps"
                        min={1}
                      />
                    )}
                    {jobConfig.config.process[0].train.expert_2_lr_scheduler === 'polynomial' && (
                      <>
                        <NumberInput
                          label="Expert 2 Power"
                          className="pt-1"
                          value={jobConfig.config.process[0].train.expert_2_lr_scheduler_params?.power ?? null}
                          onChange={value => setJobConfig(value, 'config.process[0].train.expert_2_lr_scheduler_params.power')}
                          placeholder="eg. 0.8"
                          min={0.01}
                          max={10}
                        />
                        <NumberInput
                          label="Expert 2 LR End"
                          className="pt-1"
                          value={jobConfig.config.process[0].train.expert_2_lr_scheduler_params?.lr_end ?? null}
                          onChange={value => setJobConfig(value, 'config.process[0].train.expert_2_lr_scheduler_params.lr_end')}
                          placeholder="eg. 0.0"
                          min={0}
                        />
                      </>
                    )}
                  </div>
                )}

                <NumberInput
                  label="Learning Rate"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.lr}
                  onChange={value => setJobConfig(value, 'config.process[0].train.lr')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
                <NumberInput
                  label="Weight Decay"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.optimizer_params.weight_decay}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer_params.weight_decay')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
              </div>
              <div>
                {disableSections.includes('train.timestep_type') ? null : (
                  <SelectInput
                    label="Timestep Type"
                    value={jobConfig.config.process[0].train.timestep_type}
                    disabled={disableSections.includes('train.timestep_type') || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.timestep_type')}
                    options={[
                      { value: 'sigmoid', label: 'Sigmoid' },
                      { value: 'linear', label: 'Linear' },
                      { value: 'shift', label: 'Shift' },
                      { value: 'weighted', label: 'Weighted' },
                    ]}
                  />
                )}
                <SelectInput
                  label="Timestep Bias"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.content_or_style}
                  onChange={value => setJobConfig(value, 'config.process[0].train.content_or_style')}
                  options={[
                    { value: 'balanced', label: 'Balanced' },
                    { value: 'content', label: 'High Noise' },
                    { value: 'style', label: 'Low Noise' },
                  ]}
                />
                <SelectInput
                  label="Loss Type"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.loss_type}
                  onChange={value => setJobConfig(value, 'config.process[0].train.loss_type')}
                  options={[
                    { value: 'mse', label: 'Mean Squared Error' },
                    { value: 'mae', label: 'Mean Absolute Error' },
                    { value: 'wavelet', label: 'Wavelet' },
                    { value: 'spectral', label: 'Spectral (Frequency Balancing)' },
                    { value: 'spectral_flow', label: 'Spectral + Flow (Video Motion)' },
                    { value: 'mse_spectral_flow', label: 'MSE + Spectral + Flow (Video)' },
                    { value: 'stepped', label: 'Stepped Recovery' },
                    { value: 'pseudo_huber', label: 'Pseudo Huber (Smooth L1)' },
                    { value: 'mean_flow', label: 'Mean Flow (Experimental)' },
                  ]}
                />
                <SelectInput
                  label="Prediction Target"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.prediction_target ?? 'velocity'}
                  onChange={value => setJobConfig(value, 'config.process[0].train.prediction_target')}
                  options={[
                    { value: 'velocity', label: 'Velocity (default for flow-matching)' },
                    { value: 'x0', label: 'x0 (direct latent prediction)' },
                  ]}
                />
                <p className="text-xs text-gray-500 -mt-1">
                  Velocity: model predicts ε - x₀ (standard for pretrained flow-matching models).
                  x0: model predicts x₀ directly (requires compatible backbone; x0 reconstruction changes).
                </p>
                {jobConfig.config.process[0].train.loss_type === 'pseudo_huber' && (
                  <NumberInput
                    label="Pseudo Huber C Value"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.pseudo_huber_c ?? 0.01}
                    onChange={value => setJobConfig(value, 'config.process[0].train.pseudo_huber_c')}
                    placeholder="0.01"
                    docKey={'train.pseudo_huber_c'}
                    min={0}
                    step={0.001}
                  />
                )}
                {(jobConfig.config.process[0].train.loss_type === 'spectral_flow' || jobConfig.config.process[0].train.loss_type === 'spectral' || jobConfig.config.process[0].train.loss_type === 'mse_spectral_flow') && (
                  <div className="border border-blue-900 rounded-lg p-4 mt-2 space-y-3">
                    <p className="text-xs text-blue-300">
                      {jobConfig.config.process[0].train.loss_type === 'mse_spectral_flow'
                        ? 'MSE+Spectral+Flow loss combines standard MSE, spectral (spatial frequency), and optical flow (temporal motion) consistency. '
                        : jobConfig.config.process[0].train.loss_type === 'spectral_flow'
                          ? 'Spectral+Flow loss combines spectral (spatial frequency) with optical flow (temporal motion) consistency. '
                          : ''}Spectral loss dissociates low frequencies (structure/motion) from high frequencies (texture/details).
                      {(jobConfig.config.process[0].train.loss_type === 'spectral_flow' || jobConfig.config.process[0].train.loss_type === 'mse_spectral_flow') && 'Requires cache_optical_flow_to_disk enabled on video datasets.'}
                    </p>
                    {jobConfig.config.process[0].train.loss_type === 'mse_spectral_flow' && (
                      <div className="border-t border-blue-900 pt-3 mt-2">
                        <p className="text-xs text-blue-300 mb-2">MSE Loss Weights (per-expert)</p>
                        <div className="grid grid-cols-3 gap-3">
                          <NumberInput
                            label="MSE Weight (global)"
                            className="pt-1"
                            value={jobConfig.config.process[0].train.mse_spectral_flow_mse_weight ?? 1.0}
                            onChange={value => setJobConfig(value, 'config.process[0].train.mse_spectral_flow_mse_weight')}
                            placeholder="1.0"
                            docKey={'train.mse_spectral_flow_mse_weight'}
                            min={0}
                            step={0.1}
                          />
                          <NumberInput
                            label="MSE Weight (low noise)"
                            className="pt-1"
                            value={jobConfig.config.process[0].train.mse_spectral_flow_mse_weight_low ?? 1.0}
                            onChange={value => setJobConfig(value, 'config.process[0].train.mse_spectral_flow_mse_weight_low')}
                            placeholder="1.0"
                            docKey={'train.mse_spectral_flow_mse_weight_low'}
                            min={0}
                            step={0.1}
                          />
                          <NumberInput
                            label="MSE Weight (high noise)"
                            className="pt-1"
                            value={jobConfig.config.process[0].train.mse_spectral_flow_mse_weight_high ?? 1.0}
                            onChange={value => setJobConfig(value, 'config.process[0].train.mse_spectral_flow_mse_weight_high')}
                            placeholder="1.0"
                            docKey={'train.mse_spectral_flow_mse_weight_high'}
                            min={0}
                            step={0.1}
                          />
                        </div>
                        <p className="text-xs text-gray-500 mt-1">
                          Per-expert weights override global. MSE provides standard diffusion training signal.
                          Recommended: 0.5-2.0 for Wan 2.2 I2V LoRA.
                        </p>
                      </div>
                    )}
                    <div className="grid grid-cols-3 gap-3">
                      <NumberInput
                        label="Low Freq Weight"
                        className="pt-1"
                        value={jobConfig.config.process[0].train.spectral_low_weight ?? 1.0}
                        onChange={value => setJobConfig(value, 'config.process[0].train.spectral_low_weight')}
                        placeholder="1.0"
                        docKey={'train.spectral_low_weight'}
                        min={0}
                        step={0.1}
                      />
                      <NumberInput
                        label="Mid Freq Weight"
                        className="pt-1"
                        value={jobConfig.config.process[0].train.spectral_mid_weight ?? 1.0}
                        onChange={value => setJobConfig(value, 'config.process[0].train.spectral_mid_weight')}
                        placeholder="1.0"
                        docKey={'train.spectral_mid_weight'}
                        min={0}
                        step={0.1}
                      />
                      <NumberInput
                        label="High Freq Weight"
                        className="pt-1"
                        value={jobConfig.config.process[0].train.spectral_high_weight ?? 2.0}
                        onChange={value => setJobConfig(value, 'config.process[0].train.spectral_high_weight')}
                        placeholder="2.0"
                        docKey={'train.spectral_high_weight'}
                        min={0}
                        step={0.1}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <NumberInput
                        label="Low Cutoff (0-1)"
                        className="pt-1"
                        value={jobConfig.config.process[0].train.spectral_low_cutoff ?? 0.15}
                        onChange={value => setJobConfig(value, 'config.process[0].train.spectral_low_cutoff')}
                        placeholder="0.15"
                        docKey={'train.spectral_low_cutoff'}
                        min={0}
                        max={1}
                        step={0.01}
                      />
                      <NumberInput
                        label="High Cutoff (0-1)"
                        className="pt-1"
                        value={jobConfig.config.process[0].train.spectral_high_cutoff ?? 0.5}
                        onChange={value => setJobConfig(value, 'config.process[0].train.spectral_high_cutoff')}
                        placeholder="0.5"
                        docKey={'train.spectral_high_cutoff'}
                        min={0}
                        max={1}
                        step={0.01}
                      />
                    </div>
                    <div className="flex items-center space-x-2 pt-1">
                      <input
                        type="checkbox"
                        id="spectral_use_phase"
                        checked={jobConfig.config.process[0].train.spectral_use_phase ?? true}
                        onChange={e => setJobConfig(e.target.checked, 'config.process[0].train.spectral_use_phase')}
                        className="rounded border-gray-600"
                      />
                      <label htmlFor="spectral_use_phase" className="text-sm text-gray-300">
                        Use Phase Information (more accurate, slightly slower)
                      </label>
                    </div>
                    <NumberInput
                      label="LCR Weight (SSVAE-inspired)"
                      className="pt-1"
                      value={jobConfig.config.process[0].train.spectral_lcr_weight ?? 0.0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.spectral_lcr_weight')}
                      placeholder="0.0 (disabled)"
                      docKey={'train.spectral_lcr_weight'}
                      min={0}
                      step={0.01}
                    />
                    <p className="text-xs text-gray-500">
                      Local Correlation Regularization encourages low-frequency bias in latents.
                      Per SSVAE research, this improves diffusability and convergence speed.
                      Try 0.01-0.1 for video models. 0.0 = disabled.
                    </p>
                    <div>
                      <label className="block text-xs font-medium text-gray-300 mb-1">
                        Transform Type
                      </label>
                      <select
                        className="w-full bg-gray-800 text-gray-200 text-xs rounded px-2 py-1.5 border border-gray-600 focus:border-blue-500 focus:outline-none"
                        value={jobConfig.config.process[0].train.spectral_transform ?? 'dct'}
                        onChange={e => setJobConfig(e.target.value, 'config.process[0].train.spectral_transform')}
                      >
                        <option value="dct">DCT (default, SSVAE-compliant, mirror boundaries)</option>
                        <option value="fft">FFT (fast, periodic boundaries)</option>
                      </select>
                      <p className="text-xs text-gray-500 mt-1">
                        DCT: real-valued with mirror boundaries — matches SSVAE paper methodology.<br />
                        FFT: complex transform with periodic boundaries (faster).
                      </p>
                    </div>
                  </div>
                )}
                {(jobConfig.config.process[0].train.loss_type === 'spectral_flow' || jobConfig.config.process[0].train.loss_type === 'mse_spectral_flow') && (
                  <div className="border border-blue-800 rounded-lg p-4 mt-2 space-y-3">
                    <p className="text-xs text-blue-400">
                      Flow Loss Settings (temporal motion consistency)
                    </p>
                    <div className="space-y-2">
                      <NumberInput
                        label="Flow Loss Weight (global)"
                        className="pt-1"
                        value={jobConfig.config.process[0].train.spectral_flow_weight ?? 0.1}
                        onChange={value => setJobConfig(value, 'config.process[0].train.spectral_flow_weight')}
                        placeholder="0.1"
                        docKey={'train.spectral_flow_weight'}
                        min={0}
                        step={0.01}
                      />
                      <p className="text-xs text-gray-500">Global weight used for single-expert models or as fallback when per-expert weight is not set.</p>
                      <div className="grid grid-cols-2 gap-3">
                        <NumberInput
                          label="Flow Loss Weight (low noise)"
                          className="pt-1"
                          value={jobConfig.config.process[0].train.spectral_flow_weight_low ?? null}
                          onChange={value => setJobConfig(value, 'config.process[0].train.spectral_flow_weight_low')}
                          placeholder="Leave empty for global"
                          docKey={'train.spectral_flow_weight_low'}
                          min={0}
                          step={0.01}
                        />
                        <NumberInput
                          label="Flow Loss Weight (high noise)"
                          className="pt-1"
                          value={jobConfig.config.process[0].train.spectral_flow_weight_high ?? null}
                          onChange={value => setJobConfig(value, 'config.process[0].train.spectral_flow_weight_high')}
                          placeholder="Leave empty for global"
                          docKey={'train.spectral_flow_weight_high'}
                          min={0}
                          step={0.01}
                        />
                      </div>
                      <p className="text-xs text-gray-500">For MoE models: override global weight per expert. Leave empty to use global weight.</p>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <NumberInput
                        label="Flow Max Timestep"
                        className="pt-1"
                        value={jobConfig.config.process[0].train.spectral_flow_max_timestep ?? 800}
                        onChange={value => setJobConfig(value, 'config.process[0].train.spectral_flow_max_timestep')}
                        placeholder="800"
                        docKey={'train.spectral_flow_max_timestep'}
                        min={0}
                        max={1000}
                        step={50}
                      />
                      <div className="flex items-end pb-1">
                        <div className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            id="spectral_flow_motion_weighted"
                            checked={jobConfig.config.process[0].train.spectral_flow_motion_weighted ?? true}
                            onChange={e => setJobConfig(e.target.checked, 'config.process[0].train.spectral_flow_motion_weighted')}
                            className="rounded border-gray-600"
                          />
                          <label htmlFor="spectral_flow_motion_weighted" className="text-sm text-gray-300">
                            Motion Weighted
                          </label>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4 pt-1">
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="spectral_flow_adaptive"
                          checked={jobConfig.config.process[0].train.spectral_flow_adaptive ?? false}
                          onChange={e => setJobConfig(e.target.checked, 'config.process[0].train.spectral_flow_adaptive')}
                          className="rounded border-gray-600"
                        />
                        <label htmlFor="spectral_flow_adaptive" className="text-sm text-gray-300">
                          Adaptive Weighting
                        </label>
                      </div>
                    </div>
                    <NumberInput
                      label="Rejection Threshold"
                      className="pt-1"
                      value={jobConfig.config.process[0].train.spectral_flow_rejection_threshold ?? 5.0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.spectral_flow_rejection_threshold')}
                      placeholder="5.0"
                      docKey={'train.spectral_flow_rejection_threshold'}
                      min={0}
                      step={0.5}
                    />
                    <NumberInput
                      label="Max Rejections (per expert)"
                      className="pt-1"
                      value={jobConfig.config.process[0].train.spectral_flow_max_rejections ?? 100}
                      onChange={value => setJobConfig(value, 'config.process[0].train.spectral_flow_max_rejections')}
                      placeholder="100"
                      docKey={'train.spectral_flow_max_rejections'}
                      min={0}
                      step={10}
                    />
                    <div className="border-t border-blue-900 pt-3 mt-3">
                      <div className="flex items-center space-x-2 mb-2">
                        <input
                          type="checkbox"
                          id="spectral_flow_gradient_projection_enabled"
                          checked={jobConfig.config.process[0].train.spectral_flow_gradient_projection_enabled ?? false}
                          onChange={e => setJobConfig(e.target.checked, 'config.process[0].train.spectral_flow_gradient_projection_enabled')}
                          className="rounded border-gray-600"
                        />
                        <label htmlFor="spectral_flow_gradient_projection_enabled" className="text-sm text-blue-300 font-medium">
                          Gradient Projection (PCGrad-style)
                        </label>
                      </div>
                      {jobConfig.config.process[0].train.spectral_flow_gradient_projection_enabled && (
                        <div className="ml-4 mb-3">
                          <p className="text-xs text-gray-500">
                            Computes spectral and flow gradients separately. When they conflict (one improves while other worsens),
                            projects spectral gradient to remove the conflicting component. This directly nudges optimization
                            toward directions that improve spectral loss without hurting flow loss. More effective than step rejection
                            but adds computational overhead (~2x backward cost).
                          </p>
                        </div>
                      )}
                      {jobConfig.config.process[0].train.loss_type === 'mse_spectral_flow' && (
                        <div className="border-t border-blue-900 pt-3 mt-3">
                          <div className="flex items-center space-x-2 mb-2">
                            <input
                              type="checkbox"
                              id="mse_spectral_flow_gradient_projection_enabled"
                              checked={jobConfig.config.process[0].train.mse_spectral_flow_gradient_projection_enabled ?? false}
                              onChange={e => setJobConfig(e.target.checked, 'config.process[0].train.mse_spectral_flow_gradient_projection_enabled')}
                              className="rounded border-gray-600"
                            />
                            <label htmlFor="mse_spectral_flow_gradient_projection_enabled" className="text-sm text-blue-300 font-medium">
                              Gradient Projection (PCGrad-style, 3-way)
                            </label>
                          </div>
                          {jobConfig.config.process[0].train.mse_spectral_flow_gradient_projection_enabled && (
                            <div className="ml-4 mb-3">
                              <p className="text-xs text-gray-500">
                                Computes MSE, spectral, and flow gradients separately. When any gradients conflict, projects them
                                to remove conflicting components. This allows all three losses to contribute without destructive
                                interference. More effective than step rejection but adds computational overhead (~3x backward cost).
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                      <div className="flex items-center space-x-2 mb-2">
                        <input
                          type="checkbox"
                          id="spectral_flow_loss_rejection_enabled"
                          checked={jobConfig.config.process[0].train.spectral_flow_loss_rejection_enabled ?? false}
                          onChange={e => setJobConfig(e.target.checked, 'config.process[0].train.spectral_flow_loss_rejection_enabled')}
                          className="rounded border-gray-600"
                        />
                        <label htmlFor="spectral_flow_loss_rejection_enabled" className="text-sm text-blue-300 font-medium">
                          Enable Step Loss Rejection (trust region)
                        </label>
                      </div>
                      {jobConfig.config.process[0].train.spectral_flow_loss_rejection_enabled && (
                        <div className="space-y-2 ml-4">
                          <div className="grid grid-cols-2 gap-3">
                            <NumberInput
                              label="Max Loss (low noise)"
                              className="pt-1"
                              value={jobConfig.config.process[0].train.spectral_flow_loss_rejection_max_low ?? 7.0}
                              onChange={value => setJobConfig(value, 'config.process[0].train.spectral_flow_loss_rejection_max_low')}
                              placeholder="7.0"
                              docKey={'train.spectral_flow_loss_rejection_max_low'}
                              min={0}
                              step={0.5}
                            />
                            <NumberInput
                              label="Max Loss (high noise)"
                              className="pt-1"
                              value={jobConfig.config.process[0].train.spectral_flow_loss_rejection_max_high ?? 14.0}
                              onChange={value => setJobConfig(value, 'config.process[0].train.spectral_flow_loss_rejection_max_high')}
                              placeholder="14.0"
                              docKey={'train.spectral_flow_loss_rejection_max_high'}
                              min={0}
                              step={0.5}
                            />
                          </div>
                          <NumberInput
                            label="Max Loss Increase (%)"
                            className="pt-1"
                            value={jobConfig.config.process[0].train.spectral_flow_loss_rejection_max_increase_pct ?? 20.0}
                            onChange={value => setJobConfig(value, 'config.process[0].train.spectral_flow_loss_rejection_max_increase_pct')}
                            placeholder="20.0"
                            docKey={'train.spectral_flow_loss_rejection_max_increase_pct'}
                            min={0}
                            step={5}
                          />
                          <p className="text-xs text-gray-500">
                            Rejects optimizer steps where loss exceeds thresholds or increases too much.
                            Prevents catastrophic loss spikes and training collapse. Rejected steps skip parameter updates.
                          </p>
                        </div>
                      )}
                      <div className="border-t border-blue-900 pt-3 mt-3">
                        <div className="flex items-center space-x-2 mb-2">
                          <input
                            type="checkbox"
                            id="spectral_flow_constraint_rejection_enabled"
                            checked={jobConfig.config.process[0].train.spectral_flow_constraint_rejection_enabled ?? false}
                            onChange={e => setJobConfig(e.target.checked, 'config.process[0].train.spectral_flow_constraint_rejection_enabled')}
                            className="rounded border-gray-600"
                          />
                          <label htmlFor="spectral_flow_constraint_rejection_enabled" className="text-sm text-blue-300 font-medium">
                            Constraint Mode: Prioritize Spectral, Protect Flow
                          </label>
                        </div>
                        {jobConfig.config.process[0].train.spectral_flow_constraint_rejection_enabled && (
                          <div className="space-y-2 ml-4">
                            <NumberInput
                              label="Max Flow Loss Increase (%)"
                              className="pt-1"
                              value={jobConfig.config.process[0].train.spectral_flow_constraint_flow_max_increase_pct ?? 5.0}
                              onChange={value => setJobConfig(value, 'config.process[0].train.spectral_flow_constraint_flow_max_increase_pct')}
                              placeholder="5.0"
                              docKey={'train.spectral_flow_constraint_flow_max_increase_pct'}
                              min={0}
                              step={1}
                            />
                            <div className="flex items-center space-x-2">
                              <input
                                type="checkbox"
                                id="spectral_flow_constraint_spectral_must_decrease"
                                checked={jobConfig.config.process[0].train.spectral_flow_constraint_spectral_must_decrease ?? true}
                                onChange={e => setJobConfig(e.target.checked, 'config.process[0].train.spectral_flow_constraint_spectral_must_decrease')}
                                className="rounded border-gray-600"
                              />
                              <label htmlFor="spectral_flow_constraint_spectral_must_decrease" className="text-xs text-gray-300">
                                Require spectral loss to decrease each step
                              </label>
                            </div>
                            <p className="text-xs text-gray-500">
                              Rejects steps that reduce spectral loss while increasing flow loss beyond threshold.
                              Enforces multi-objective balance — flow loss constrains optimization without dominating.
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                    <p className="text-xs text-gray-500">
                      Flow loss enforces temporal motion consistency by warping predicted frames with ground truth optical flow.
                      Only active at timesteps &lt; Flow Max Timestep (low-noise regime where motion is meaningful).
                      Adaptive weighting dynamically adjusts flow weight based on deviation history.
                      Steps with deviation exceeding rejection threshold have their gradients zeroed.
                    </p>
                  </div>
                )}
                {modelArch?.additionalSections?.includes('train.audio_loss_multiplier') && (
                  <NumberInput
                    label="Audio Loss Multiplier"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.audio_loss_multiplier ?? 1.0}
                    onChange={value => setJobConfig(value, 'config.process[0].train.audio_loss_multiplier')}
                    placeholder="eg. 1.0"
                    docKey={'train.audio_loss_multiplier'}
                    min={0}
                  />
                )}
              </div>
              <div>
                <FormGroup label="EMA (Exponential Moving Average)">
                  <Checkbox
                    label="Use EMA"
                    className="pt-1"
                    checked={jobConfig.config.process[0].train.ema_config?.use_ema || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.use_ema')}
                  />
                </FormGroup>
                {jobConfig.config.process[0].train.ema_config?.use_ema && (
                  <NumberInput
                    label="EMA Decay"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.ema_config?.ema_decay as number}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.ema_decay')}
                    placeholder="eg. 0.99"
                    min={0}
                  />
                )}

                <FormGroup label="Text Encoder Optimizations" className="pt-2">
                  {!disableSections.includes('train.unload_text_encoder') && (
                    <Checkbox
                      label="Unload TE to RAM"
                      disabled={jobConfig.config.process[0].train.cache_text_embeddings}
                      checked={jobConfig.config.process[0].train.unload_text_encoder || false}
                      docKey={'train.unload_text_encoder'}
                      onChange={value => setJobConfig(value, 'config.process[0].train.unload_text_encoder')}
                    />
                  )}
                  <Checkbox
                    label="Cache Text Embeddings"
                    checked={jobConfig.config.process[0].train.cache_text_embeddings || false}
                    docKey={'train.cache_text_embeddings'}
                    onChange={value => setJobConfig(value, 'config.process[0].train.cache_text_embeddings')}
                  />
                </FormGroup>
              </div>
              <div>
                {disableSections.includes('train.diff_output_preservation') ||
                disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <FormGroup label="Regularization">
                    <></>
                  </FormGroup>
                )}
                {disableSections.includes('train.diff_output_preservation') ? null : (
                  <>
                    <Checkbox
                      label="Differential Output Preservation"
                      docKey={'train.diff_output_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.diff_output_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.diff_output_preservation');
                        if (value && jobConfig.config.process[0].train.blank_prompt_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.blank_prompt_preservation');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.diff_output_preservation && (
                      <>
                        <NumberInput
                          label="DOP Loss Multiplier"
                          className="pt-2"
                          value={jobConfig.config.process[0].train.diff_output_preservation_multiplier as number}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                        <TextInput
                          label="DOP Preservation Class"
                          className="pt-2 pb-4"
                          value={jobConfig.config.process[0].train.diff_output_preservation_class as string}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_class')
                          }
                          placeholder="eg. woman"
                        />
                      </>
                    )}
                  </>
                )}
                {disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <>
                    <Checkbox
                      label="Blank Prompt Preservation"
                      docKey={'train.blank_prompt_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.blank_prompt_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.blank_prompt_preservation');
                        if (value && jobConfig.config.process[0].train.diff_output_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.diff_output_preservation');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.blank_prompt_preservation && (
                      <>
                        <NumberInput
                          label="BPP Loss Multiplier"
                          className="pt-2"
                          value={
                            (jobConfig.config.process[0].train.blank_prompt_preservation_multiplier as number) || 1.0
                          }
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.blank_prompt_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                      </>
                    )}
                  </>
                )}
                <FormGroup label="Other" className="pt-2">
                  <>
                    <Checkbox
                      label="Contrastive Guidance Loss"
                      docKey={'train.do_guidance_loss'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.do_guidance_loss || false}
                      onChange={value => {
                        if (value) {
                          setJobConfig(true, 'config.process[0].train.do_guidance_loss');
                          if (!jobConfig.config.process[0].train.guidance_loss_target) {
                            setJobConfig(4.0, 'config.process[0].train.guidance_loss_target');
                          }
                        } else {
                          setJobConfig(undefined, 'config.process[0].train.do_guidance_loss');
                          setJobConfig(undefined, 'config.process[0].train.guidance_loss_target');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.do_guidance_loss && (
                      <>
                        <NumberInput
                          label="Guidance Loss Target"
                          docKey={'train.guidance_loss_target'}
                          value={(jobConfig.config.process[0].train.guidance_loss_target as number) || 4.0}
                          onChange={value => setJobConfig(value, 'config.process[0].train.guidance_loss_target')}
                          placeholder="eg. 3.0"
                          min={0}
                        />
                      </>
                    )}
                  </>
                </FormGroup>
              </div>
            </div>
          </Card>
        </div>
        <div>
          <Card title="Advanced" collapsible>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <Checkbox
                  label="Do Differential Guidance"
                  docKey={'train.do_differential_guidance'}
                  className="pt-1"
                  checked={jobConfig.config.process[0].train.do_differential_guidance || false}
                  onChange={value => {
                    let newValue = value == false ? undefined : value;
                    setJobConfig(newValue, 'config.process[0].train.do_differential_guidance');
                    if (!newValue) {
                      setJobConfig(undefined, 'config.process[0].train.differential_guidance_scale');
                    } else if (
                      jobConfig.config.process[0].train.differential_guidance_scale === undefined ||
                      jobConfig.config.process[0].train.differential_guidance_scale === null
                    ) {
                      // set default differential guidance scale to 3.0
                      setJobConfig(3.0, 'config.process[0].train.differential_guidance_scale');
                    }
                  }}
                />
                {jobConfig.config.process[0].train.differential_guidance_scale && (
                  <>
                    <NumberInput
                      label="Differential Guidance Scale"
                      className="pt-2"
                      value={(jobConfig.config.process[0].train.differential_guidance_scale as number) || 3.0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.differential_guidance_scale')}
                      placeholder="eg. 3.0"
                      min={0}
                    />
                  </>
                )}
              </div>
            </div>
          </Card>
        </div>
        <div>
          <Card title="Datasets">
            <>
              {jobConfig.config.process[0].datasets.map((dataset, i) => (
                <div key={i} className="p-4 rounded-lg bg-gray-800 relative">
                  <div className="absolute top-2 right-2 flex gap-1">
                    <button
                      type="button"
                      onClick={() => {
                        const duplicated = objectCopy(dataset);
                        const datasets = [...jobConfig.config.process[0].datasets];
                        datasets.splice(i + 1, 0, duplicated);
                        setJobConfig(datasets, 'config.process[0].datasets');
                      }}
                      className="bg-gray-700 hover:bg-gray-600 rounded-full p-2 text-sm transition-colors"
                      title="Duplicate Dataset"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          jobConfig.config.process[0].datasets.filter((_, index) => index !== i),
                          'config.process[0].datasets',
                        )
                      }
                      className="bg-red-600 hover:bg-red-700 text-white rounded-full p-2 text-sm transition-colors"
                      title="Remove Dataset"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  <h2 className="text-lg font-bold mb-4">Dataset {i + 1}</h2>
                  <div className={datasetStyleClass}>
                    <div>
                      <SelectInput
                        label="Target Dataset"
                        value={dataset.folder_path}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].folder_path`)}
                        options={datasetOptions}
                      />
                      {modelArch?.additionalSections?.includes('datasets.control_path') && (
                        <SelectInput
                          label="Control Dataset"
                          docKey="datasets.control_path"
                          value={dataset.control_path ?? ''}
                          className="pt-2"
                          onChange={value =>
                            setJobConfig(value == '' ? null : value, `config.process[0].datasets[${i}].control_path`)
                          }
                          options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                        />
                      )}
                      {modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (
                        <>
                          <SelectInput
                            label="Control Dataset 1"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_1 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_1`,
                              )
                            }
                            options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                          />
                          <SelectInput
                            label="Control Dataset 2"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_2 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_2`,
                              )
                            }
                            options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                          />
                          <SelectInput
                            label="Control Dataset 3"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_3 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_3`,
                              )
                            }
                            options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                          />
                        </>
                      )}
                      <NumberInput
                        label="LoRA Weight"
                        value={dataset.network_weight}
                        className="pt-2"
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].network_weight`)}
                        placeholder="eg. 1.0"
                      />
                      <NumberInput
                        label="Num Repeats"
                        value={dataset.num_repeats || 1}
                        className="pt-2"
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].num_repeats`)}
                        placeholder="eg. 1"
                        docKey={'dataset.num_repeats'}
                      />
                    </div>
                    <div>
                      <TextInput
                        label="Default Caption"
                        value={dataset.default_caption}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].default_caption`)}
                        placeholder="eg. A photo of a cat"
                      />
                      <NumberInput
                        label="Caption Dropout Rate"
                        className="pt-2"
                        value={dataset.caption_dropout_rate}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].caption_dropout_rate`)}
                        placeholder="eg. 0.05"
                        min={0}
                        required
                      />
                      <CreatableSelectInput
                        label="Caption Extension"
                        className="pt-2"
                        value={dataset.caption_ext || 'txt'}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].caption_ext`)}
                        options={[
                          { value: 'txt', label: 'txt' },
                          { value: 'json', label: 'json' },
                          { value: 'caption', label: 'caption' },
                        ]}
                      />
                      <SelectInput
                        label="Resize Method"
                        className="pt-2"
                        value={dataset.resize_method || 'lanczos'}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].resize_method`)}
                        options={[
                          { value: 'lanczos', label: 'Lanczos' },
                          { value: 'bicubic', label: 'Bicubic' },
                        ]}
                      />

                      {modelArch?.additionalSections?.includes('datasets.num_frames') && !dataset.auto_frame_count && (
                        <NumberInput
                          label="Num Frames"
                          className="pt-2"
                          docKey="datasets.num_frames"
                          value={dataset.num_frames}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].num_frames`)}
                          placeholder="eg. 41"
                          min={1}
                          required
                        />
                      )}
                    </div>
                    <div>
                      <FormGroup label="Settings" className="">
                        <Checkbox
                          label="Cache latents to disk"
                          checked={dataset.cache_latents_to_disk || false}
                          onChange={value => {
                            setJobConfig(value, `config.process[0].datasets[${i}].cache_latents_to_disk`);
                            if (value) {
                              setJobConfig(false, `config.process[0].datasets[${i}].cache_latents`);
                            }
                          }}
                        />
                        <Checkbox
                          label="Cache latents to RAM"
                          checked={dataset.cache_latents || false}
                          onChange={value => {
                            setJobConfig(value, `config.process[0].datasets[${i}].cache_latents`);
                            if (value) {
                              setJobConfig(false, `config.process[0].datasets[${i}].cache_latents_to_disk`);
                            }
                          }}
                        />
                        <Checkbox
                          label="Is Regularization"
                          checked={dataset.is_reg || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].is_reg`)}
                        />
                        {modelArch?.additionalSections?.includes('datasets.auto_frame_count') && (
                          <Checkbox
                            label="Auto Frame Count"
                            checked={dataset.auto_frame_count || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].auto_frame_count`)}
                            docKey="datasets.auto_frame_count"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.do_i2v') && (
                          <Checkbox
                            label="Do I2V"
                            checked={dataset.do_i2v || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].do_i2v`)}
                            docKey="datasets.do_i2v"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.do_t2v') && (
                          <>
                            <Checkbox
                              label="Do T2V"
                              checked={dataset.do_t2v || false}
                              onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].do_t2v`)}
                              docKey="datasets.do_t2v"
                            />
                            <NumberInput
                              label="Caption Dropout Rate for T2V"
                              className="pt-2"
                              value={dataset.caption_dropout_rate_t2v ?? 0}
                              onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].caption_dropout_rate_t2v`)}
                              placeholder="eg. 0.05 (0 = no dropout for T2V)"
                              min={0}
                              max={1}
                              step={0.01}
                              docKey="datasets.caption_dropout_rate_t2v"
                            />
                          </>
                        )}
                        {modelArch?.additionalSections?.includes('datasets.do_audio') && (
                          <Checkbox
                            label="Do Audio"
                            checked={dataset.do_audio || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].do_audio`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].do_audio`);
                              }
                            }}
                            docKey="datasets.do_audio"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.audio_normalize') && (
                          <Checkbox
                            label="Audio Normalize"
                            checked={dataset.audio_normalize || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].audio_normalize`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].audio_normalize`);
                              }
                            }}
                            docKey="datasets.audio_normalize"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.audio_preserve_pitch') && (
                          <Checkbox
                            label="Audio Preserve Pitch"
                            checked={dataset.audio_preserve_pitch || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].audio_preserve_pitch`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].audio_preserve_pitch`);
                              }
                            }}
                            docKey="datasets.audio_preserve_pitch"
                          />
                        )}
                        {/* Optical flow caching options for video datasets */}
                        {(dataset.num_frames > 1 || dataset.auto_frame_count || jobConfig.config.process[0].train.loss_type === 'spectral_flow' || jobConfig.config.process[0].train.loss_type === 'mse_spectral_flow') && (
                          <div className="mt-2 space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-gray-300">Optical Flow Caching</span>
                              <span className="text-xs text-gray-500">Required for spectral_flow loss</span>
                            </div>
                            <Checkbox
                              label="Cache optical flow to disk"
                              checked={dataset.cache_optical_flow_to_disk || false}
                              onChange={value => {
                                setJobConfig(value, `config.process[0].datasets[${i}].cache_optical_flow_to_disk`);
                              }}
                              docKey="datasets.cache_optical_flow_to_disk"
                            />
                            {dataset.cache_optical_flow_to_disk && (
                              <div className="pl-4 pt-1">
                                <SelectInput
                                  label="Flow Model"
                                  className="pt-1"
                                  value={dataset.optical_flow_model ?? 'sea-raft-m'}
                                  onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].optical_flow_model`)}
                                  docKey="datasets.optical_flow_model"
                                  options={[
                                    { value: 'sea-raft-m', label: 'SEA-RAFT-M (Medium, Recommended)' },
                                    { value: 'sea-raft-s', label: 'SEA-RAFT-S (Small, Faster)' },
                                  ]}
                                />
                                <p className="text-xs text-gray-500 mt-1">
                                  SEA-RAFT-M: Best accuracy/speed balance (~3GB VRAM).
                                  SEA-RAFT-S: Faster but slightly less accurate (~2GB VRAM).
                                  Model downloads on first use.
                                </p>
                              </div>
                            )}
                          </div>
                        )}
                      </FormGroup>
                      {!isAudioModel && (
                        <FormGroup label="Flipping" docKey={'datasets.flip'} className="mt-2">
                          <Checkbox
                            label={
                              <>
                                Flip X <FlipHorizontal2 className="inline-block w-4 h-4 ml-1" />
                              </>
                            }
                            checked={dataset.flip_x || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_x`)}
                          />
                          <Checkbox
                            label={
                              <>
                                Flip Y <FlipVertical2 className="inline-block w-4 h-4 ml-1" />
                              </>
                            }
                            checked={dataset.flip_y || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_y`)}
                          />
                        </FormGroup>
                      )}
                    </div>
                    {!isAudioModel && (
                      <div>
                        <FormGroup label="Resolutions" className="pt-2">
                          <div className="grid grid-cols-2 gap-2">
                            {[
                              [256, 512, 768, 1024],
                              [1280, 1328, 1536, 2048],
                            ].map(resGroup => (
                              <div key={resGroup[0]} className="space-y-2">
                                {resGroup.map(res => (
                                  <Checkbox
                                    key={res}
                                    label={res.toString()}
                                    checked={dataset.resolution.includes(res)}
                                    onChange={value => {
                                      const resolutions = dataset.resolution.includes(res)
                                        ? dataset.resolution.filter(r => r !== res)
                                        : [...dataset.resolution, res];
                                      setJobConfig(resolutions, `config.process[0].datasets[${i}].resolution`);
                                    }}
                                  />
                                ))}
                              </div>
                            ))}
                          </div>
                        </FormGroup>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <button
                type="button"
                onClick={() => {
                  const newDataset = objectCopy(defaultDatasetConfig);
                  // automaticallt add the controls for a new dataset
                  const controls = modelArch?.controls ?? [];
                  newDataset.controls = controls;
                  setJobConfig([...jobConfig.config.process[0].datasets, newDataset], 'config.process[0].datasets');
                }}
                className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                Add Dataset
              </button>
            </>
          </Card>
        </div>
        <div>
          <Card title="Sample">
            <div className={sampleTopStyleClass}>
              <div>
                <NumberInput
                  label="Sample Every"
                  value={jobConfig.config.process[0].sample.sample_every}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_every')}
                  placeholder="eg. 250"
                  min={1}
                  required
                />
                <SelectInput
                  label="Sampler"
                  className="pt-2"
                  value={jobConfig.config.process[0].sample.sampler}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sampler')}
                  options={schedulerOptions}
                />
                <NumberInput
                  label="Guidance Scale"
                  value={jobConfig.config.process[0].sample.guidance_scale}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.guidance_scale')}
                  placeholder="eg. 1.0"
                  className="pt-2"
                  min={0}
                  required
                />
                <NumberInput
                  label="Sample Steps"
                  value={jobConfig.config.process[0].sample.sample_steps}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_steps')}
                  placeholder="eg. 1"
                  className="pt-2"
                  min={1}
                  required
                />
              </div>

              {/* NAG (Negative Attention Guidance) Parameters */}
              <div>
                <div className="text-xs font-semibold text-gray-400 mb-2">NAG (Negative Attention Guidance)</div>
                <NumberInput
                  label="NAG Scale"
                  value={jobConfig.config.process[0].sample.nag_scale ?? 1.0}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.nag_scale')}
                  placeholder="1.0 (disabled)"
                  className="pt-2"
                  min={0}
                  step={0.1}
                />
                <NumberInput
                  label="NAG Alpha"
                  value={jobConfig.config.process[0].sample.nag_alpha ?? 0.5}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.nag_alpha')}
                  placeholder="0.5"
                  className="pt-2"
                  min={0}
                  max={2}
                  step={0.1}
                />
                <NumberInput
                  label="NAG Tau"
                  value={jobConfig.config.process[0].sample.nag_tau ?? 3.5}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.nag_tau')}
                  placeholder="3.5"
                  className="pt-2"
                  min={0}
                  step={0.1}
                />
              </div>

              {!isAudioModel && (
                <div>
                  <NumberInput
                    label="Width"
                    value={jobConfig.config.process[0].sample.width}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.width')}
                    placeholder="eg. 1024"
                    min={0}
                    required
                  />
                  <NumberInput
                    label="Height"
                    value={jobConfig.config.process[0].sample.height}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.height')}
                    placeholder="eg. 1024"
                    className="pt-2"
                    min={0}
                    required
                  />
                  {isVideoModel && (
                    <div>
                      <NumberInput
                        label="Num Frames"
                        value={jobConfig.config.process[0].sample.num_frames}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.num_frames')}
                        placeholder="eg. 0"
                        className="pt-2"
                        min={0}
                        required
                      />
                      <NumberInput
                        label="FPS"
                        value={jobConfig.config.process[0].sample.fps}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.fps')}
                        placeholder="eg. 0"
                        className="pt-2"
                        min={0}
                        required
                      />
                    </div>
                  )}
                </div>
              )}

              <div>
                <NumberInput
                  label="Seed"
                  value={jobConfig.config.process[0].sample.seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.seed')}
                  placeholder="eg. 0"
                  min={0}
                  required
                />
                <Checkbox
                  label="Walk Seed"
                  className="pt-4 pl-2"
                  checked={jobConfig.config.process[0].sample.walk_seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.walk_seed')}
                />
              </div>
              <div>
                <FormGroup label="Advanced Sampling" className="pt-2">
                  <div>
                    <Checkbox
                      label="Skip First Sample"
                      className="pt-4"
                      checked={jobConfig.config.process[0].train.skip_first_sample || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.skip_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Force First Sample"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.force_first_sample || false}
                      docKey={'train.force_first_sample'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.force_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.skip_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Disable Sampling"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.disable_sampling || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.disable_sampling');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                </FormGroup>
              </div>
            </div>
            <div className="pt-2 mb-2 flex items-center justify-between">
              <label className="block text-xs text-gray-300">
                Sample Prompts ({jobConfig.config.process[0].sample.samples.length})
              </label>
              {modelArch?.additionalSections?.includes('ideogram_4_prompt') && (
                <button
                  type="button"
                  disabled={jobConfig.config.process[0].sample.samples.length === 0}
                  onClick={() => {
                    const sampleCfg = jobConfig.config.process[0].sample;
                    const items = sampleCfg.samples
                      .map((s, i) => ({
                        index: i,
                        prompt: s.prompt || '',
                        aspectRatio: toAspectRatio(s.width || sampleCfg.width, s.height || sampleCfg.height),
                      }))
                      .filter(it => it.prompt.trim() !== '');
                    if (items.length === 0) return;
                    openUpsamplePromptsModal(items, (index, newPrompt) => {
                      setJobConfig(newPrompt, `config.process[0].sample.samples[${index}].prompt`);
                    });
                  }}
                  className="px-3 py-1.5 text-sm bg-purple-600 hover:bg-purple-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-md inline-flex items-center gap-2"
                >
                  <Wand2 className="w-4 h-4" />
                  Upsample Prompts
                </button>
              )}
            </div>
            {jobConfig.config.process[0].sample.samples.map((sample, i) => (
              <div key={i} className="rounded-lg pl-4 pr-1 mb-4 bg-gray-950">
                <div className="flex items-center space-x-2">
                  <div className="flex-1">
                    <div className="flex">
                      <div className="flex-1">
                        {modelArch?.sampleTags && taggedSampleArr && modelArchTagSections ? (
                          <>
                            {modelArchTagSections.map((sampleTagSection, sti) => (
                              <div key={sti} className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                                {Object.entries(sampleTagSection).map(([tagKey, tag]) => (
                                  <div key={tagKey} className="mb-2">
                                    {tag.type === 'text' && (
                                      <TextInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                    {tag.type === 'multiline' && (
                                      <TextAreaInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                    {tag.type === 'number' && (
                                      <NumberInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                  </div>
                                ))}
                              </div>
                            ))}
                          </>
                        ) : (
                          <>
                            {modelArch?.hasMultiLinePrompts ? (
                              <TextAreaInput
                                label={`Prompt`}
                                value={sample.prompt}
                                onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].prompt`)}
                                placeholder="Enter prompt"
                                required
                              />
                            ) : (
                              <TextInput
                                label={`Prompt`}
                                value={sample.prompt}
                                onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].prompt`)}
                                placeholder="Enter prompt"
                                required
                              />
                            )}
                          </>
                        )}

                        {modelArch?.additionalSections?.includes('ideogram_4_prompt') && (
                          <div className="mt-2">
                            <button
                              type="button"
                              onClick={() => {
                                const sampleCfg = jobConfig.config.process[0].sample;
                                openPromptBoxEditor({
                                  prompt: sample.prompt || '',
                                  aspectRatio: toAspectRatio(
                                    sample.width || sampleCfg.width,
                                    sample.height || sampleCfg.height,
                                  ),
                                  title: `Prompt #${i + 1}`,
                                  onApply: newPrompt =>
                                    setJobConfig(newPrompt, `config.process[0].sample.samples[${i}].prompt`),
                                });
                              }}
                              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md border border-gray-600 text-gray-300 hover:bg-gray-800 transition-colors"
                            >
                              <SquareDashed className="w-3.5 h-3.5" />
                              Edit caption &amp; boxes
                            </button>
                          </div>
                        )}

                        <div className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                          {!isAudioModel && (
                            <TextInput
                              label={`Width`}
                              value={sample.width ? `${sample.width}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].width;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].width`);
                                  } else {
                                    console.warn('Invalid width value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.width} (default)`}
                            />
                          )}
                          {!isAudioModel && (
                            <TextInput
                              label={`Height`}
                              value={sample.height ? `${sample.height}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].height;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].height`);
                                  } else {
                                    console.warn('Invalid height value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.height} (default)`}
                            />
                          )}
                          <TextInput
                            label={`Seed`}
                            value={sample.seed ? `${sample.seed}` : ''}
                            onChange={value => {
                              // remove any non-numeric characters
                              value = value.replace(/\D/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].seed;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const intValue = parseInt(value);
                                if (!isNaN(intValue)) {
                                  setJobConfig(intValue, `config.process[0].sample.samples[${i}].seed`);
                                } else {
                                  console.warn('Invalid seed value:', value);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.walk_seed ? jobConfig.config.process[0].sample.seed + i : jobConfig.config.process[0].sample.seed} (default)`}
                          />
                          <TextInput
                            label={`LoRA Scale`}
                            value={sample.network_multiplier ? `${sample.network_multiplier}` : ''}
                            onChange={value => {
                              // remove any non-numeric, - or . characters
                              value = value.replace(/[^0-9.-]/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].network_multiplier;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                // set it as a string
                                setJobConfig(value, `config.process[0].sample.samples[${i}].network_multiplier`);
                                return;
                              }
                            }}
                            placeholder={`1.0 (default)`}
                          />
                          <TextInput
                            label={`Negative Prompt`}
                            value={sample.neg ?? ''}
                            onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].neg`)}
                            placeholder={`${jobConfig.config.process[0].sample.neg ?? '(global)'}`}
                          />
                        </div>

                        {/* Per-Sample NAG Parameters */}
                        <div className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                          <NumberInput
                            label={`NAG Scale`}
                            value={sample.nag_scale ?? ''}
                            onChange={value => {
                              if (value === '') {
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].nag_scale;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const numValue = parseFloat(value);
                                if (!isNaN(numValue)) {
                                  setJobConfig(numValue, `config.process[0].sample.samples[${i}].nag_scale`);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.nag_scale ?? '(global)'} (default)`}
                            min={0}
                            step={0.1}
                          />
                          <NumberInput
                            label={`NAG Alpha`}
                            value={sample.nag_alpha ?? ''}
                            onChange={value => {
                              if (value === '') {
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].nag_alpha;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const numValue = parseFloat(value);
                                if (!isNaN(numValue)) {
                                  setJobConfig(numValue, `config.process[0].sample.samples[${i}].nag_alpha`);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.nag_alpha ?? '(global)'} (default)`}
                            min={0}
                            max={2}
                            step={0.1}
                          />
                          <NumberInput
                            label={`NAG Tau`}
                            value={sample.nag_tau ?? ''}
                            onChange={value => {
                              if (value === '') {
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].nag_tau;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const numValue = parseFloat(value);
                                if (!isNaN(numValue)) {
                                  setJobConfig(numValue, `config.process[0].sample.samples[${i}].nag_tau`);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.nag_tau ?? '(global)'} (default)`}
                            min={0}
                            step={0.1}
                          />
                        </div>

                        {/* Per-Sample Video Params (fps, num_frames) */}
                        {isVideoModel && (
                          <div className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                            <TextInput
                              label={`FPS`}
                              value={sample.fps ? `${sample.fps}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].fps;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].fps`);
                                  } else {
                                    console.warn('Invalid fps value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.fps} (default)`}
                            />
                            <TextInput
                              label={`Num Frames`}
                              value={sample.num_frames ? `${sample.num_frames}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].num_frames;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].num_frames`);
                                  } else {
                                    console.warn('Invalid num_frames value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.num_frames} (default)`}
                            />
                          </div>
                        )}
                      </div>
                      {modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (
                        <FormGroup label="Control Images" className="pt-2 ml-4">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2 mt-2">
                            {['ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3'].map((ctrlKey, ctrl_idx) => (
                              <SampleControlImage
                                key={ctrlKey}
                                instruction={`Add Control Image ${ctrl_idx + 1}`}
                                className=""
                                src={sample[ctrlKey as keyof typeof sample] as string}
                                onNewImageSelected={imagePath => {
                                  if (!imagePath) {
                                    let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                                    delete newSamples[i][ctrlKey as keyof typeof sample];
                                    setJobConfig(newSamples, 'config.process[0].sample.samples');
                                  } else {
                                    setJobConfig(imagePath, `config.process[0].sample.samples[${i}].${ctrlKey}`);
                                  }
                                }}
                              />
                            ))}
                          </div>
                        </FormGroup>
                      )}
                      {modelArch?.additionalSections?.includes('sample.ctrl_img') && (
                        <SampleControlImage
                          className="mt-6 ml-4"
                          src={sample.ctrl_img}
                          onNewImageSelected={imagePath => {
                            if (!imagePath) {
                              let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                              delete newSamples[i].ctrl_img;
                              setJobConfig(newSamples, 'config.process[0].sample.samples');
                            } else {
                              setJobConfig(imagePath, `config.process[0].sample.samples[${i}].ctrl_img`);
                            }
                          }}
                        />
                      )}
                    </div>
                    <div className="pb-4"></div>
                  </div>
                  <div>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          jobConfig.config.process[0].sample.samples.filter((_, index) => index !== i),
                          'config.process[0].sample.samples',
                        )
                      }
                      className="rounded-full p-1 text-sm"
                    >
                      <X />
                    </button>
                  </div>
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={() =>
                setJobConfig(
                  [...jobConfig.config.process[0].sample.samples, { prompt: '' }],
                  'config.process[0].sample.samples',
                )
              }
              className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
            >
              Add Prompt
            </button>
          </Card>
        </div>

        {status === 'success' && <p className="text-green-500 text-center">Training saved successfully!</p>}
        {status === 'error' && <p className="text-red-500 text-center">Error saving training. Please try again.</p>}
      </form>
      <AddSingleImageModal />
    </>
  );
}
