export interface TensorInfo {
  id: number;
  name: string;
  shape: number[];
  dtype: string;
  numel: number;
  kind: 'matrix' | 'vector' | 'scalar';
  float: boolean;
}

export interface Item {
  t: number;
  label: string;
  kind: 'matrix' | 'vector' | 'scalar';
  module: string;
  attach?: { to: number; how: 'side' | 'below' };
  expert?: number;
}

export interface Group {
  name: string;
  role: 'attn' | 'mlp' | 'other';
  items: Item[];
  experts: Item[][];
}

export interface Block {
  index: number;
  layer_type?: string | null;
  norms: Item[];
  groups: Group[];
}

export interface Tower {
  name: string;
  prefix: string;
  ref_dim: number;
  params: number;
  pre: Item[];
  post: Item[];
  blocks: Block[];
  tied_head: number | null;
}

export interface Structure {
  model_dir: string;
  name: string;
  model_type: string;
  architectures: string[];
  config: Record<string, any>;
  total_params: number;
  tensors: TensorInfo[];
  towers: Tower[];
  main_tower: number;
  dtype_counts: Record<string, number>;
}

export interface RuntimeInfo {
  state: 'unloaded' | 'loading' | 'ready' | 'error';
  busy: string | null;
  error: string | null;
  device?: string;
  dtype?: string;
  vram_used?: number;
  vram_total?: number;
  lora?: { modules: number; r: number; alpha: number; enabled: boolean; steps: number } | null;
}

export interface InferStep {
  step: number;
  token: string;
  token_id: number;
  piece: string;
  text: string;
  top: [string, number, number][];
  act_ids: number[];
  act: number[];
  head_act: number | null;
  resid: number[];
  lens: [string, number][][];
  n_layers: number;
  pos: number;
  tps: number;
}
