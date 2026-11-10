import type { EventSocket } from './api';
import type { ModelView } from './scene/modelView';
import type { Viewer } from './scene/viewer';
import type { RuntimeInfo, Structure } from './types';

export interface Settings {
  mode: 'rms' | 'mean' | 'sample' | 'absmax';
  cmap: string;
  deltaCmap: string;
  view: 'weights' | 'delta';
  layout: 'flow' | 'tower';
}

/** Shared app state handed to the panels. */
export interface AppCtx {
  sock: EventSocket;
  viewer: Viewer;
  structure: Structure | null;
  view: ModelView | null;
  settings: Settings;
  runtime: RuntimeInfo | null;
  grads: Map<number, number>; // latest gradient norm per tensor
  rel: Map<number, number>; // latest ||ΔW|| / ||W|| per tensor
  acts: Map<number, number>; // latest activation RMS per tensor
  toast(msg: string, level?: 'info' | 'ok' | 'error'): void;
  /** Ask before an action that throws away unsaved training results; true = go ahead. */
  confirmDiscard(action: string): Promise<boolean>;
  /** Training results were undone: forget ΔW / gradients in the view and the panels. */
  clearTrainingView(): void;
  select(tid: number): void;
  setView(v: 'weights' | 'delta'): void;
  setActivityMode(mode: 'none' | 'inference' | 'training'): void;
  showTab(name: 'inspect' | 'infer' | 'train'): void;
}
