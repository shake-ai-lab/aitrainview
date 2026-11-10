import * as THREE from 'three';

export interface Colormap {
  label: string;
  signed: boolean;
  stops: string[];
}

// Evenly spaced sRGB stops, linearly interpolated.
export const COLORMAPS: Record<string, Colormap> = {
  inferno: {
    label: 'Inferno',
    signed: false,
    stops: ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4'],
  },
  viridis: {
    label: 'Viridis',
    signed: false,
    stops: ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
  },
  ice: {
    label: 'Ice',
    signed: false,
    stops: ['#04060c', '#0b1a33', '#123a66', '#1b5f95', '#2f86bd', '#5aaedb', '#9bd3ee', '#e6f6ff'],
  },
  // dark-centred diverging maps: zero is dark, so only large weights / changes glow
  fireice: {
    label: 'Fire / Ice（符号付き）',
    signed: true,
    stops: ['#b8ecff', '#43b3f0', '#1f5fb8', '#142a5c', '#0b0d16', '#5c1a14', '#b8361c', '#f28a2e', '#ffe6a3'],
  },
  rdbu: {
    label: 'RdBu（符号付き）',
    signed: true,
    stops: ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
  },
};

export const CMAP_NAMES = Object.keys(COLORMAPS);

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const lutCache = new Map<string, Uint8Array>();

/** 256 x RGB lookup table (sRGB bytes). */
export function lut(name: string): Uint8Array {
  let t = lutCache.get(name);
  if (t) return t;
  const stops = (COLORMAPS[name] ?? COLORMAPS.inferno).stops.map(hexToRgb);
  t = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(x));
    const f = x - k;
    for (let c = 0; c < 3; c++) t[i * 3 + c] = Math.round(stops[k][c] * (1 - f) + stops[k + 1][c] * f);
  }
  lutCache.set(name, t);
  return t;
}

export function cssGradient(name: string): string {
  const stops = (COLORMAPS[name] ?? COLORMAPS.inferno).stops;
  return `linear-gradient(90deg, ${stops.map((s, i) => `${s} ${((i / (stops.length - 1)) * 100).toFixed(1)}%`).join(', ')})`;
}

/** All colormaps stacked as rows of one RGBA texture for the panel shader. */
export function buildLutTexture(): { texture: THREE.DataTexture; row: (name: string) => number } {
  const W = 256;
  const H = CMAP_NAMES.length;
  const data = new Uint8Array(W * H * 4);
  CMAP_NAMES.forEach((name, r) => {
    const t = lut(name);
    for (let i = 0; i < W; i++) {
      const o = (r * W + i) * 4;
      data[o] = t[i * 3];
      data[o + 1] = t[i * 3 + 1];
      data[o + 2] = t[i * 3 + 2];
      data[o + 3] = 255;
    }
  });
  const texture = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return { texture, row: (name: string) => (Math.max(0, CMAP_NAMES.indexOf(name)) + 0.5) / H };
}
