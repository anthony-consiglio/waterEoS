import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// Global Plotly mock — react-plotly.js never loads in unit tests.
vi.mock('react-plotly.js', () => ({ default: () => null }));

// Polyfill ResizeObserver (Plotly/responsive layouts touch it even when mocked).
if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.dataset.theme = '';
});

afterEach(() => {
  cleanup();
});
