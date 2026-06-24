import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import H2OPhaseDiagram from '../src/screens/H2OPhaseDiagram.jsx';
import { ThemeProvider } from '../src/theme/ThemeContext.jsx';
import { SettingsProvider } from '../src/settings/SettingsContext.jsx';
import * as client from '../src/api/client.js';

function wrap(ui) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ThemeProvider>
        <SettingsProvider>{ui}</SettingsProvider>
      </ThemeProvider>
    </SWRConfig>
  );
}

describe('H2OPhaseDiagram screen', () => {
  it('renders the projection segmented control and a plot card', async () => {
    vi.spyOn(client, 'fetchH2OPhase').mockResolvedValue({
      figure: { data: [{ type: 'scatter', x: [], y: [] }], layout: {} },
      warnings: [],
    });
    render(wrap(<H2OPhaseDiagram />));
    // sidebar shows the 3 projection options
    expect(screen.getByRole('button', { name: 'T–V' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'T–P' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '3D P–T–V' })).toBeInTheDocument();
    // plot-card renders
    await waitFor(() => expect(document.querySelector('.plot-card')).toBeInTheDocument());
  });
});
