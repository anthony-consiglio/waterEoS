import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';

import App from '../src/App.jsx';
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

describe('App shell', () => {
  it('renders TopBar with all 7 tabs', async () => {
    vi.spyOn(client, 'fetchMetadata').mockResolvedValue({
      models: [],
      properties: {},
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    render(wrap(<App />));
    await waitFor(() => expect(screen.getByText('Info')).toBeInTheDocument());
    [
      'Info',
      'Property Explorer',
      'H₂O Phase Diagram',
      'EoS Phase Diagram',
      'Model Comparison',
      'Point Calculator',
      'Settings',
    ].forEach((label) => expect(screen.getByText(label)).toBeInTheDocument());
  });

  it('clicking ⌘K opens the palette', async () => {
    vi.spyOn(client, 'fetchMetadata').mockResolvedValue({
      models: [],
      properties: {},
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    render(wrap(<App />));
    fireEvent.click(screen.getByText(/search models/i));
    expect(screen.getByPlaceholderText(/search/i)).toBeInTheDocument();
  });
});
