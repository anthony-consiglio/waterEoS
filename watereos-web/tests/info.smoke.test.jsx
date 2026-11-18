import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import Info from '../src/screens/Info.jsx';
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

describe('Info screen', () => {
  it('renders the hero heading and metadata-driven models table', async () => {
    vi.spyOn(client, 'fetchMetadata').mockResolvedValue({
      models: [
        {
          key: 'duska2020',
          display_name: 'Duska (2020)',
          is_two_state: true,
          has_phase_diagram: true,
          has_transport: false,
          T_min: 200,
          T_max: 370,
          P_min: 0.1,
          P_max: 200,
          properties: ['rho'],
        },
      ],
      properties: {},
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    render(wrap(<Info />));
    expect(
      screen.getByRole('heading', { name: /thermodynamic|supercooled water/i })
    ).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Duska (2020)')).toBeInTheDocument());
  });

  it('lists every reference with a DOI link, and the package version', async () => {
    vi.spyOn(client, 'fetchMetadata').mockResolvedValue({
      version: '0.7.1',
      models: [
        {
          key: 'caupin2019_kim',
          display_name: 'Caupin (2019, with Kim)',
          is_two_state: true,
          has_phase_diagram: true,
          has_transport: false,
          T_min: 200,
          T_max: 300,
          P_min: -140,
          P_max: 400,
          properties: ['rho'],
          references: ['caupin2019', 'kim2017'],
        },
      ],
      references: {
        caupin2019: {
          short: 'Caupin & Anisimov (2019)',
          citation:
            'F. Caupin and M. A. Anisimov. "Thermodynamics of supercooled and stretched water."',
          doi: '10.1063/1.5100228',
          notes: [
            { text: 'Erratum: J. Chem. Phys. 163, 039902 (2025).', doi: '10.1063/5.0284908' },
          ],
        },
        kim2017: {
          short: 'Kim et al. (2017)',
          citation: 'K. H. Kim et al. Science 358, 1589–1593 (2017).',
          doi: '10.1126/science.aap8269',
          notes: [],
        },
      },
      reference_groups: [
        { title: 'Equations of state for real water', keys: ['caupin2019', 'kim2017'] },
      ],
      properties: {},
      units: { options: {}, defaults: {}, category_labels: {} },
    });
    render(wrap(<Info />));
    await waitFor(() =>
      expect(screen.getByText('Equations of state for real water')).toBeInTheDocument()
    );
    expect(screen.getByText(/Open source · v0\.7\.1/)).toBeInTheDocument();
    // DOI links in the list, including the erratum
    for (const doi of ['10.1063/1.5100228', '10.1063/5.0284908', '10.1126/science.aap8269']) {
      const link = screen.getByRole('link', { name: `doi:${doi}` });
      expect(link).toHaveAttribute('href', `https://doi.org/${doi}`);
    }
    // The models table links each model to its references
    expect(screen.getAllByRole('link', { name: 'Kim et al. (2017)' })[0]).toHaveAttribute(
      'href',
      'https://doi.org/10.1126/science.aap8269'
    );
    expect(screen.getAllByText(/Models: caupin2019_kim/).length).toBe(2);
    // Updated concept cards
    expect(screen.getByText('Liquid–vapor spinodal')).toBeInTheDocument();
    expect(screen.getByText('x = 1/2')).toBeInTheDocument();
  });
});
