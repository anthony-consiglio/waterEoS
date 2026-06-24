import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PlotCard } from '../src/components/PlotCard.jsx';

describe('PlotCard', () => {
  it('renders title and subtitle when provided', () => {
    render(<PlotCard title="My plot" subtitle="some context" />);
    expect(screen.getByRole('heading', { name: 'My plot' })).toBeInTheDocument();
    expect(screen.getByText('some context')).toBeInTheDocument();
  });

  it('renders the loading indicator when loading and no figure', () => {
    render(<PlotCard title="t" loading />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it('renders the error message when error is provided', () => {
    render(<PlotCard title="t" error={new Error('boom')} />);
    expect(screen.getByText(/boom/)).toBeInTheDocument();
  });

  it('does not render the loading indicator when a figure is available', () => {
    render(<PlotCard title="t" figure={{ data: [], layout: {} }} />);
    expect(screen.queryByText(/^loading/i)).not.toBeInTheDocument();
  });
});
