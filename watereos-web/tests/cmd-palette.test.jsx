import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CmdPalette } from '../src/components/CmdPalette.jsx';

const GROUPS = [
  {
    name: 'Navigate',
    items: [
      { id: 'info', label: 'Info' },
      { id: 'explorer', label: 'Property Explorer' },
    ],
  },
  {
    name: 'Models',
    items: [
      { id: 'model:duska2020', label: 'Duska (2020)' },
      { id: 'model:holten2014', label: 'Holten (2014)' },
    ],
  },
];

describe('CmdPalette', () => {
  it('renders nothing when closed', () => {
    const { container } = render(
      <CmdPalette open={false} onClose={() => {}} groups={GROUPS} onPick={() => {}} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders all groups + items when open', () => {
    render(<CmdPalette open onClose={() => {}} groups={GROUPS} onPick={() => {}} />);
    expect(screen.getByText('Navigate')).toBeInTheDocument();
    expect(screen.getByText('Models')).toBeInTheDocument();
    expect(screen.getByText('Duska (2020)')).toBeInTheDocument();
  });

  it('filters items by the query', () => {
    render(<CmdPalette open onClose={() => {}} groups={GROUPS} onPick={() => {}} />);
    const input = screen.getByPlaceholderText(/search/i);
    fireEvent.change(input, { target: { value: 'holten' } });
    expect(screen.queryByText('Duska (2020)')).not.toBeInTheDocument();
    expect(screen.getByText('Holten (2014)')).toBeInTheDocument();
  });

  it('calls onPick when an item is clicked', () => {
    const onPick = vi.fn();
    render(<CmdPalette open onClose={() => {}} groups={GROUPS} onPick={onPick} />);
    fireEvent.click(screen.getByText('Info'));
    expect(onPick).toHaveBeenCalledWith(expect.objectContaining({ id: 'info' }));
  });

  it('Escape calls onClose', () => {
    const onClose = vi.fn();
    render(<CmdPalette open onClose={onClose} groups={GROUPS} onPick={() => {}} />);
    fireEvent.keyDown(screen.getByPlaceholderText(/search/i), { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });
});
