import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TopBar } from '../src/components/TopBar.jsx';
import { Sidebar } from '../src/components/Sidebar.jsx';
import { Field } from '../src/components/Field.jsx';
import { Stepper } from '../src/components/Stepper.jsx';
import { Segmented } from '../src/components/Segmented.jsx';
import { Checkbox } from '../src/components/Checkbox.jsx';
import { ThemeProvider } from '../src/theme/ThemeContext.jsx';

const TABS = [
  { key: 'info', label: 'Info' },
  { key: 'explorer', label: 'Property Explorer' },
];

describe('components', () => {
  it('TopBar renders all tab labels', () => {
    render(
      <ThemeProvider>
        <TopBar tabs={TABS} current="info" onChange={() => {}} onOpenPalette={() => {}} />
      </ThemeProvider>
    );
    expect(screen.getByText('Info')).toBeInTheDocument();
    expect(screen.getByText('Property Explorer')).toBeInTheDocument();
  });

  it('Sidebar renders its children', () => {
    render(
      <Sidebar>
        <div data-testid="x">x</div>
      </Sidebar>
    );
    expect(screen.getByTestId('x')).toBeInTheDocument();
  });

  it('Field shows the label', () => {
    render(
      <Field label="T range">
        <input />
      </Field>
    );
    expect(screen.getByText('T range')).toBeInTheDocument();
  });

  it('Stepper calls onChange with the new value when up clicked', () => {
    let v = 5;
    const set = (x) => (v = x);
    const { rerender } = render(<Stepper value={v} onChange={set} step={1} />);
    fireEvent.click(screen.getByLabelText(/increase/i));
    rerender(<Stepper value={v} onChange={set} step={1} />);
    expect(v).toBe(6);
  });

  it('Segmented highlights the selected option', () => {
    render(
      <Segmented
        options={[
          { value: 'a', label: 'A' },
          { value: 'b', label: 'B' },
        ]}
        value="b"
        onChange={() => {}}
      />
    );
    expect(screen.getByRole('button', { name: 'B' })).toHaveClass('active');
  });

  it('Checkbox toggles on click', () => {
    let v = false;
    const set = (x) => (v = x);
    const { rerender } = render(<Checkbox label="ph" checked={v} onChange={set} />);
    fireEvent.click(screen.getByText('ph'));
    rerender(<Checkbox label="ph" checked={v} onChange={set} />);
    expect(v).toBe(true);
  });
});
