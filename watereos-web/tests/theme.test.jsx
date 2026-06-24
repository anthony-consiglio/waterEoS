import { describe, it, expect } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { ThemeProvider, useTheme } from '../src/theme/ThemeContext.jsx';

function Harness() {
  const { theme, toggle } = useTheme();
  return (
    <>
      <span data-testid="t">{theme}</span>
      <button onClick={toggle}>toggle</button>
    </>
  );
}

describe('ThemeContext', () => {
  it('defaults to dark and sets data-theme on <html>', () => {
    render(
      <ThemeProvider>
        <Harness />
      </ThemeProvider>
    );
    expect(screen.getByTestId('t').textContent).toBe('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');
  });

  it('toggle flips theme and persists to localStorage', () => {
    render(
      <ThemeProvider>
        <Harness />
      </ThemeProvider>
    );
    act(() => screen.getByText('toggle').click());
    expect(screen.getByTestId('t').textContent).toBe('light');
    expect(document.documentElement.dataset.theme).toBe('light');
    expect(localStorage.getItem('watereos_theme')).toBe('light');
  });

  it('restores persisted theme on mount', () => {
    localStorage.setItem('watereos_theme', 'light');
    render(
      <ThemeProvider>
        <Harness />
      </ThemeProvider>
    );
    expect(screen.getByTestId('t').textContent).toBe('light');
    expect(document.documentElement.dataset.theme).toBe('light');
  });
});
