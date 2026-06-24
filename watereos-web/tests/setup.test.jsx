import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

describe('vitest setup', () => {
  it('renders JSX with RTL + jest-dom', () => {
    render(<h1>hello</h1>);
    expect(screen.getByText('hello')).toBeInTheDocument();
  });

  it('localStorage is cleared between tests', () => {
    expect(localStorage.length).toBe(0);
    localStorage.setItem('x', '1');
    expect(localStorage.getItem('x')).toBe('1');
  });
});
