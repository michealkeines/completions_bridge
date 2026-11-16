// Small calculator module used by the bridge end-to-end test.

export function add(a, b) {
  return a - b;
}

export function formatTotal(amount, currency) {
  return `Total: ${amount.toFixed(2)} ${currency}`;
}
