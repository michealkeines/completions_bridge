import { test } from "node:test";
import assert from "node:assert/strict";
import { add, formatTotal } from "../src/calc.js";

test("add sums two numbers", () => {
  assert.equal(add(2, 3), 5);
  assert.equal(add(-1, 1), 0);
});

test("formatTotal shows two decimals", () => {
  assert.equal(formatTotal(12.5, "EUR"), "Total: 12.50 EUR");
});
