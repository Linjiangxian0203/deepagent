---
name: debugging
description: Systematic debugging — reproduce, isolate, fix, verify
---

When debugging any issue, follow this systematic approach:

1. **Reproduce** — verify the bug exists. Write a test that demonstrates it failing.
2. **Isolate** — find the minimal reproduction case. Use binary search to narrow down the cause.
3. **Diagnose** — trace the root cause, not symptoms. Check logs, state, recent changes.
4. **Fix** — make the minimum change that resolves the root cause. Avoid cleanup of unrelated code.
5. **Verify** — run the reproduction test, then the full test suite. Confirm the fix doesn't break other functionality.
