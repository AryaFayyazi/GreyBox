# Grey Box

Pre-execution monitoring for AI agents you can't open: decide whether an agent's next action should go to a human
**before it runs**, from what every deployment can see (instructions, task, tool results, the proposed action).
No access to the model's weights is needed, so it works on closed models.

**Website:** https://aryafayyazi.github.io/GreyBox/

- Train a monitor on your own labelled agent steps, in your browser (nothing is uploaded).
- See it on a closed frontier agent (Claude Haiku 5.5), action by action.
- The evidence: every result comes from a preregistered, hash-locked test, failures included.

The site is built from the project's results by `site/build.py` and published on the `gh-pages` branch.
