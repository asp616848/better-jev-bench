"""First-party loaders.

Every loader in here is a worked example of the contributor interface -- there is
no privileged path. Each one is resolved by its `manifest.toml`'s `loader`
pointer exactly the way a third-party package's would be.

Shared conventions, arrived at while writing these eight rather than specified up
front:

* **The instructions text is written once, as a constant, and it is written for a
  model that has never seen the dataset.** `build_primitives_slice.py` in
  ekVachan does the same and it matters more than it looks: the instructions are
  the only place a `choice` item can say what the option vocabulary *means*.
* **State is truncated to a stated character budget** per dataset, never
  silently. Contract and complaint text runs to tens of thousands of characters
  and a benchmark whose items don't fit in a context window measures context
  length.
* **Nothing is rebalanced.** Real label distributions are kept, and skew is
  handled by declaring `chance_mode = "majority"` in the manifest so the
  Intelligence axis adjusts for it (PRD §5.3). Rebalancing would have made
  Civil Comments' `score` task look more impressive and would have measured a
  distribution we invented.
"""
