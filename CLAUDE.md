# Repository instructions

## Principle

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.

## Development Rules

### Code Quality

After **every code modification**, run:

```bash
ruff check .
ty check
```

A change is not complete until both checks pass.

If either command reports errors, fix them and re-run **both** checks until they pass.

### Configuration

All configurable parameters **must be provided through `hydra-core` configuration**.

* Do **not** define default values for configurable parameters in application code.
* Do **not** add fallback values when reading configuration.
* Do **not** duplicate configuration values between Python code and Hydra configuration files.
* Required parameters must be explicitly defined in the Hydra configuration.
* Pass configuration values explicitly from Hydra into the components that need them.

Avoid patterns such as:

```python
def create_client(timeout: int = 30):
    ...

timeout = config.get("timeout", 30)
timeout = getattr(config, "timeout", 30)
```

Instead, require the value explicitly:

```python
def create_client(timeout: int):
    ...

create_client(timeout=cfg.client.timeout)
```

`hydra-core` configuration is the **single source of truth** for configurable parameters.
