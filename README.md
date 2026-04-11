# cuda-cache

Caching — LRU eviction, TTL expiration, namespaces, hit/miss stats (Rust)

Part of the Cocapn data layer — structured data handling, indexing, and analytics.

## What It Does

### Key Types

- `CacheEntry` — core data structure
- `CacheStats` — core data structure
- `Cache` — core data structure

## Quick Start

```bash
# Clone
git clone https://github.com/Lucineer/cuda-cache.git
cd cuda-cache

# Build
cargo build

# Run tests
cargo test
```

## Usage

```rust
use cuda_cache::*;

// See src/lib.rs for full API
// 11 unit tests included
```

### Available Implementations

- `CacheEntry` — see source for methods
- `CacheStats` — see source for methods
- `Cache` — see source for methods

## Testing

```bash
cargo test
```

11 unit tests covering core functionality.

## Architecture

This crate is part of the **Cocapn Fleet** — a git-native multi-agent ecosystem.

- **Category**: data
- **Language**: Rust
- **Dependencies**: See `Cargo.toml`
- **Status**: Active development

## Related Crates

- [cuda-pattern-match](https://github.com/Lucineer/cuda-pattern-match)
- [cuda-anomaly](https://github.com/Lucineer/cuda-anomaly)
- [cuda-stats](https://github.com/Lucineer/cuda-stats)
- [cuda-serde](https://github.com/Lucineer/cuda-serde)
- [cuda-schema](https://github.com/Lucineer/cuda-schema)
- [cuda-index](https://github.com/Lucineer/cuda-index)
- [cuda-compression](https://github.com/Lucineer/cuda-compression)
- [cuda-hash](https://github.com/Lucineer/cuda-hash)
- [cuda-sort](https://github.com/Lucineer/cuda-sort)

## Fleet Position

```
Casey (Captain)
├── JetsonClaw1 (Lucineer realm — hardware, low-level systems, fleet infrastructure)
├── Oracle1 (SuperInstance — lighthouse, architecture, consensus)
└── Babel (SuperInstance — multilingual scout)
```

## Contributing

This is a fleet vessel component. Fork it, improve it, push a bottle to `message-in-a-bottle/for-jetsonclaw1/`.

## License

MIT

---

*Built by JetsonClaw1 — part of the Cocapn fleet*
*See [cocapn-fleet-readme](https://github.com/Lucineer/cocapn-fleet-readme) for the full fleet roadmap*
