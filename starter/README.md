# myfer - the reader's engine skeleton

Copy this directory somewhere of your own and work there. It is six empty crates and a notes file, and
it compiles:

```bash
cp -R starter ~/myfer && cd ~/myfer
cargo build        # works offline: there are no dependencies yet
cargo test         # six trivial tests, one per layer
```

The crates are the book's layers, in the order it fills them:

| Crate | Gets your | Filled by |
|---|---|---|
| `myfer-file` | GGUF parsing, tensors, strides, quantized blocks | Chapters 2-5 |
| `myfer-nums` | quantized dots, SIMD, matmul shapes, vector ops, attention | Chapters 6-10 |
| `myfer-graph` | the declarative graph, fusion, backends, allocation, KV | Chapters 14-19 |
| `myfer-engine` | tokenizer, templates, sampler, the generation loop | Chapters 11-13, 20 |
| `myfer-backends` | Metal, then CUDA, behind one registry | Chapters 21-26 |
| `myfer-cli` | the binary, the harness, the server | Chapters 20, 28, 29 |

Dependency direction is one-way by construction: `nums` may use `file`, `graph` may use both, `engine`
may use those three, `backends` may use `file` and `graph`, and `cli` may use everything. If a chapter
makes you want an arrow in the other direction, that is worth a note in `notes.md` before it is worth
a `Cargo.toml` edit.

`notes.md` is the file Part 0 asks for: machines, models, measurements, decisions. The columns are
empty on purpose.
