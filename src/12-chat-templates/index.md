# Chapter 12 · Chat Templates and Special Tokens

> **Compute tier**: ✅ any laptop | **Status**: stub
> **Build step S12**: A template renderer that turns a list of messages into the exact prompt bytes the model was trained on, and refuses loudly when it cannot.
> **Rule of this book**: Tokenization is Chapter 11. Tool-call parsing of the model's output appears here only as far as the prompt is concerned; the sampler is Chapter 13.

## First, the gist

An instruction-tuned model expects a very specific string: role markers, separators, a generation prefix, and often a reasoning block with its own delimiters. That string is not part of the weights, it is a template, and most models ship it as a **Jinja** program inside the model file. Rendering it faithfully means running a small template language with the model's own variables, which is exactly the kind of component that works on examples and fails on edge cases. Hence this book's rule for it: if the template cannot be rendered, the load fails. A silent fallback to some generic chat format produces a model that answers, badly, and nobody can tell why.

## What you'll be able to do

- Render a chat template from the model file, including the reasoning block and the generation prefix.
- Explain why the tools parameter reaches the prompt in the client's own key order, and what breaks when a map is sorted.
- Apply the refusal rule: name the conditions under which an engine must refuse to load rather than guess a template.

## Lessons

- [ ] 1. The prompt is a program: roles, delimiters, and the generation prefix
- [ ] 2. Jinja in practice: the small subset real templates use, plus the Python string methods they call
- [ ] 3. Reasoning blocks, and the flag that turns them off
- [ ] 4. Tool definitions in the prompt: key order, escaping, and the client's own text
- [ ] 5. The refusal rule: a template that cannot render is an error (payoff: silent fallbacks surface as "the model is dumb")

## Template surfaces

| Surface | What the file provides | What your renderer must guarantee | Your status |
|---|---|---|---|
| system / user / assistant roles |  |  |  |
| reasoning block |  |  |  |
| tool definitions |  |  |  |
| generation prompt |  |  |  |
| override file |  |  |  |

## Project

**Build step S12 · render, then diff.** Capture the reference engine's rendered prompt for a handful of conversations and diff yours byte for byte. Whitespace matters: a trailing newline is a different prompt.

Keep the refusal path: an unrenderable template must fail at load, and the message must name the missing variable.

## Deliverable

- A renderer driven by the model's own template, with a byte-exact fixture test against a reference render
- A refusal for at least one broken template, with the error naming the variable
- A short note on how tool definitions are ordered and escaped

## How you know you passed

- Rendered prompts are byte-identical to the reference for every fixture conversation.
- An overridden template file takes precedence, and the behaviour around it is tested.
- No code path silently substitutes a generic chat format.

## Reading

- code: `src/template.rs`, `src/template/tool_json.rs` at commit `88ffa5d`
- decisions: ADR-0019 (a template that cannot render refuses the load), ADR-0057 (a template file overrides the model's), ADR-0058 (tools reach the prompt as the client wrote them), ADR-0062 (a request message reaches the prompt as written)
- docs: `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md`

## Backfill

- [ ] Question 1: Where does the time go? (compute-bound or bandwidth-bound, and how do I tell?)
- [ ] Question 2: Why are the weights bytes instead of floats? (what quantization buys and costs)
- [ ] Question 3: Where does the boundary between the engine and the device live? (graph, backend seam, dispatch)
- [ ] Question 4: How do I know it is correct, and how do I know it is faster? (identity gate, measurement)
