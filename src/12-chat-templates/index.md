# Chapter 12 · Chat Templates and Special Tokens

> - **Compute tier**: ✅ any laptop
> - **Status**: draft
> - **Build step S12**: A template renderer that turns a list of messages into the exact prompt bytes the model was trained on, and refuses loudly when it cannot.
> - **Rule of this book**: Tokenization is Chapter 11. Tool-call parsing of the model's output appears here only as far as the prompt is concerned; the sampler is Chapter 13.

## First, the gist

A language model takes a token sequence. A user takes turns with a chat assistant. Something has to translate,
and that something is a **chat template**: a small program, shipped inside the model file, that turns a list of
messages into the exact string the model was trained on - role markers, newlines and all.

The translation is not free. The string carries the model's private markup (`<|im_start|>`, `<|im_end|>`),
often a default system message the user never wrote, and, when tools are involved, a JSON dump of the function
signatures. On the path back the model emits the same markup, which the engine has to recognise. Get any byte
of it wrong and the model still answers - badly, and without complaint.

The template is a **Jinja** program, and the file gives you the program, not a description of it: the GGUF
carries the source text in `tokenizer.chat_template`, and the model card carries the same text in
`tokenizer_config.json`. A renderer therefore needs two things: a Jinja engine, and the parts of Python that
Jinja templates are allowed to reach - string methods, `tojson`, and the truthiness rules - because a Jinja
template is written as if it were Python.

That distinction is where this chapter lives. The template language is small and every renderer implements it.
What differs between renderers is the *runtime underneath* it, and the differences are the kind that do not
raise: a different `tojson` spacing, a different key order, a `title()` that capitalises after an apostrophe.
This chapter measures those, because they are visible in one place only - the prompt's bytes.

Two things are measured here that surprised me, and both are in the tables below. Turning a reasoning model's
thinking **off** makes its prompt six tokens *longer*, not shorter. And the template inside the GGUF this book
measures is **not the same file** as the one on the model card: it differs by two characters, and the two
characters are braces in the instruction text that teaches the model how to emit a tool call.

## What you'll be able to do

- Render a chat template from the model file, with the generation prefix, the default system message, and the
  tools block, and measure what each of those costs in tokens.
- Reproduce the Python semantics a template depends on, and name the ones that bite (`split(None, 1)`,
  `title()`, `tojson` spacing and key order).
- Explain what "turn thinking off" does to the prompt, with the token count that proves it.
- Apply the refusal rule: name the conditions under which an engine must refuse to load rather than guess a
  template.

## Lessons

- [ ] 1. The prompt is a program: roles, delimiters, and the generation prefix
- [ ] 2. Jinja in practice: the small subset real templates use, plus the Python string methods they call
- [ ] 3. Reasoning blocks, and the flag that turns them off
- [ ] 4. Tool definitions in the prompt: key order, escaping, and the client's own text
- [ ] 5. The refusal rule: a template that cannot render is an error

### Lesson 1 · The prompt is a program: what it consumes, and what it emits

Here is the whole template this chapter measures, as a program rather than as prose:

<!-- facts: template.tmpl-program -->
| Quantity | Value |
|---|---|
| Template length | 2,509 characters, 2,509 bytes, 55 lines |
| `{{ ... }}` expressions | 25 |
| `{% ... %}` statements | 30 |
| Variables it reads | `add_generation_prompt`, `messages`, `tools` |
| Filters it calls | `tojson` |
| Roles it branches on | system, user, assistant, tool |
| sha256 | `d5495a1e5db06111...` |
<!-- /facts -->

Twenty-five expressions and thirty statements, three variables, four roles, one filter. That is a real program,
and it is small enough to read: the full text is 55 lines. What it consumes is a list of messages; what it emits
is one string.

Three of the four are *read* by this template - `messages`, `tools` and `add_generation_prompt` - and
`bos_token` is exposed for the templates that want it at the front of the prompt. That set is the interface of
the template language, and it is the same in every engine that renders these files:

| Variable | Meaning |
|---|---|
| `messages` | the conversation, each with `role` and `content`, plus `tool_calls` on assistant turns |
| `tools` | the function definitions, or `None` when the request carries none |
| `add_generation_prompt` | whether to append the assistant header the model must continue from |
| `bos_token` | the begin-of-sequence token, when the model wants it at the front |

The engine's render call is the whole interface in one place:

<!-- code: src/template.rs::render_at@88ffa5d -->
```rust
/// The one render core: `template` (`None` = the ChatML fallback) plus the
/// `tools` context value, as of one fixed instant.
///
/// `messages` are the client's own JSON objects (#613) — never rebuilt — so a
/// template that echoes a message (`message | tojson`) sees the client's key order
/// and every key it sent. `serde_json::Map` would sort that order away; the
/// carrier does not.
fn render_at(
    now: CivilTime,
    template: Option<&str>,
    messages: &[ToolJson],
    tools: Option<&[ToolJson]>,
    add_generation_prompt: bool,
    bos_token: &str,
) -> Result<String, TemplateError> {
    let Some(template) = template else {
        // The fallback has no tool or message format; `render_checked_at` is what
        // refuses a request that brought `tools` here.
        return Ok(fallback_chatml_messages(messages, add_generation_prompt));
    };
    let mut env = environment_at(now);
    if let Err(e) = env.add_template("chat", template) {
        return Err(TemplateError::classify(&e));
    }
    let tmpl = env
        .get_template("chat")
        .map_err(|e| TemplateError::classify(&e))?;

    // Serialized as they are: `Value::from_serialize` walks `ToolJson`'s entries in
    // document order (minijinja's `preserve_order`), which is the order the client
    // wrote.
    let msgs_value = Value::from_serialize(messages);

    // transformers passes `tools=None`; `none` keeps `{% if tools %}` and
    // `tools is defined` behaving the way jinja2 does.
    let tools_value = match tools {
        Some(tools) => Value::from_serialize(tools),
        None => Value::from(()),
    };

    tmpl.render(context! {
        messages => msgs_value,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_token,
        tools => tools_value,
    })
    .map_err(|e| TemplateError::classify(&e))
}
```
<!-- /code -->

Two details in that function are worth stopping on, because both are silent when wrong. `Value::from_serialize`
walks the entries **in document order** - the order the client wrote the keys in, which the JSON spec does not
require anyone to preserve and which every generic serializer is free to reorder. And `tools` is set to a
non-value (`Value::from(())`) rather than to an empty list, because the template tests `{% if tools %}` and
`tools is defined`: an empty list and a missing variable take different branches in Python, and the template was
written against Python.

Now the output. Take the shortest possible request - one user message, six words long:

<!-- facts: template.tmpl-anatomy -->
178 bytes, 36 tokens. The segments below are cut at the special
tokens and add up to the whole prompt:

| At byte | Segment | Bytes | Tokens |
|---|---|---|---|
| 0 | `<\|im_start\|>` | 12 | 1 |
| 12 | `system\nYou are Qwen, created by Alibaba Cloud. You ...` | 75 | 18 |
| 87 | `<\|im_end\|>` | 10 | 1 |
| 97 | `\n` | 1 | 1 |
| 98 | `<\|im_start\|>` | 12 | 1 |
| 110 | `user\nWhat is the capital of France?` | 35 | 9 |
| 145 | `<\|im_end\|>` | 10 | 1 |
| 155 | `\n` | 1 | 1 |
| 156 | `<\|im_start\|>` | 12 | 1 |
| 168 | `assistant\n` | 10 | 2 |

Sum of the segments: **36** tokens against **36** for the
whole prompt - the cut is at the special tokens, and a special token never merges with its
neighbours, so the counts add up.
<!-- /facts -->

Thirty-six tokens to ask a nine-token question. The user's own words are the `user` segment and nothing else;
everything around it came from the template. The largest single piece is not a marker at all: it is the
**default system message** the template inserts when the conversation has no system turn - 18 tokens, one
sentence nobody typed, and the model reads it as an instruction.

<figure class="book-figure">{{#include ../figures/12-prompt-anatomy.svg}}</figure>

*Figure 12.1 - the prompt as a byte strip, measured. Each block is a segment cut at a special token; the widths are the segment's bytes and the labels carry its token count. The user's question is the light block in the middle: nine of the thirty-six tokens. The default system sentence is the largest block in the prompt.*

The same conversation with one knob moved, which is where the arithmetic of a context window comes from:

<!-- facts: template.tmpl-surfaces -->
| Variant | Bytes | Tokens | Ends with |
|---|---|---|---|
| no system message, prompt off | 156 | 33 | `hat is the capital of France?<\|im_end\|>\n` |
| no system message, prompt on | 178 | 36 | `France?<\|im_end\|>\n<\|im_start\|>assistant\n` |
| explicit system, prompt off | 116 | 23 | `hat is the capital of France?<\|im_end\|>\n` |
| explicit system, prompt on | 138 | 26 | `France?<\|im_end\|>\n<\|im_start\|>assistant\n` |
| one tool, generation prompt | 689 | 154 | `France?<\|im_end\|>\n<\|im_start\|>assistant\n` |
| one tool, explicit system | 649 | 144 | `France?<\|im_end\|>\n<\|im_start\|>assistant\n` |
<!-- /facts -->

Read the first two rows together: the generation prompt (the `<|im_start|>assistant` block the model continues
from) is **22 bytes and 3 tokens**, and without it the model's answer continues the *user* turn instead of
starting an assistant one - a failure that produces coherent, wrong output. Rows three and four repeat the pair
with an explicit system message, which is 40 bytes and 10 tokens *shorter* than the built-in default: writing
your own system prompt is cheaper than accepting the model's.

### Lesson 2 · Jinja in practice: the subset, and the Python underneath it

The template language is Jinja2, and the subset these files use is small enough to list from the source text
above: `if`/`elif`/`else`, `for` with `loop.first`/`loop.index0`/`loop.last`, `set`, `is defined`, string
concatenation, and one filter. There is no inheritance, no macro, no include, and no whitespace control beyond
the `{%-` and `-}}` markers the files use everywhere.

If your language has a Jinja library, use it. Writing a template engine is a month of work for a component
nobody will thank you for. The engine this book studies uses `minijinja`, and the interesting part is everything
it had to add *around* that library, which is exactly the part a template test suite will catch:

<!-- code: src/template.rs::SUPPORTED_STR_METHODS@88ffa5d -->
```rust
/// The Python `str` methods the unknown-method hook implements. Listed in the
/// refusal message so an unsupported template tells the reader what *is*
/// available.
const SUPPORTED_STR_METHODS: &[&str] = &[
    "capitalize",
    "count",
    "endswith",
    "find",
    "join",
    "lower",
    "ljust",
    "lstrip",
    "replace",
    "rfind",
    "rjust",
    "rsplit",
    "rstrip",
    "split",
    "splitlines",
    "startswith",
    "strip",
    "title",
    "upper",
];
```
<!-- /code -->

Nineteen string methods, because Jinja templates call Python methods on Python strings, and a Jinja engine
written for another language does not necessarily have them. The list is not a wish list - it is the set the
templates in this engine's fixture corpus actually reach, and every one of them is defined by CPython, not by
Jinja. Two examples show why that matters:

<!-- code: src/template.rs::py_title@88ffa5d -->
```rust
/// CPython `str.title`: a letter is upper-cased when it follows a non-cased
/// character, lower-cased otherwise.
fn py_title(recv: &str) -> String {
    let mut out = String::with_capacity(recv.len());
    let mut prev_cased = false;
    for c in recv.chars() {
        if prev_cased {
            out.extend(c.to_lowercase());
        } else {
            out.extend(c.to_uppercase());
        }
        prev_cased = c.is_alphanumeric();
    }
    out
}
```
<!-- /code -->

`title()` is not `capitalize()` per word: CPython upper-cases a letter when it follows a *non-cased* character,
and an apostrophe is a non-cased character. So the greeting `don't stop` becomes `Don'T Stop`, not `Don't Stop`.
Any renderer that implements "capitalise the first letter of each word" produces a different string, in a place
nobody thinks to look - a system prompt assembled from user text.

<!-- code: src/template.rs::py_split@88ffa5d -->
```rust
/// CPython `str.split` / `str.rsplit`.
fn py_split(recv: &str, args: &[PyArg], from_right: bool) -> Result<PyValue, String> {
    let sep = match args.first() {
        None | Some(PyArg::None_) => None,
        Some(PyArg::Str(s)) => Some(s.as_str()),
        Some(other) => return Err(format!("separator must be a string, got {other:?}")),
    };
    let maxsplit = match args.get(1) {
        None => -1,
        Some(PyArg::Int(i)) => *i,
        Some(other) => return Err(format!("maxsplit must be an integer, got {other:?}")),
    };
    let Some(sep) = sep else {
        // Whitespace mode: runs of whitespace separate, no empty elements, and
        // leading/trailing whitespace is ignored.
        // ...
    }
```
<!-- /code -->

The first argument of `split` decides which of two algorithms runs, and `None` selects the other one: whitespace
mode, where runs of whitespace separate, no empty strings are returned, and leading and trailing whitespace is
dropped. `split(",")` on `"a,,b"` returns an empty string in the middle; `split()` never returns one. Templates
use both, and the Qwen3 reasoning template uses `split(None, 1)` to separate a model's reasoning from its answer.

These are not corner cases invented for the chapter. They are the shapes CPython gives, verified by running
CPython:

<!-- facts: template.tmpl-pystr -->
| Expression | Python returns |
|---|---|
| `"don't stop".title()` | `"Don'T Stop"` |
| `"  a  b  ".split(None, 1)` | `['a', 'b  ']` |
| `"a,b,,c".split(",")` | `['a', 'b', '', 'c']` |
| `"x".ljust(4, "-")` | `'x---'` |
| `"a\nb\n".splitlines(True)` | `['a\n', 'b\n']` |
| `"  x  ".rstrip()` | `'  x'` |
| `"</think>abc".split("</think>")[-1].lstrip("\n")` | `'abc'` |
<!-- /facts -->

That table is the specification a renderer has to meet, and it is one line of Python to check any of it. A
renderer that passes the model's own template on a happy conversation can still fail on a title-cased name, a
`split` with a separator that does not appear, or a `rstrip` that was given no argument.

### Lesson 3 · Reasoning blocks, and the flag that turns them off

A reasoning model - Qwen3 and its relatives - was trained to think before it answers: it emits a block of
deliberation inside `<think>` and `</think>` markers, then the answer. The template handles both directions.

**Going in**, a replayed assistant turn that already contains `</think>` is split apart, so the model sees the
structure it was trained on rather than one opaque content string: the text before the marker becomes the
reasoning block, the text after becomes the answer, and the template re-emits both with the markers in place.

**Coming out**, the interesting part is the flag. The Qwen3 template has an `enable_thinking` variable, and this
is what it does with it:

```
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```

Turning thinking off does not remove the block. It emits an **empty** one - because the model was trained to
always see a think block, and a prompt with no block at all is a prompt the model has never met. Measured on one
user turn:

<!-- facts: template.tmpl-thinking -->
The template is 4,168 characters; one user turn, generation prompt on:

| Variant | Characters | Tokens | Ends with |
|---|---|---|---|
| thinking on (default) | 81 | 16 | ` in strawberry?<\|im_end\|>\n<\|im_start\|>assistant\n` |
| thinking off | 100 | 22 | `_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
<!-- /facts -->

Read the token column twice. Thinking off is **6 tokens longer** than thinking on: 100 characters against 81,
because the empty block costs more than its absence. Anyone budgeting a context window from intuition ("disable
thinking and I get my tokens back") has the sign backwards. What thinking off saves is not prompt tokens at all;
it is the *output* tokens the model would have spent deliberating.

Nothing about that is inferable from a flag name. It is in the template, and the template is in the file.

The engine side of this lesson is not the rendering - it is knowing that the reasoning branch exists at load
time, before any request. That is what its load-time canary is for:

<!-- code: src/template.rs::validate@88ffa5d -->
```rust
/// Compile the template and render a canary conversation through it.
///
/// This is the load-time gate: the CLI and `serve` call it once so an
/// unrenderable template is refused *before* any request, with the construct
/// named. The canary includes a `<think>` assistant turn because Qwen3's
/// `str.split`/`lstrip`/`rstrip` calls only execute on that branch.
pub fn validate(template: &str) -> Result<(), TemplateError> {
    let canary: Vec<(String, Option<String>)> = vec![
        ("system".into(), Some("You are a helpful assistant.".into())),
        ("user".into(), Some("hi".into())),
        (
            "assistant".into(),
            Some("<think>\nchecking\n</think>\n\nhello".into()),
        ),
        ("user".into(), Some("again".into())),
    ];
    render_messages(template, &canary, true, "")?;
    // Qwen3's `str.split`/`lstrip`/`rstrip` only execute when an assistant turn
    // is replayed *after* the last user query, so the canary covers that too.
    let replay: Vec<(String, Option<String>)> = vec![
        ("user".into(), Some("hi".into())),
        (
            "assistant".into(),
            Some("<think>\nchecking\n</think>\n\nhello".into()),
        ),
    ];
    render_messages(template, &replay, false, "").map(|_| ())
}
```
<!-- /code -->

The canary is a conversation, not a syntax check, and its shape is chosen for a reason the comment states: the
template's `str.split`/`lstrip`/`rstrip` calls live on the *assistant replay* branch, so a canary that only
walks a user turn never executes them. A template with a broken reasoning branch would compile, pass a
user-only canary, and fail on the first real conversation that replayed an assistant turn - which is every
second request in a chat session.

### Lesson 4 · Tool definitions in the prompt: key order, escaping, and the client's own text

Tools reach the model as text. The template dumps each definition with `tojson` and wraps the result in
`<tools>` tags, and the model is expected to answer with a JSON object inside `<tool_call>` tags. Two
properties of that dump are decided by the *filter*, not by the template:

- **Key order** - whatever the renderer's serializer does with a mapping's order.
- **Spacing and escaping** - whether `{"a": 1}` or `{"a":1}`, whether `<` stays `<` or becomes `\u003c`.

Neither is visible in the template's source; both are visible in the prompt's bytes. Four policies on one tool
definition:

<!-- facts: template.tmpl-tojson -->
| Policy | Characters | Tokens | Output starts |
|---|---|---|---|
| reference | 307 | 86 | `{"type": "function", "function": {"name": "get_weather", "descri` |
| jinja2-default | 307 | 85 | `{"function": {"description": "Get the current weather for a city` |
| compact | 284 | 64 | `{"type":"function","function":{"name":"get_weather","description` |
| ascii | 307 | 86 | `{"type": "function", "function": {"name": "get_weather", "descri` |
<!-- /facts -->

The first line is what the reference implementation produces, and the third is what a serializer's compact mode
produces by default. They differ by **23 characters and 22 tokens** on one small tool definition - and the
reference renderer is the one the model was trained against. Note the second line: same 307 characters as the
first, one token *fewer*, because sorting the keys puts different characters next to each other and the merge
table joins them differently. Same length, different ids: a byte-count test cannot see this and an id test can.

The engine reproduces CPython's separators explicitly, with a comment naming the failure:

<!-- code: src/template.rs::PyJsonFormatter@88ffa5d -->
```rust
/// Python's `json.dumps` separators for `indent=None`: `", "` and `": "`.
///
/// serde_json's compact form uses `,`/`:` with no spaces, which is not what
/// CPython emits, so every `| tojson` in a template would drift by a space per
/// separator (#597). Only the three separators differ; the escaping, number and
/// string handling are serde_json's, and those match CPython for the value forms
/// a tool definition uses.
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }
}
```
<!-- /code -->

The doc comment is the whole lesson: a serializer's compact form uses `,` and `:`; CPython writes `", "` and
`": "`; every `| tojson` in a template therefore drifts by one space per separator, in a prompt the model reads.
And every Jinja implementation ships a built-in `tojson` that is not the reference, in its own way: Python's
Jinja2 sorts the keys and escapes `<`, `>`, `&` and `'` (the row above), while the engine's own library makes it
compact *and* HTML-escaping, writing `\u003c` where the reference writes `<`. Three implementations, three
different strings from one filter name - which is why the engine replaces the filter rather than configuring it.
Every one of those divergences produces a prompt that renders, loads, and answers.

**Now the thing that should worry you.** The fixture corpus that witnessed this chapter is 12 conversations, and
rendering them here with the reference renderer's own template reproduces every one of them byte for byte:

<!-- facts: template.tmpl-fixtures -->
| Case | Tools | Bytes | Tokens | Reference | Jinja2 `tojson` | GGUF copy |
|---|---|---|---|---|---|---|
| single_user | 0 | 178 | 36 | same | same | same |
| system_and_user | 0 | 133 | 24 | same | same | same |
| multi_turn | 0 | 205 | 44 | same | same | same |
| multi_turn_no_generation_prompt | 0 | 167 | 32 | same | same | same |
| unicode_and_newlines | 0 | 225 | 52 | same | same | same |
| reasoning_assistant_replay | 0 | 304 | 83 | same | same | same |
| reasoning_assistant_turn | 0 | 362 | 94 | same | same | same |
| single_function | 1 | 818 | 190 | same | **differs** | **differs** |
| two_functions_nested_schema | 2 | 1,508 | 393 | same | **differs** | **differs** |
| no_system_message_no_generation_prompt | 1 | 831 | 196 | same | **differs** | **differs** |
| assistant_tool_call_replay | 1 | 1,064 | 250 | same | **differs** | **differs** |
| tool_response_replay | 1 | 991 | 233 | same | **differs** | **differs** |

Rendered with the model card's copy: **12 of 12** byte-identical to the
reference renderer. With Jinja2's own `tojson`: **7 of 12**. With the
copy inside the GGUF: **7 of 12**.
<!-- /facts -->

Twelve of twelve, and the interesting column is the last two. The five cases that pass tools are exactly the
five that differ under a different `tojson` **and** under the GGUF's own copy of the template. Two independent
causes, the same five rows, and neither cause touches the seven cases without tools: the fixture corpus is a net
with a hole in exactly the shape of the feature most likely to break.

That last column needs explaining, because it is not a renderer bug. The template is shipped twice, and the two
copies are not identical:

<!-- facts: template.tmpl-copies -->
| Copy | Characters | sha256 |
|---|---|---|
| `tokenizer_config.json` (the model card) | 2,507 | `cd8e9439f0570856...` |
| `tokenizer.chat_template` in the GGUF | 2,509 | `d5495a1e5db06111...` |

Identical: **no**. 2 edit(s).

| At character | Removed | Inserted |
|---|---|---|
| 687 | `-` | `{` |
| 747 | `-` | `}` |

The model-card copy reads ` XML tags:\n<tool_call>\n{\"name\": <function-name>, \"argum`;
the GGUF copy reads ` XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"argume`.
<!-- /facts -->

Two characters. One `{` inserted at character 687 and one `}` at 747, and both sit inside the instruction
sentence that tells the model how to write a tool call: the model card says

```
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

and the GGUF this book measures says

```
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
```

Both render, both load, neither raises. Whichever copy your engine reads, it teaches the model a slightly
different output format from the one in the other file, and a model that emits `{{...}}` because the system
prompt showed it doubled braces produces tool calls your parser will reject - for reasons that live in a
quantizer's copy step, not in your code.

Two things keep this from being a one-file curiosity. The engine's own fixture records both hashes - the model
card's and the GGUF's - and the GGUF hash it recorded is this file's hash, so the doubling is in the distribution
rather than in one download. And the fix, whatever produced it, is not something an engine can apply: it does not
know which copy the model was trained against.

This is why the chapter's rule is "quote the file you actually read, and record its hash". Recording both makes
the disagreement visible; recording one makes it invisible in whichever direction you did not look.

### Lesson 5 · The refusal rule: a template that cannot render is an error

Every failure in this chapter so far has been silent: a reordered key, a compact JSON, a doubled brace. The
engine's answer to the class is a rule that sounds blunt until you have debugged one of them - **a template that
cannot render refuses the load**, and it never falls back to a generic format.

The reason is asymmetry. A wrong template produces a model that answers, badly; the user concludes the model is
dumb; nothing in the logs says otherwise. A refusal produces a message naming the construct and the line. The
first failure costs a day, the second costs a minute.

What "cannot render" means in practice depends on the renderer's defaults, and the defaults are the danger:

<!-- facts: template.tmpl-refusal -->
| Template | Jinja2 default | Jinja2 strict |
|---|---|---|
| `{{ nosuchvar }}` | renders '' | raises UndefinedError |
| `{{ 'x' \| nosuchfilter }}` | raises TemplateAssertionError | raises TemplateAssertionError |
| `{% if tools %}x` | raises TemplateSyntaxError | raises TemplateSyntaxError |
| `{{ messages[0]['content'] }}` | raises UndefinedError | raises UndefinedError |
| `{{ 'x'.nosuchmethod() }}` | raises UndefinedError | raises UndefinedError |
<!-- /facts -->

The first row is the one to internalise. Jinja2's default behaviour for an undefined variable is to render it as
the **empty string**. A template that reads `{{ system_prompt }}` because someone edited it to take a variable
the engine does not pass will not fail - it will render a prompt with a hole in it, and the model will answer
without the instruction. Only `StrictUndefined` (or the equivalent) turns that into an error. The other four
cases raise in both modes, for four different reasons that no undefined-behaviour setting can swallow: an unknown
filter and an unclosed block fail at compile time, calling a method that does not exist fails at lookup, and
indexing into nothing fails because Python cannot subscript a value that is not there.

The engine's resolver is a single function, and its contract is the rule in one sentence:

<!-- code: src/template.rs::resolve_template@88ffa5d -->
```rust
/// The single place that decides which template a run uses.
///
/// A `--chat-template <FILE>` override (its text, already loaded and validated by
/// `load_template_file`) wins over the GGUF's copy; whichever is chosen is
/// re-validated here, so no caller can skip the check. `Ok(None)` is the
/// documented ChatML case (no template anywhere) — the caller announces it, it is
/// never silent.
pub fn resolve_template(
    embedded: Option<&str>,
    override_text: Option<&str>,
) -> Result<Option<String>, TemplateError> {
    let chosen = match override_text {
        Some(text) => Some(text.to_string()),
        None => embedded.map(str::to_string),
    };
    if let Some(text) = chosen.as_deref() {
        validate(text)?;
    }
    Ok(chosen)
}
```
<!-- /code -->

Three properties, each doing work:

1. **One place decides.** The override file wins over the embedded copy, and no caller can pick a third thing.
2. **Whichever wins is validated.** A user's `--chat-template` is run through the same canary as the model's
   own, so an override cannot smuggle in a template the engine never tested.
3. **`None` is a documented outcome, not a silent one.** A file with no template at all is a real case (older
   GGUFs, base models), and the resolver says so - `Ok(None)` - so the caller has to decide what to print.

That third property is paired with a constant, so four different entry points cannot word the notice
differently:

<!-- code: src/template.rs::CHATML_NOTICE@88ffa5d -->
```rust
/// The one-line notice a caller prints when a run has no template at all.
///
/// One constant, so the four entry points cannot word it differently, and a test
/// can pin the text (#602).
pub const CHATML_NOTICE: &str = "Notice: this GGUF has no tokenizer.chat_template; using the \
                                 generic ChatML renderer";
```
<!-- /code -->

Announcing a fallback is not the same as preferring one. The distinction is what separates "the engine told me
it was using a generic ChatML renderer" from "the engine silently used a generic ChatML renderer and I found out
by reading its source".

## Template surfaces

| Surface | What the file provides | What your renderer must guarantee | Status |
|---|---|---|---|
| system / user / assistant roles | role branches in the template, with `<\|im_start\|>` / `<\|im_end\|>` delimiters | each role's marker and trailing newline reproduced exactly | measured: 36 tokens for one turn, 18 of them the default system sentence |
| reasoning block | `<think>` handling in the template, plus an `enable_thinking` flag | the flag's *actual* effect, which is an empty block, not its absence | measured: 16 tokens on, 22 off |
| tool definitions | a `<tools>` block, one `tojson` per function | key order as the client wrote it, CPython separators, no HTML escaping | measured: 307 characters / 86 tokens under the reference filter, 284 / 64 compact |
| generation prompt | `add_generation_prompt` appends `<\|im_start\|>assistant` | the assistant header present, and only when asked | measured: 22 bytes / 3 tokens, and its absence changes whose turn continues |
| override file | nothing - an engine feature | override wins, is validated, and `None` is announced | engine rule: one resolver, one notice constant |

## Project

**Build step S12 · render, then diff.** Take the template out of your model file, render a handful of
conversations, and diff your output against the reference engine's, byte for byte. Include the four surfaces
above: a turn without a system message, a turn with one, a request with a tool, and a reasoning replay. Add the
two cases that broke this chapter's own fixtures: a tool whose keys are not in alphabetical order, and a
conversation long enough that the context window matters.

Keep the refusal path. An unrenderable template must fail at load, and the message must name the construct and
the line.

## Deliverable

- A renderer driven by the model's own template, with a byte-exact fixture test against a reference render
- A refusal for at least one broken template, with the error naming the variable
- A short note on how tool definitions are ordered and escaped, with the two `tojson` policies you rejected and
  the bytes that made you reject them

## How you know you passed

- Rendered prompts are byte-identical to the reference for every fixture conversation - and at least one of
  your fixtures carries tools, or the `tojson` branch is untested.
- Your renderer reproduces `"don't stop".title()` and `"  a  b  ".split(None, 1)` as CPython does, checked by
  running CPython.
- An overridden template file takes precedence, and an undefined variable is an error rather than an empty
  string.
- No code path silently substitutes a generic chat format.

## Reading

- code: `src/template.rs`, `src/template/tool_json.rs` at the book's [pinned baseline](../appendix/baseline.md)
- decisions: ADR-0019 (a template that cannot render refuses the load), ADR-0057 (a template file overrides the
  model's), ADR-0058 (tools reach the prompt as the client wrote them), ADR-0062 (a request message reaches the
  prompt as written)
- docs: `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md`
- [jinja] Jinja2 template engine documentation - no arXiv ID (documentation)
- [gguf] The GGUF file-format specification, in the ggml repository - no arXiv ID (specification)

## Backfill

- [ ] Question 1: Why can the same visible text produce two id sequences? (normalizer, pre-tokenizer, and what the file does not record)
- [ ] Question 2: Where does the algorithm end and the dialect begin? (one merge loop, one regex table)
- [ ] Question 3: What must never be silent? (a missing byte token, an unsupported dialect, a special token inferred from its name)
- [ ] Question 4: What does a round trip actually prove? (bytes survive, text may not)

<!-- facts: template.provenance -->
*`scripts/tmpl.py`; renders the model's own `tokenizer.chat_template` (read out of the GGUF) with Jinja2, and compares every case byte for byte with the reference renderer's committed output (HuggingFace `transformers` 5.17.0 `apply_chat_template`); the token counts come from the Chapter 11 tokenizer on the same bytes; on **arm64, Darwin 27.2.0, python 3.15.2, jinja2 3.1.6**; byte equality on every case; a byte or token count that differs is a failure, not a statistic. Command: see [the provenance index](../appendix/provenance.md#template). Raw numbers: `facts/template.json`.*
<!-- /facts -->
