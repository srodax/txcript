//! A compact, one-way text projection of the canonical transcript model.
//!
//! [`to_text`] is intended for LLM context, not archival or round trips. It
//! preserves conversational content and compact tool data while discarding
//! replay-only data: message timestamps, usage, stop reasons, reasoning
//! signatures/encrypted payloads, and inline image bytes.
//!
//! [`to_text_fragment`] renders a [`Span`] of the body in the same format,
//! with a `── #N ──` rule numbering each message by its 1-based position in
//! the full session — the same ordinals fragment refs (`<id>#5-12`) use, so
//! what a reader sees is what they can reference.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::common::{Block, Message, Meta, Role, ToolOutput};
use crate::{Common, Span, Transcript};

/// Which block categories to include in a text projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextFilter {
    pub user: bool,
    pub assistant: bool,
    pub thinking: bool,
    pub tool_use: bool,
    pub tool_result: bool,
    /// If set, tool-use argument JSON longer than this many chars is cut
    /// after a prefix, with a truncation marker. `None` keeps arguments intact.
    pub max_tool_arg_chars: Option<usize>,
}

impl TextFilter {
    pub const fn all() -> Self {
        Self {
            user: true,
            assistant: true,
            thinking: true,
            tool_use: true,
            tool_result: true,
            max_tool_arg_chars: None,
        }
    }

    fn includes_images(self) -> bool {
        self.user && self.assistant && self.thinking && self.tool_use && self.tool_result
    }
}

/// Render a canonical transcript as compact, LLM-oriented text.
///
/// The format uses short bracketed labels instead of repeating the canonical
/// JSON schema. Tool-call ids are remapped to session-local integers so tool
/// results remain paired without carrying provider-generated identifiers.
#[must_use]
pub fn to_text(transcript: &Transcript<Common>) -> String {
    to_text_with_filter(transcript, TextFilter::all())
}

/// Render a canonical transcript as compact, LLM-oriented text, including
/// only the block categories enabled in `filter`.
#[must_use]
pub fn to_text_with_filter(transcript: &Transcript<Common>, filter: TextFilter) -> String {
    let mut out = String::new();
    header(&mut out, &transcript.meta);

    let mut tool_ids = HashMap::<&str, usize>::new();
    let mut next_tool_id = 1;
    for message in &transcript.body {
        let _ = blocks(&mut out, &mut tool_ids, &mut next_tool_id, message, filter);
    }

    out
}

/// Render `span` of the transcript in [`to_text`]'s format, a `── #N ──`
/// rule before each message carrying its 1-based ordinal in the full body.
/// Partial spans add `fragment=`/`of=` header fields naming the slice.
/// `None` when `span` is out of bounds, mirroring [`Transcript::fragment`].
#[must_use]
pub fn to_text_fragment(transcript: &Transcript<Common>, span: &Span) -> Option<String> {
    to_text_fragment_with_filter(transcript, span, TextFilter::all())
}

/// Render `span` of the transcript in [`to_text`]'s format, including only
/// the block categories enabled in `filter`.
#[must_use]
pub fn to_text_fragment_with_filter(
    transcript: &Transcript<Common>,
    span: &Span,
    filter: TextFilter,
) -> Option<String> {
    transcript.fragment(span).map(|messages| {
        let mut out = String::new();
        header(&mut out, &transcript.meta);
        let total = transcript.body.len();
        if span.0 != (0..total) {
            field(&mut out, "fragment", &format_span(span));
            field(&mut out, "of", &total.to_string());
        }

        let mut tool_ids = HashMap::<&str, usize>::new();
        let mut next_tool_id = 1;
        for (offset, message) in messages.iter().enumerate() {
            let mut message_out = String::new();
            if blocks(
                &mut message_out,
                &mut tool_ids,
                &mut next_tool_id,
                message,
                filter,
            ) {
                // No trailing newline: `section` supplies the separator, keeping
                // the rule flush against the first label under it.
                let _ = write!(out, "\n── #{} ──", span.0.start + offset + 1);
                out.push_str(&message_out);
            }
        }
        out
    })
}

/// Human-facing `#a-b` (`#a` for a single message) for a resolved span.
fn format_span(span: &Span) -> String {
    match span.0.len() {
        1 => format!("#{}", span.0.start + 1),
        _ => format!("#{}-{}", span.0.start + 1, span.0.end),
    }
}

fn header(out: &mut String, meta: &Meta) {
    out.push_str("[session]\n");
    field(out, "id", &meta.id);
    field(out, "started", &meta.timestamp.to_rfc3339());
    optional_field(out, "title", meta.title.as_deref());
    optional_field(out, "cwd", meta.cwd.as_deref());
    optional_field(out, "branch", meta.git_branch.as_deref());
    optional_field(out, "model", meta.model.as_deref());
}

/// Render one message's blocks. The tool-id map is threaded across calls so
/// `[tool N …]`/`[result N]` stay paired over the whole render.
fn blocks<'a>(
    out: &mut String,
    tool_ids: &mut HashMap<&'a str, usize>,
    next_tool_id: &mut usize,
    message: &'a Message,
    filter: TextFilter,
) -> bool {
    let mut rendered_any = false;
    for block in &message.content {
        match block {
            Block::Text { text } => {
                let (include, label) = match message.role {
                    Role::User if filter.user => (true, "user"),
                    Role::Assistant if filter.assistant => (true, "assistant"),
                    _ => (false, ""),
                };
                if include {
                    section(out, label, text);
                    rendered_any = true;
                }
            }
            Block::Thinking { text, .. } if filter.thinking => {
                section(out, "thinking", text);
                rendered_any = true;
            }
            Block::ToolUse { id, tool } if filter.tool_use => {
                let short_id = short_tool_id(tool_ids, next_tool_id, id);
                let (name, input) = tool.to_canonical();
                // A tool invoked with no arguments — a bare slash command,
                // say — renders as its label alone rather than a stray `{}`.
                let body = match &input {
                    serde_json::Value::Null => String::new(),
                    serde_json::Value::Object(map) if map.is_empty() => String::new(),
                    input => maybe_truncate_tool_arg(&input.to_string(), filter.max_tool_arg_chars),
                };
                section(out, &format!("tool {short_id} {}", one_line(&name)), &body);
                rendered_any = true;
            }
            Block::ToolResult {
                tool_use_id,
                content,
                is_error,
            } if filter.tool_result => {
                let short_id = short_tool_id(tool_ids, next_tool_id, tool_use_id);
                let error = if *is_error { " error" } else { "" };
                let label = format!("result {short_id}{error}");
                match content {
                    ToolOutput::Text(text) => section(out, &label, text),
                    ToolOutput::Json(value) => section(out, &label, &value.to_string()),
                }
                rendered_any = true;
            }
            Block::Image { source } if filter.includes_images() => {
                section(
                    out,
                    &format!("image {} omitted", one_line(&source.media_type)),
                    "",
                );
                rendered_any = true;
            }
            Block::Thinking { .. }
            | Block::ToolUse { .. }
            | Block::ToolResult { .. }
            | Block::Image { .. } => {}
        }
    }
    rendered_any
}

fn field(out: &mut String, name: &str, value: &str) {
    let _ = writeln!(out, "{name}={}", one_line(value));
}

fn optional_field(out: &mut String, name: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        field(out, name, value);
    }
}

fn one_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn maybe_truncate_tool_arg(text: &str, max_chars: Option<usize>) -> String {
    let Some(max_chars) = max_chars else {
        return text.to_string();
    };
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }
    let kept: String = text.chars().take(max_chars).collect();
    format!("{kept}\n[truncated tool argument: kept {max_chars} of {total} chars]")
}

fn section(out: &mut String, label: &str, text: &str) {
    if !out.ends_with("\n\n") {
        out.push('\n');
    }
    let _ = writeln!(out, "[{label}]");
    out.push_str(text);
    if !text.ends_with('\n') {
        out.push('\n');
    }
}

fn short_tool_id<'a>(
    ids: &mut HashMap<&'a str, usize>,
    next_id: &mut usize,
    provider_id: &'a str,
) -> usize {
    *ids.entry(provider_id).or_insert_with(|| {
        let id = *next_id;
        *next_id += 1;
        id
    })
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};
    use serde_json::json;

    use super::*;
    use crate::common::{ImageSource, Message, Meta, StopReason, Tool, Usage};

    fn transcript(body: Vec<Message>) -> Transcript<Common> {
        Transcript::new(
            Meta {
                id: "provider-session-id".into(),
                timestamp: DateTime::<Utc>::UNIX_EPOCH,
                cwd: Some("/work/project".into()),
                git_branch: Some("main".into()),
                title: Some("Fix the parser".into()),
                cli_version: Some("9.9.9".into()),
                model: Some("model-name".into()),
            },
            body,
        )
    }

    fn message(role: Role, content: Vec<Block>) -> Message {
        Message {
            role,
            content,
            timestamp: DateTime::<Utc>::UNIX_EPOCH,
            model: None,
            stop_reason: Some(StopReason::EndTurn),
            usage: Some(Usage {
                input_tokens: 100,
                output_tokens: 20,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            }),
        }
    }

    #[test]
    fn renders_conversation_and_omits_replay_only_data() {
        let rendered = to_text(&transcript(vec![
            message(
                Role::User,
                vec![Block::Text {
                    text: "Please fix it.".into(),
                }],
            ),
            message(
                Role::Assistant,
                vec![
                    Block::Thinking {
                        text: "I should inspect the file.".into(),
                        signature: Some("large-signature".into()),
                        encrypted: Some("large-encrypted-payload".into()),
                    },
                    Block::Text {
                        text: "I found the issue.".into(),
                    },
                    Block::Image {
                        source: ImageSource {
                            source_type: "base64".into(),
                            media_type: "image/png".into(),
                            data: "large-base64-payload".into(),
                        },
                    },
                ],
            ),
        ]));

        assert!(rendered.contains("[user]\nPlease fix it."));
        assert!(rendered.contains("[thinking]\nI should inspect the file."));
        assert!(rendered.contains("[assistant]\nI found the issue."));
        assert!(rendered.contains("[image image/png omitted]"));
        assert!(!rendered.contains("large-signature"));
        assert!(!rendered.contains("large-encrypted-payload"));
        assert!(!rendered.contains("large-base64-payload"));
        assert!(!rendered.contains("input_tokens"));
        assert!(!rendered.contains("cli_version"));
    }

    #[test]
    fn compacts_tool_json_and_shortens_provider_ids() {
        let rendered = to_text(&transcript(vec![
            message(
                Role::Assistant,
                vec![Block::ToolUse {
                    id: "provider-generated-tool-id-with-many-tokens".into(),
                    tool: Tool::Read {
                        file_path: "src/lib.rs".into(),
                        offset: None,
                        limit: Some(20),
                    },
                }],
            ),
            message(
                Role::User,
                vec![Block::ToolResult {
                    tool_use_id: "provider-generated-tool-id-with-many-tokens".into(),
                    content: ToolOutput::Json(json!({"lines": ["one", "two"]})),
                    is_error: false,
                }],
            ),
        ]));

        assert!(rendered.contains("[tool 1 Read]\n{\"file_path\":\"src/lib.rs\",\"limit\":20}"));
        assert!(rendered.contains("[result 1]\n{\"lines\":[\"one\",\"two\"]}"));
        assert!(!rendered.contains("provider-generated-tool-id-with-many-tokens"));
    }

    #[test]
    fn unmatched_results_still_receive_stable_short_ids() {
        let rendered = to_text(&transcript(vec![message(
            Role::User,
            vec![
                Block::ToolResult {
                    tool_use_id: "second".into(),
                    content: ToolOutput::Text("failed".into()),
                    is_error: true,
                },
                Block::ToolResult {
                    tool_use_id: "second".into(),
                    content: ToolOutput::Text("again".into()),
                    is_error: false,
                },
            ],
        )]));

        assert!(rendered.contains("[result 1 error]\nfailed"));
        assert!(rendered.contains("[result 1]\nagain"));
    }

    fn three_messages() -> Transcript<Common> {
        transcript(vec![
            message(
                Role::User,
                vec![Block::Text {
                    text: "first".into(),
                }],
            ),
            message(
                Role::Assistant,
                vec![Block::Text {
                    text: "second".into(),
                }],
            ),
            message(
                Role::User,
                vec![Block::Text {
                    text: "third".into(),
                }],
            ),
        ])
    }

    #[test]
    fn fragment_numbers_messages_with_full_session_ordinals() {
        let full = to_text_fragment(&three_messages(), &Span(0..3)).unwrap();
        assert!(full.contains("── #1 ──\n[user]\nfirst"));
        assert!(full.contains("── #3 ──\n[user]\nthird"));
        assert!(!full.contains("fragment="));

        let partial = to_text_fragment(&three_messages(), &Span(1..3)).unwrap();
        assert!(partial.contains("fragment=#2-3"));
        assert!(partial.contains("of=3"));
        assert!(partial.contains("── #2 ──\n[assistant]\nsecond"));
        assert!(!partial.contains("first"));

        let single = to_text_fragment(&three_messages(), &Span(1..2)).unwrap();
        assert!(single.contains("fragment=#2\n"));

        assert!(to_text_fragment(&three_messages(), &Span(1..4)).is_none());
    }

    #[test]
    fn fragment_tool_ids_stay_paired_across_messages() {
        let rendered = to_text_fragment(
            &transcript(vec![
                message(
                    Role::Assistant,
                    vec![Block::ToolUse {
                        id: "provider-a".into(),
                        tool: Tool::Raw {
                            tool_name: "Bash".into(),
                            input: json!({"command": "ls"}),
                        },
                    }],
                ),
                message(
                    Role::User,
                    vec![Block::ToolResult {
                        tool_use_id: "provider-a".into(),
                        content: ToolOutput::Text("ok".into()),
                        is_error: false,
                    }],
                ),
            ]),
            &Span(0..2),
        )
        .unwrap();

        assert!(rendered.contains("[tool 1 Bash]"));
        assert!(rendered.contains("[result 1]\nok"));
        assert!(!rendered.contains("provider-a"));
    }

    fn representative_transcript() -> Transcript<Common> {
        transcript(vec![
            message(
                Role::User,
                vec![Block::Text {
                    text: "line one\nline two".into(),
                }],
            ),
            message(
                Role::Assistant,
                vec![
                    Block::Thinking {
                        text: "secret reasoning".into(),
                        signature: None,
                        encrypted: None,
                    },
                    Block::Text {
                        text: "assistant\nexplanation".into(),
                    },
                    Block::ToolUse {
                        id: "tool-1".into(),
                        tool: Tool::Raw {
                            tool_name: "Shell".into(),
                            input: json!({"command": "echo hi"}),
                        },
                    },
                ],
            ),
            message(
                Role::User,
                vec![Block::ToolResult {
                    tool_use_id: "tool-1".into(),
                    content: ToolOutput::Text("UNIQUE_TOOL_STDOUT".into()),
                    is_error: false,
                }],
            ),
            message(
                Role::User,
                vec![Block::Text {
                    text: "follow up".into(),
                }],
            ),
        ])
    }

    const HANDOFF: TextFilter = TextFilter {
        user: true,
        assistant: true,
        thinking: false,
        tool_use: true,
        tool_result: false,
        max_tool_arg_chars: None,
    };

    #[test]
    fn unfiltered_output_matches_all_filter() {
        let t = representative_transcript();
        assert_eq!(to_text(&t), to_text_with_filter(&t, TextFilter::all()));
        assert_eq!(
            to_text_fragment(&t, &Span(0..4)),
            to_text_fragment_with_filter(&t, &Span(0..4), TextFilter::all())
        );
    }

    #[test]
    fn handoff_filter_keeps_user_assistant_and_tool_use_verbatim() {
        let rendered = to_text_with_filter(&representative_transcript(), HANDOFF);
        assert!(rendered.contains("[user]\nline one\nline two"));
        assert!(rendered.contains("[assistant]\nassistant\nexplanation"));
        assert!(rendered.contains("[tool 1 Shell]\n{\"command\":\"echo hi\"}"));
    }

    #[test]
    fn handoff_filter_drops_thinking_and_tool_results() {
        let rendered = to_text_with_filter(&representative_transcript(), HANDOFF);
        assert!(!rendered.contains("secret reasoning"));
        assert!(!rendered.contains("UNIQUE_TOOL_STDOUT"));
        assert!(!rendered.contains("[result"));
    }

    #[test]
    fn tool_result_only_message_disappears_entirely() {
        let rendered =
            to_text_fragment_with_filter(&representative_transcript(), &Span(0..4), HANDOFF)
                .unwrap();
        assert!(!rendered.contains("UNIQUE_TOOL_STDOUT"));
        assert!(
            !rendered
                .lines()
                .any(|line| line.contains("UNIQUE_TOOL_STDOUT"))
        );
    }

    #[test]
    fn assistant_text_and_tool_use_stay_in_order() {
        let rendered = to_text_with_filter(&representative_transcript(), HANDOFF);
        let assistant = rendered
            .find("[assistant]\nassistant")
            .expect("assistant text");
        let tool = rendered.find("[tool 1 Shell]").expect("tool use");
        assert!(assistant < tool);
    }

    #[test]
    fn fragment_range_resolves_before_filtering() {
        let rendered =
            to_text_fragment_with_filter(&representative_transcript(), &Span(2..3), HANDOFF)
                .unwrap();
        assert!(rendered.contains("fragment=#3"));
        assert!(rendered.contains("of=4"));
        assert!(!rendered.contains("── #1 ──"));
        assert!(!rendered.contains("follow up"));
        assert!(!rendered.contains("UNIQUE_TOOL_STDOUT"));
    }

    #[test]
    fn selective_filter_omits_image_shells() {
        let t = transcript(vec![message(
            Role::Assistant,
            vec![
                Block::Text {
                    text: "see this".into(),
                },
                Block::Image {
                    source: ImageSource {
                        source_type: "base64".into(),
                        media_type: "image/png".into(),
                        data: "payload".into(),
                    },
                },
            ],
        )]);
        let all = to_text(&t);
        assert!(all.contains("[image image/png omitted]"));
        let filtered = to_text_with_filter(&t, HANDOFF);
        assert!(filtered.contains("[assistant]\nsee this"));
        assert!(!filtered.contains("[image"));
    }

    #[test]
    fn truncates_long_tool_arguments_and_keeps_a_prefix() {
        let long = "BEGIN_PATCH ".to_string() + &"ABCDEFGHIJ".repeat(20);
        let t = transcript(vec![message(
            Role::Assistant,
            vec![Block::ToolUse {
                id: "tool-1".into(),
                tool: Tool::Raw {
                    tool_name: "exec".into(),
                    input: json!({"cmd": long}),
                },
            }],
        )]);
        let mut filter = HANDOFF;
        filter.max_tool_arg_chars = Some(40);
        let rendered = to_text_with_filter(&t, filter);
        assert!(rendered.contains("[tool 1 exec]"));
        assert!(rendered.contains("BEGIN_PATCH"));
        assert!(rendered.contains("[truncated tool argument: kept 40 of"));
        assert!(!rendered.contains(&"ABCDEFGHIJ".repeat(20)));
        let total = rendered
            .split("[tool 1 exec]\n")
            .nth(1)
            .unwrap()
            .chars()
            .count();
        assert!(total < 200, "truncated body should stay small, got {total}");
    }

    #[test]
    fn short_tool_arguments_are_not_truncated() {
        let rendered = to_text_with_filter(
            &representative_transcript(),
            TextFilter {
                max_tool_arg_chars: Some(40),
                ..HANDOFF
            },
        );
        assert!(rendered.contains("{\"command\":\"echo hi\"}"));
        assert!(!rendered.contains("truncated tool argument"));
    }

    #[test]
    fn tool_arg_limit_does_not_truncate_user_or_assistant_text() {
        let long = "USERTEXT ".to_string() + &"x".repeat(80);
        let t = transcript(vec![message(
            Role::User,
            vec![Block::Text { text: long.clone() }],
        )]);
        let rendered = to_text_with_filter(
            &t,
            TextFilter {
                max_tool_arg_chars: Some(20),
                ..HANDOFF
            },
        );
        assert!(rendered.contains(&long));
        assert!(!rendered.contains("truncated tool argument"));
    }
}
