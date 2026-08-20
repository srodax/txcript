//! View-time trims for cross-agent handoff projections.
//!
//! These drop harness/repo environment that a receiving agent will reload,
//! and the trailing handoff-save ceremony. They do not change harness
//! conversion.

use crate::common::{Block, Message, Role, Tool, ToolOutput};

const SETUP_PREFIXES: &[&str] = &[
    "<environment_context>",
    "<permissions instructions>",
    "<collaboration_mode>",
    "<sandbox_mode>",
    "<approval_policy>",
    "<recommended_plugins>",
    "<user_info>",
    "<agent_skills>",
    "<mcp_meta_tools>",
    "<rules>",
    "<open_and_recently_viewed_files>",
    "<always_applied_workspace_rules>",
    "<system-reminder>",
    "<INSTRUCTIONS>",
    "# AGENTS.md",
];

fn is_setup_text(text: &str) -> bool {
    let t = text.trim_start();
    SETUP_PREFIXES.iter().any(|prefix| t.starts_with(prefix))
}

fn extract_user_query(text: &str) -> Option<String> {
    let start = text.find("<user_query>")?;
    let rest = &text[start + "<user_query>".len()..];
    let end = rest.find("</user_query>")?;
    let query = rest[..end].trim();
    (!query.is_empty()).then(|| query.to_string())
}

fn is_conversational_block(block: &Block) -> bool {
    matches!(
        block,
        Block::ToolUse { .. } | Block::ToolResult { .. } | Block::Image { .. }
    )
}

fn trim_leading_user_message(message: &Message) -> Option<Message> {
    if message.role != Role::User {
        return Some(message.clone());
    }
    if message.content.iter().any(is_conversational_block) {
        return Some(message.clone());
    }

    let mut content = Vec::new();
    for block in &message.content {
        match block {
            Block::Text { text } => {
                if let Some(query) = extract_user_query(text) {
                    content.push(Block::Text { text: query });
                } else if !is_setup_text(text) {
                    content.push(block.clone());
                }
            }
            other => content.push(other.clone()),
        }
    }
    (!content.is_empty()).then(|| Message {
        content,
        ..message.clone()
    })
}

/// Drop leading user messages that are only environment/repo scaffolding.
#[must_use]
pub fn drop_leading_setup(messages: &[Message]) -> Vec<Message> {
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        if message.role == Role::User {
            if let Some(trimmed) = trim_leading_user_message(message) {
                let mut kept = Vec::with_capacity(messages.len() - index);
                kept.push(trimmed);
                kept.extend(messages[index + 1..].iter().cloned());
                return kept;
            }
            index += 1;
            continue;
        }
        return messages[index..].to_vec();
    }
    Vec::new()
}

fn tool_blob(tool: &Tool) -> String {
    match tool {
        Tool::Command { command, args } => match args {
            Some(args) => format!("{command} {args}"),
            None => command.clone(),
        },
        Tool::Read { file_path, .. }
        | Tool::Edit { file_path, .. }
        | Tool::MultiEdit { file_path, .. } => file_path.clone(),
        Tool::Write { file_path, content } => format!("{file_path}\n{content}"),
        Tool::Bash { command, .. } => command.clone(),
        Tool::Raw { tool_name, input } => format!("{tool_name} {input}"),
    }
}

fn message_blob(message: &Message) -> String {
    message
        .content
        .iter()
        .map(|block| match block {
            Block::Text { text } | Block::Thinking { text, .. } => text.clone(),
            Block::ToolUse { tool, .. } => tool_blob(tool),
            Block::ToolResult { content, .. } => match content {
                ToolOutput::Text(text) => text.clone(),
                ToolOutput::Json(value) => value.to_string(),
            },
            Block::Image { .. } => String::new(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn has_handoff_slash(text: &str) -> bool {
    text.lines().any(|line| {
        let trimmed = line.trim_start();
        trimmed == "/handoff" || trimmed.starts_with("/handoff ")
    })
}

fn has_user_text(message: &Message) -> bool {
    message.role == Role::User
        && message
            .content
            .iter()
            .any(|block| matches!(block, Block::Text { .. }))
}

fn is_load_request(blob: &str) -> bool {
    blob.lines().any(|line| {
        let trimmed = line.trim_start();
        trimmed.starts_with("/handoff load")
            || (trimmed.starts_with("[$handoff]")
                && trimmed.to_ascii_lowercase().contains(" load "))
    })
}

fn is_handoff_save_user(message: &Message) -> bool {
    if !has_user_text(message) {
        return false;
    }
    let blob = message_blob(message);
    if is_load_request(&blob) {
        return false;
    }
    blob.contains("[$handoff]")
        || blob.contains("<name>handoff</name>")
        || blob.contains("skills/handoff/SKILL.md")
        || has_handoff_slash(&blob)
}

/// Drop a trailing suffix of handoff-save ceremony messages.
#[must_use]
pub fn drop_trailing_handoff(messages: &[Message]) -> Vec<Message> {
    let last_work_user = messages
        .iter()
        .rposition(|message| has_user_text(message) && !is_handoff_save_user(message));
    let search_from = last_work_user.map_or(0, |index| index + 1);
    match messages[search_from..]
        .iter()
        .position(is_handoff_save_user)
    {
        Some(relative) => messages[..search_from + relative].to_vec(),
        None => messages.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};

    use super::*;
    use crate::common::{Message, Role, StopReason, Usage};

    fn message(role: Role, text: &str) -> Message {
        Message {
            role,
            content: vec![Block::Text {
                text: text.to_string(),
            }],
            timestamp: DateTime::<Utc>::UNIX_EPOCH,
            model: None,
            stop_reason: Some(StopReason::EndTurn),
            usage: Some(Usage {
                input_tokens: 1,
                output_tokens: 1,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            }),
        }
    }

    fn texts(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .map(|m| {
                m.content
                    .iter()
                    .filter_map(|b| match b {
                        Block::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect()
    }

    #[test]
    fn drops_codex_preamble_until_first_real_user_turn() {
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![
                    Block::Text {
                        text: "<recommended_plugins>\n- Figma\n</recommended_plugins>".into(),
                    },
                    Block::Text {
                        text:
                            "# AGENTS.md instructions for /repo\n\nMatch deliberation to the task."
                                .into(),
                    },
                    Block::Text {
                        text: "<environment_context>\n  <cwd>/repo</cwd>\n</environment_context>"
                            .into(),
                    },
                ],
                timestamp: DateTime::<Utc>::UNIX_EPOCH,
                model: None,
                stop_reason: None,
                usage: None,
            },
            message(
                Role::User,
                "[$distill-physics-context](/skills/distill-physics-context/SKILL.md)\nWrite the QTT notes.",
            ),
            message(Role::Assistant, "Using distill-physics-context."),
        ];

        let kept = drop_leading_setup(&messages);
        assert_eq!(
            texts(&kept),
            vec![
                "[$distill-physics-context](/skills/distill-physics-context/SKILL.md)\nWrite the QTT notes."
                    .to_string(),
                "Using distill-physics-context.".to_string(),
            ]
        );
    }

    #[test]
    fn keeps_user_query_from_mixed_cursor_blob() {
        let messages = vec![
            message(
                Role::User,
                "<user_info>\nOS Version: linux\nWorkspace Path: /repo\n</user_info>\n\n<user_query>\nfix the parser\n</user_query>",
            ),
            message(Role::Assistant, "I'll inspect the parser."),
        ];

        let kept = drop_leading_setup(&messages);
        assert_eq!(
            texts(&kept),
            vec![
                "fix the parser".to_string(),
                "I'll inspect the parser.".to_string()
            ]
        );
    }

    #[test]
    fn keeps_mid_session_work_skill() {
        let messages = vec![
            message(Role::User, "continue the notes"),
            message(
                Role::User,
                "<skill>\n<name>distill-physics-context</name>\nTeach the topic.\n</skill>",
            ),
            message(Role::Assistant, "Using distill-physics-context."),
        ];

        assert_eq!(drop_leading_setup(&messages).len(), 3);
    }

    #[test]
    fn drops_trailing_handoff_save_ceremony() {
        let messages = vec![
            message(Role::User, "Write the QTT notes."),
            message(Role::Assistant, "Approve Layer 1?"),
            message(
                Role::User,
                "[$handoff](/home/srodam/Software/agent-config/.agents/skills/handoff/SKILL.md)",
            ),
            message(
                Role::User,
                "<skill>\n<name>handoff</name>\n<path>/home/srodam/Software/agent-config/.agents/skills/handoff/SKILL.md</path>\n# Cross-Agent Handoff\n</skill>",
            ),
            message(
                Role::Assistant,
                "Using handoff. The required TXCRIPT_SESSION marker is absent.",
            ),
        ];

        let kept = drop_trailing_handoff(&messages);
        assert_eq!(
            texts(&kept),
            vec![
                "Write the QTT notes.".to_string(),
                "Approve Layer 1?".to_string(),
            ]
        );
    }

    #[test]
    fn drops_interstitial_save_chatter_after_handoff_invocation() {
        let messages = vec![
            message(Role::User, "Write the QTT notes."),
            message(Role::Assistant, "Approve Layer 1?"),
            message(
                Role::User,
                "[$handoff](/home/srodam/Software/agent-config/.agents/skills/handoff/SKILL.md)",
            ),
            message(Role::Assistant, "Using handoff."),
            message(
                Role::Assistant,
                "The first save attempt was blocked before execution because the command guard rejected the skill's rm -f failure cleanup.",
            ),
            bash("txcript view abc --from codex > docs/agent-transcripts/out.txcript"),
        ];

        let kept = drop_trailing_handoff(&messages);
        assert_eq!(
            texts(&kept),
            vec![
                "Write the QTT notes.".to_string(),
                "Approve Layer 1?".to_string(),
            ]
        );
    }

    #[test]
    fn keeps_leading_load_when_cutting_trailing_save() {
        let messages = vec![
            message(Role::User, "/handoff load 20260820-203740"),
            message(Role::Assistant, "Handoff context is loaded."),
            message(Role::User, "continue fixing the cache tests"),
            message(Role::Assistant, "I'll inspect the failing tests."),
            message(Role::User, "/handoff save"),
            message(Role::Assistant, "Using handoff."),
        ];

        let kept = drop_trailing_handoff(&messages);
        assert_eq!(
            texts(&kept),
            vec![
                "/handoff load 20260820-203740".to_string(),
                "Handoff context is loaded.".to_string(),
                "continue fixing the cache tests".to_string(),
                "I'll inspect the failing tests.".to_string(),
            ]
        );
    }

    fn bash(command: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![Block::ToolUse {
                id: "call-1".into(),
                tool: Tool::Bash {
                    command: command.to_string(),
                    workdir: None,
                    timeout_ms: None,
                    description: None,
                    run_in_background: false,
                },
            }],
            timestamp: DateTime::<Utc>::UNIX_EPOCH,
            model: None,
            stop_reason: None,
            usage: None,
        }
    }

    #[test]
    fn drops_trailing_txcript_view_save() {
        let messages = vec![
            message(Role::Assistant, "Approve Layer 1?"),
            message(Role::User, "/handoff"),
            bash(
                "txcript view abc --from codex --include user,assistant,tool-use > docs/agent-transcripts/out.txcript",
            ),
        ];
        let kept = drop_trailing_handoff(&messages);
        assert_eq!(texts(&kept), vec!["Approve Layer 1?".to_string()]);
    }

    #[test]
    fn skill_docs_mentioning_load_do_not_count_as_work() {
        let messages = vec![
            message(Role::User, "Write the QTT notes."),
            message(Role::Assistant, "Approve Layer 1?"),
            message(
                Role::User,
                "<skill>\n<name>handoff</name>\nIf the user's entire request is `/handoff load foo`:\n</skill>",
            ),
            message(Role::Assistant, "Using handoff."),
        ];
        let kept = drop_trailing_handoff(&messages);
        assert_eq!(
            texts(&kept),
            vec![
                "Write the QTT notes.".to_string(),
                "Approve Layer 1?".to_string(),
            ]
        );
    }

    #[test]
    fn identity_when_nothing_to_trim() {
        let messages = vec![message(Role::User, "hello"), message(Role::Assistant, "hi")];
        assert_eq!(drop_leading_setup(&messages), messages);
        assert_eq!(drop_trailing_handoff(&messages), messages);
    }
}
