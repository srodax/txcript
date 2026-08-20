//! `txcript view` — print a session as compact text.
//!
//! The source is a session id or exact title, looked up like `continue`,
//! with an optional `#range` fragment (see `fragment.rs`). Output goes to
//! stdout, colorless and pager-free, so it pipes cleanly into `pbcopy` or an
//! LLM prompt. Message numbers are printed in the output (`── #N ──` rules),
//! so what you see is what you reference.

use std::process::ExitCode;

use clap::ValueEnum;
use txcript::{
    HarnessId, Span,
    text::{self, TextFilter},
};

use crate::fragment;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ViewInclude {
    User,
    Assistant,
    Thinking,
    #[value(name = "tool-use")]
    ToolUse,
    #[value(name = "tool-result")]
    ToolResult,
}

pub fn text_filter_from_includes(include: &[ViewInclude]) -> TextFilter {
    if include.is_empty() {
        return TextFilter::all();
    }
    TextFilter {
        user: include.contains(&ViewInclude::User),
        assistant: include.contains(&ViewInclude::Assistant),
        thinking: include.contains(&ViewInclude::Thinking),
        tool_use: include.contains(&ViewInclude::ToolUse),
        tool_result: include.contains(&ViewInclude::ToolResult),
        max_tool_arg_chars: None,
    }
}

pub fn cmd_view(
    source: &str,
    from: Option<HarnessId>,
    include: &[ViewInclude],
    drop_leading_setup: bool,
    drop_trailing_handoff: bool,
    max_tool_arg_chars: Option<usize>,
) -> Result<ExitCode, String> {
    let sessions = super::discover_with_spinner();
    // A whole-input match (a title that itself contains `#12`) beats the
    // fragment interpretation.
    let (src, request) = match fragment::parse_ref(source) {
        (_, Some(_)) if super::find_exact(&sessions, from, source).is_some() => (source, None),
        parsed => parsed,
    };

    let session = super::find_session(&sessions, from, src)?.ok_or_else(|| {
        let scope = from.map_or(String::new(), |h| format!(" {h}"));
        format!("no local{scope} session matches `{src}` (try `txcript list`)")
    })?;
    let mut common = session
        .read()
        .map_err(|e| format!("reading session `{src}`: {e}"))?;
    if drop_leading_setup {
        common.body = txcript::trim::drop_leading_setup(&common.body);
    }
    if drop_trailing_handoff {
        common.body = txcript::trim::drop_trailing_handoff(&common.body);
    }

    let total = common.body.len();
    let span = match &request {
        Some(req) => req.resolve(total)?,
        None => Span(0..total),
    };
    let mut filter = text_filter_from_includes(include);
    filter.max_tool_arg_chars = max_tool_arg_chars;
    // `resolve` bounds-checked against `total`, so the render always lands.
    let rendered = text::to_text_fragment_with_filter(&common, &span, filter)
        .ok_or_else(|| format!("range is out of bounds — the session has {total} messages"))?;
    // A failed write means the reader is gone (`txcript view … | head`):
    // finish quietly instead of panicking the way `print!` would.
    let _ = std::io::Write::write_all(&mut std::io::stdout(), rendered.as_bytes());
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use clap::{Parser, Subcommand};

    use super::*;

    #[derive(Parser)]
    struct Cli {
        #[command(subcommand)]
        command: Command,
    }

    #[derive(Debug, PartialEq, Subcommand)]
    enum Command {
        View {
            source: String,
            #[arg(long)]
            from: Option<HarnessId>,
            #[arg(long, value_delimiter = ',', value_enum)]
            include: Vec<ViewInclude>,
            #[arg(long)]
            drop_leading_setup: bool,
            #[arg(long)]
            drop_trailing_handoff: bool,
            #[arg(long)]
            max_tool_arg_chars: Option<usize>,
        },
    }

    fn parse(args: &[&str]) -> Command {
        let argv: Vec<&str> = std::iter::once("txcript")
            .chain(args.iter().copied())
            .collect();
        Cli::try_parse_from(argv).unwrap().command
    }

    #[test]
    fn parses_single_include_category() {
        assert_eq!(
            parse(&["view", "abc", "--include", "user"]),
            Command::View {
                source: "abc".into(),
                from: None,
                include: vec![ViewInclude::User],
                drop_leading_setup: false,
                drop_trailing_handoff: false,
                max_tool_arg_chars: None,
            }
        );
    }

    #[test]
    fn parses_multiple_include_categories() {
        assert_eq!(
            parse(&["view", "abc", "--include", "user,assistant,tool-use"]),
            Command::View {
                source: "abc".into(),
                from: None,
                include: vec![
                    ViewInclude::User,
                    ViewInclude::Assistant,
                    ViewInclude::ToolUse,
                ],
                drop_leading_setup: false,
                drop_trailing_handoff: false,
                max_tool_arg_chars: None,
            }
        );
    }

    #[test]
    fn parses_include_with_from() {
        assert_eq!(
            parse(&[
                "view",
                "abc",
                "--from",
                "codex",
                "--include",
                "user,assistant,tool-use",
            ]),
            Command::View {
                source: "abc".into(),
                from: Some(HarnessId::Codex),
                include: vec![
                    ViewInclude::User,
                    ViewInclude::Assistant,
                    ViewInclude::ToolUse,
                ],
                drop_leading_setup: false,
                drop_trailing_handoff: false,
                max_tool_arg_chars: None,
            }
        );
    }

    #[test]
    fn rejects_unknown_include_category() {
        let argv = ["txcript", "view", "abc", "--include", "banana"];
        assert!(Cli::try_parse_from(argv).is_err());
    }

    #[test]
    fn parses_handoff_trim_flags() {
        assert_eq!(
            parse(&[
                "view",
                "abc",
                "--drop-leading-setup",
                "--drop-trailing-handoff",
            ]),
            Command::View {
                source: "abc".into(),
                from: None,
                include: vec![],
                drop_leading_setup: true,
                drop_trailing_handoff: true,
                max_tool_arg_chars: None,
            }
        );
    }

    #[test]
    fn parses_max_tool_arg_chars() {
        assert_eq!(
            parse(&["view", "abc", "--max-tool-arg-chars", "2000"]),
            Command::View {
                source: "abc".into(),
                from: None,
                include: vec![],
                drop_leading_setup: false,
                drop_trailing_handoff: false,
                max_tool_arg_chars: Some(2000),
            }
        );
    }

    #[test]
    fn empty_include_means_all_categories() {
        assert_eq!(text_filter_from_includes(&[]), TextFilter::all());
    }

    #[test]
    fn selective_include_disables_unlisted_categories() {
        let filter = text_filter_from_includes(&[
            ViewInclude::User,
            ViewInclude::Assistant,
            ViewInclude::ToolUse,
        ]);
        assert!(filter.user);
        assert!(filter.assistant);
        assert!(filter.tool_use);
        assert!(!filter.thinking);
        assert!(!filter.tool_result);
    }
}
