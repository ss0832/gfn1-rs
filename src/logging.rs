// SPDX-License-Identifier: GPL-3.0-or-later
//! Optional append-only JSON Lines calculation logs. No hostnames, environment
//! dumps, user names, working directories or absolute input paths are recorded.
use crate::error::{Gfn1Error, Result};
use serde_json::{json, Value};
use std::{
    cell::RefCell,
    fmt,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Error,
    Info,
    Debug,
    Trace,
}
impl LogLevel {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "error" => Ok(Self::Error),
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            "trace" => Ok(Self::Trace),
            _ => Err(Gfn1Error::InvalidInput(
                "log level must be error, info, debug or trace".into(),
            )),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}
struct Inner {
    id: u64,
    writer: Mutex<BufWriter<File>>,
    maximum: LogLevel,
    console: bool,
    session: String,
    sequence: AtomicU64,
}
#[derive(Clone)]
pub struct CalculationLogger(Arc<Inner>);
// Stable Debug deliberately excludes paths, clocks and mutable writer state.
impl fmt::Debug for CalculationLogger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CalculationLogger")
    }
}
static LOGGER_ID: AtomicU64 = AtomicU64::new(1);
thread_local! { static SPANS: RefCell<Vec<(u64,String)>> = const { RefCell::new(Vec::new()) }; }
fn timestamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}
fn safe_message(message: &str) -> String {
    message
        .split_whitespace()
        .map(|word| {
            let token = word.trim_matches(['`', '"', '\'', '(', ')', '[', ']', ',']);
            let b = token.as_bytes();
            let drive = b.len() > 2 && b[1] == b':' && (b[2] == b'\\' || b[2] == b'/');
            let non_english = word
                .chars()
                .any(|c| matches!(c as u32,0x3040..=0x30ff|0x3400..=0x9fff|0xf900..=0xfaff));
            if drive
                || token.starts_with('/')
                || token.contains('\\')
                || token.contains('@')
                || non_english
            {
                "<redacted>"
            } else {
                word
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
impl CalculationLogger {
    pub fn append(path: impl AsRef<Path>, maximum: LogLevel, console: bool) -> Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let id = LOGGER_ID.fetch_add(1, Ordering::Relaxed);
        let ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let logger = Self(Arc::new(Inner {
            id,
            writer: Mutex::new(BufWriter::new(file)),
            maximum,
            console,
            session: format!("{ns:x}-{id:x}"),
            sequence: AtomicU64::new(1),
        }));
        logger.event(
            LogLevel::Info,
            "logger_started",
            json!({"format":"jsonl","maximum_level":maximum.name()}),
        )?;
        Ok(logger)
    }
    fn current(&self) -> Option<String> {
        SPANS.with(|s| {
            s.borrow()
                .iter()
                .rev()
                .find(|(id, _)| *id == self.0.id)
                .map(|(_, span)| span.clone())
        })
    }
    fn write(
        &self,
        level: LogLevel,
        event: &str,
        id: Option<&str>,
        parent: Option<&str>,
        data: Value,
    ) -> Result<()> {
        if level > self.0.maximum {
            return Ok(());
        }
        let record = json!({"schema":"gfn1-rs-log-v1","timestamp_unix_ms":timestamp(),"level":level.name(),"event":event,
            "session":self.0.session,"operation_id":id,"parent_id":parent,"data":data});
        let mut writer = self
            .0
            .writer
            .lock()
            .map_err(|_| std::io::Error::other("calculation log lock poisoned"))?;
        serde_json::to_writer(&mut *writer, &record).map_err(std::io::Error::other)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        if self.0.console
            && (event.starts_with("operation_")
                || event.contains("fallback")
                || event.contains("polish")
                || event == "ml_model_loaded")
        {
            eprintln!("[gfn1-rs {}] {} {}", level.name(), event, record["data"]);
        }
        Ok(())
    }
    pub fn event(&self, level: LogLevel, event: &str, data: Value) -> Result<()> {
        let id = self.current();
        self.write(level, event, id.as_deref(), None, data)
    }
    pub fn begin(&self, operation: &str, data: Value) -> Result<LogSpan> {
        self.begin_level(LogLevel::Info, operation, data)
    }
    pub fn begin_level(&self, level: LogLevel, operation: &str, data: Value) -> Result<LogSpan> {
        let parent = self.current();
        let id = format!(
            "{}-{}",
            self.0.session,
            self.0.sequence.fetch_add(1, Ordering::Relaxed)
        );
        self.write(
            level,
            "operation_start",
            Some(&id),
            parent.as_deref(),
            json!({"operation":operation,"configuration":data}),
        )?;
        if level <= self.0.maximum {
            SPANS.with(|s| s.borrow_mut().push((self.0.id, id.clone())));
        }
        Ok(LogSpan {
            logger: self.clone(),
            id,
            parent,
            operation: operation.into(),
            level,
            start: Instant::now(),
            finished: false,
        })
    }
}
pub struct LogSpan {
    logger: CalculationLogger,
    id: String,
    parent: Option<String>,
    operation: String,
    level: LogLevel,
    start: Instant,
    finished: bool,
}
impl LogSpan {
    pub fn id(&self) -> &str {
        &self.id
    }
    fn pop(&self) {
        SPANS.with(|s| {
            s.borrow_mut()
                .retain(|(logger, id)| *logger != self.logger.0.id || id != &self.id)
        });
    }
    pub fn success(mut self, data: Value) -> Result<()> {
        self.pop();
        self.finished = true;
        self.logger.write(self.level,"operation_end",Some(&self.id),self.parent.as_deref(),json!({"operation":self.operation,"status":"success","elapsed_seconds":self.start.elapsed().as_secs_f64(),"result":data}))
    }
    pub fn failure(mut self, error: &Gfn1Error) -> Result<()> {
        self.pop();
        self.finished = true;
        let kind = match error {
            Gfn1Error::Io(_) => "io",
            Gfn1Error::Parse { .. } => "parse",
            Gfn1Error::InvalidInput(_) => "invalid_input",
            Gfn1Error::MissingElement(_) | Gfn1Error::MissingGlobal(_) => "missing_parameter",
            Gfn1Error::LinearAlgebra(_) => "linear_algebra",
            Gfn1Error::SingularCell => "singular_cell",
            Gfn1Error::SccNotConverged { .. } => "scf_not_converged",
        };
        self.logger.write(LogLevel::Error,"operation_end",Some(&self.id),self.parent.as_deref(),json!({"operation":self.operation,"status":"error","elapsed_seconds":self.start.elapsed().as_secs_f64(),"error_kind":kind,"reason":safe_message(&error.to_string())}))
    }
}
impl Drop for LogSpan {
    fn drop(&mut self) {
        if !self.finished {
            self.pop();
            let _=self.logger.write(LogLevel::Error,"operation_unfinished",Some(&self.id),self.parent.as_deref(),json!({"operation":self.operation,"status":"aborted","elapsed_seconds":self.start.elapsed().as_secs_f64()}));
        }
    }
}

pub(crate) fn finish<T>(
    span: Option<LogSpan>,
    result: Result<T>,
    summary: impl FnOnce(&T) -> Value,
) -> Result<T> {
    match result {
        Ok(value) => {
            if let Some(span) = span {
                span.success(summary(&value))?;
            }
            Ok(value)
        }
        Err(error) => {
            if let Some(span) = span {
                span.failure(&error)?;
            }
            Err(error)
        }
    }
}
