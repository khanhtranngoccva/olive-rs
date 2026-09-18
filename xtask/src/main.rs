//! Developer task runner for the Olive workspace.
//!
//! Usage:
//! ```text
//! cargo xtask test       # run the basic unit + integration test suite
//! cargo xtask leak-test  # run tests on nightly with the leak sanitizer
//! cargo xtask miri-test  # run tests under Miri for undefined-behavior detection
//! cargo xtask control-doc  # generate full HTML docs (private + hidden) for std/alloc/core
//! ```

use clap::{Parser, Subcommand};
use std::path::Path;
use std::process::{Command, ExitCode};

/// The no_std-capable library crates that make up the Olive stack. Verification
/// tasks scope to these by default; `--workspace` widens to every member.
const CORE_CRATES: &[&str] = &["olive-core", "olive-alloc", "olive-std"];

#[derive(Parser)]
#[command(name = "xtask", about = "Olive developer task runner")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Subcommand)]
enum CommandKind {
    /// Run the basic unit and integration test suite (stable toolchain).
    Test {
        /// Include every workspace member instead of just the core crates.
        #[arg(long)]
        workspace: bool,
    },
    /// Run tests on nightly with -Zsanitizer=leak to catch memory leaks.
    LeakTest {
        /// Include every workspace member instead of just the core crates.
        #[arg(long)]
        workspace: bool,
    },
    /// Run tests under Miri to detect undefined behavior.
    MiriTest {
        /// Include every workspace member instead of just the core crates.
        #[arg(long)]
        workspace: bool,
    },
    /// Generate full HTML documentation (including private and hidden items)
    /// for std, alloc, and/or core from the nightly toolchain source.
    ControlDoc(DocArgs),
}

#[derive(clap::Args)]
struct DocArgs {
    /// Which crate(s) to document. Default: all three (std, alloc, core).
    #[arg(short, long, value_enum, num_args = 0..=3)]
    crate_: Vec<DocCrate>,

    /// Open the generated docs in Chrome after building. When multiple crates
    /// are selected, opens the first one listed.
    #[arg(long)]
    open: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum DocCrate {
    Std,
    Alloc,
    Core,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::Test { workspace } => cmd_test(workspace),
        CommandKind::LeakTest { workspace } => cmd_leak_test(workspace),
        CommandKind::MiriTest { workspace } => cmd_miri_test(workspace),
        CommandKind::ControlDoc(args) => cmd_control_doc(args),
    }
}

/// `-p` flags selecting the target crates.
fn scope_flags() -> Vec<String> {
    CORE_CRATES
        .iter()
        .flat_map(|c| vec!["-p".to_string(), c.to_string()])
        .collect()
}

/// Run the ordinary test suite on the pinned stable toolchain.
fn cmd_test(workspace: bool) -> ExitCode {
    let mut args: Vec<String> = vec!["test".into()];
    if workspace {
        args.push("--workspace".into());
    } else {
        args.extend(scope_flags());
    }
    run_cargo(&args)
}

/// Run `cargo +nightly test` with the leak sanitizer enabled. Requires a
/// nightly toolchain and an LLVM backend built with sanitizers.
fn cmd_leak_test(workspace: bool) -> ExitCode {
    let mut args: Vec<String> = vec![
        "+nightly".into(),
        "test".into(),
        "-Z".into(),
        "build-std".into(),
        "--target".into(),
        "x86_64-unknown-linux-gnu".into(),
        "--lib".into(),
        "--tests".into(),
    ];
    if workspace {
        args.push("--workspace".into());
    } else {
        args.extend(scope_flags());
    }
    let mut cmd = Command::new("cargo");
    cmd.args(&args)
        .env("RUSTFLAGS", "-Zunstable-options -Zsanitizer=leak");
    println!("$ {}", render_cmd(&cmd));
    exec(cmd)
}

/// Run `cargo +nightly miri test`. Requires the Miri component installed.
fn cmd_miri_test(workspace: bool) -> ExitCode {
    let mut args: Vec<String> = vec!["+nightly".into(), "miri".into(), "test".into()];
    if workspace {
        args.push("--workspace".into());
    } else {
        args.extend(scope_flags());
    }
    run_cargo(&args)
}

/// Generate full HTML docs (private + hidden items) for std/alloc/core from
/// the nightly toolchain's bundled source tree. Serves as a "control" reference
/// against which Olive's ported surface is compared for completeness.
fn cmd_control_doc(args: DocArgs) -> ExitCode {
    // Determine which crates to document.
    let crates: Vec<&str> = if args.crate_.is_empty() {
        vec!["std", "alloc", "core"]
    } else {
        args.crate_
            .iter()
            .map(|c| match c {
                DocCrate::Std => "std",
                DocCrate::Alloc => "alloc",
                DocCrate::Core => "core",
            })
            .collect()
    };

    // Locate the nightly sysroot and its library source directory.
    let sysroot = match run_capture(vec!["+nightly", "--print", "sysroot"]) {
        Some(s) => s,
        None => return ExitCode::FAILURE,
    };
    let library_dir = Path::new(sysroot.trim())
        .join("lib")
        .join("rustlib")
        .join("src")
        .join("rust")
        .join("library");

    if !library_dir.is_dir() {
        eprintln!(
            "error: nightly rust-src not found at {}\n\
             Install it with: rustup component add rust-src --toolchain nightly",
            library_dir.display()
        );
        return ExitCode::FAILURE;
    }

    // Determine target triple.
    let target = match run_capture(vec!["+nightly", "--print", "host-tuple"]) {
        Some(t) => t.trim().to_string(),
        None => return ExitCode::FAILURE,
    };

    // Build the cargo doc command. We invoke on the library workspace manifest
    // so that internal crate dependencies resolve without hitting the implicit
    // std linkage issue.
    let manifest_path = library_dir.join("Cargo.toml");

    let rustdocflags =
        r#"["--document-private-items","--document-hidden-items","-Z","unstable-options"]"#;

    let mut cmd = Command::new("cargo");
    cmd.arg("+nightly")
        .arg("doc")
        .arg("--manifest-path")
        .arg(&manifest_path);
    for c in &crates {
        cmd.arg("-p").arg(c);
    }
    cmd.arg("--no-deps")
        .arg("--target")
        .arg(&target)
        .arg("--config")
        .arg(format!("build.rustdocflags={rustdocflags}"))
        .env("STD_ENV_ARCH", arch_env_var());

    println!("$ {}", render_cmd(&cmd));
    println!();
    println!("Documenting: {}", crates.join(", "));
    println!("Source:      {}", library_dir.display());
    println!("Target:      {target}");
    println!();

    let status = exec(cmd);
    if status != ExitCode::SUCCESS {
        return status;
    }

    // The HTML output lands in <library_dir>/target/<triple>/doc/.
    let doc_out = library_dir.join("target").join(&target).join("doc");
    println!();
    println!("Docs written to: {}", doc_out.display());
    for c in &crates {
        let index = doc_out.join(c).join("index.html");
        if index.exists() {
            println!("  {c}: file://{}", index.display());
        }
    }

    if args.open {
        // Open the first crate's docs in the user's preferred browser.
        let target_crate = crates[0];
        let url = format!(
            "file://{}{}index.html",
            doc_out.join(target_crate).display(),
            std::path::MAIN_SEPARATOR
        );
        let browser = std::env::var("BROWSER").unwrap_or_else(|_| "xdg-open".into());
        println!();
        println!("Opening {target_crate} docs in {browser}...");
        match Command::new(&browser).arg(&url).spawn() {
            Ok(_) => {}
            Err(e) => {
                eprintln!(
                    "warning: failed to open browser '{browser}' ({e}); use the URL above manually."
                );
            }
        }
    } else {
        println!();
        println!("Open in a browser to browse the full API surface (or rerun with --open).");
    }
    ExitCode::SUCCESS
}

/// Map the host architecture to the value `STD_ENV_ARCH` expects.
fn arch_env_var() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        "x86_64"
    }
    #[cfg(target_arch = "aarch64")]
    {
        "aarch64"
    }
    #[cfg(target_arch = "arm")]
    {
        "arm"
    }
    #[cfg(target_arch = "riscv64")]
    {
        "riscv64"
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "riscv64"
    )))]
    {
        "x86_64"
    }
}

/// Run `rustc` with the given args and return trimmed stdout, or None on error.
fn run_capture(args: Vec<&str>) -> Option<String> {
    let output = Command::new("rustc").args(&args).output().ok()?;
    if !output.status.success() {
        eprintln!(
            "rustc {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn run_cargo(args: &[String]) -> ExitCode {
    let mut cmd = Command::new("cargo");
    cmd.args(args);
    println!("$ {}", render_cmd(&cmd));
    exec(cmd)
}

fn render_cmd(cmd: &Command) -> String {
    let prog = cmd.get_program().to_string_lossy();
    let arg_strs: Vec<String> = cmd
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    format!("{prog} {}", arg_strs.join(" "))
}

fn exec(mut cmd: Command) -> ExitCode {
    match cmd.status() {
        Ok(status) => {
            if status.success() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!(
                "failed to run `{}`: {e}",
                cmd.get_program().to_string_lossy()
            );
            ExitCode::FAILURE
        }
    }
}
