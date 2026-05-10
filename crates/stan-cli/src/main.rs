//! Native CLI for local development and golden-value testing.
//! Phase 0: smoke test only.

fn main() -> anyhow::Result<()> {
    println!("stan-cli v{}", env!("CARGO_PKG_VERSION"));
    println!("(scaffolding only — implementation coming in later phases)");
    Ok(())
}
