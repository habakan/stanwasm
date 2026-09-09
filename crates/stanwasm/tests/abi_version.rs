//! The module ABI number exists in two places — the emitter that stamps it into
//! a module, and the host that refuses a module not carrying its own. The host
//! keeps its own copy because the sampler-only build does not depend on the
//! emitter, so this is what keeps the two from drifting apart.

#[test]
fn the_host_and_the_emitter_agree_on_the_module_abi() {
    assert_eq!(
        stanwasm::ABI_VERSION,
        stanwasm_codegen::ABI_VERSION,
        "bump both, or neither: a module built by one and run by the other is \
         exactly what the number is for"
    );
}

#[test]
fn an_emitted_module_carries_the_number() {
    use stanwasm_runtime::{Env, Model};

    let mut data = Env::new();
    data.set_scalar("N", 3.0);
    data.set_vector("y", &[0.1, 0.2, 0.3]);
    let model = Model::parse_and_load(
        "data { int<lower=0> N; vector[N] y; }
         parameters { real mu; }
         model { mu ~ normal(0, 1); y ~ normal(mu, 1); }",
        data,
    )
    .unwrap();
    let compiled = stanwasm_codegen::compile(&model, &[0.1]).unwrap();

    let mut found = None;
    for payload in wasmparser::Parser::new(0).parse_all(&compiled.wasm) {
        if let wasmparser::Payload::ExportSection(section) = payload.unwrap() {
            for export in section {
                let export = export.unwrap();
                if export.name == "tapewasm_abi_version" {
                    found = Some(export.kind);
                }
            }
        }
    }
    assert_eq!(
        found,
        Some(wasmparser::ExternalKind::Global),
        "an emitted module exports no ABI version, so no host can check it"
    );
}
