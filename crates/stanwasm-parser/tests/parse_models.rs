//! Integration test: parse every .stan file under tests/models/ and assert
//! the resulting AST has at least one parameter or model statement.
//!
//! These are curated Stan model fixtures covering the language surface this
//! project supports. If any of them stops parsing here, that's a regression.

use stanwasm_parser::parse;
use std::fs;
use std::path::Path;

#[test]
fn all_reference_models_parse() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models");
    let entries: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir({}): {e}", dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "stan"))
        .collect();

    assert!(
        !entries.is_empty(),
        "no .stan files found in {}",
        dir.display()
    );

    let mut failures = Vec::new();
    let mut ok_count = 0;
    for entry in &entries {
        let path = entry.path();
        let src = fs::read_to_string(&path).unwrap();
        match parse(&src) {
            Ok(prog) => {
                if prog.parameters.is_empty() && prog.model.is_empty() && prog.data.is_empty() {
                    failures.push(format!(
                        "{}: parsed but produced empty program",
                        path.file_name().unwrap().to_string_lossy()
                    ));
                } else {
                    ok_count += 1;
                }
            }
            Err(e) => failures.push(format!(
                "{}: {e}",
                path.file_name().unwrap().to_string_lossy()
            )),
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} model(s) failed to parse:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
    assert_eq!(ok_count, entries.len(), "every model should parse");
    assert!(
        ok_count >= 20,
        "expected at least 20 models, got {ok_count}"
    );
}
