export interface CsvParseError {
  message: string;
}

export interface CsvParseResult {
  /** Stan data object ready to pass to StanModel. Columns become vectors;
   *  row count is exposed as `N`, `J`, and `K` so most idiomatic Stan size
   *  scalar names just work without renaming. */
  data: Record<string, number | number[]>;
  /** Column names as they appeared in the header (for display only). */
  columns: string[];
}

/**
 * Parse a CSV string into Stan data. The CSV must have:
 *   - a header row (column names become Stan vector names)
 *   - data rows of finite numbers
 *
 * No schema validation against any preset — the user is expected to edit
 * the Stan program so its `data { ... }` block matches the CSV columns.
 * Row count is auto-published as N / J / K.
 */
export function csvToData(text: string): CsvParseResult | CsvParseError {
  // Strip UTF-8 BOM that Excel and some editors add to CSV exports.
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);

  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !l.startsWith("#"));
  if (lines.length < 2) {
    return { message: "CSV needs a header row and at least one data row." };
  }

  const header = splitCsvLine(lines[0]);
  const cols: Record<string, number[]> = {};
  for (const c of header) cols[c] = [];

  for (let i = 1; i < lines.length; i++) {
    const row = splitCsvLine(lines[i]);
    if (row.length !== header.length) {
      return {
        message: `Row ${i} has ${row.length} fields but the header has ${header.length}.`,
      };
    }
    for (let j = 0; j < header.length; j++) {
      const v = Number(row[j]);
      if (!Number.isFinite(v)) {
        return {
          message: `Row ${i} column "${header[j]}" is not a finite number: ${row[j]}`,
        };
      }
      cols[header[j]].push(v);
    }
  }

  const n = cols[header[0]].length;
  // Publish the row count under common scalar names. Stan data blocks
  // typically declare one of these.
  const data: Record<string, number | number[]> = {
    N: n,
    J: n,
    K: n,
    ...cols,
  };
  return { data, columns: header };
}

function splitCsvLine(line: string): string[] {
  // Minimal split: no quoted-field support. Stan data is numeric so commas
  // in values aren't expected.
  return line.split(",").map((c) => c.trim());
}
