export interface CsvParseError {
  message: string;
}

export interface CsvParseResult {
  /** Stan data object ready to pass to StanModel. Numeric columns become
   *  vectors; row count is exposed as `N`, `J`, and `K`. */
  data: Record<string, number | number[]>;
  /** All header column names, in the order they appear. */
  columns: string[];
  /** Columns whose values are all numeric and that ended up in `data`. */
  numericColumns: string[];
  /** Columns dropped from `data` because at least one value wasn't numeric
   *  (Stan only accepts numbers; text columns become metadata in the table
   *  view but aren't passed to the model). */
  skippedColumns: string[];
}

/**
 * Parse a CSV string into Stan data. The CSV must have:
 *   - a header row (column names become Stan vector names)
 *   - data rows whose numeric columns can be parsed by `Number(...)`
 *
 * Text columns (non-numeric) are kept in `columns` for display but
 * skipped from `data`. Row count is auto-published as N / J / K.
 */
export function csvToData(text: string): CsvParseResult | CsvParseError {
  // Strip UTF-8 BOM that Excel and some editors add to CSV exports.
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);

  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length < 2) {
    return { message: "CSV needs a header row and at least one data row." };
  }

  const header = splitCsvLine(lines[0]);
  // Per-column raw strings (so we can decide numeric vs text after parsing).
  const rawCols: Record<string, string[]> = {};
  for (const c of header) rawCols[c] = [];

  for (let i = 1; i < lines.length; i++) {
    const row = splitCsvLine(lines[i]);
    if (row.length !== header.length) {
      return {
        message: `Row ${i} has ${row.length} fields but the header has ${header.length}.`,
      };
    }
    for (let j = 0; j < header.length; j++) {
      rawCols[header[j]].push(row[j]);
    }
  }

  const n = lines.length - 1;
  const data: Record<string, number | number[]> = { N: n, J: n, K: n };
  const numericColumns: string[] = [];
  const skippedColumns: string[] = [];

  for (const col of header) {
    const parsed: number[] = [];
    let allNumeric = true;
    for (const v of rawCols[col]) {
      if (v.length === 0) {
        allNumeric = false;
        break;
      }
      const num = Number(v);
      if (!Number.isFinite(num)) {
        allNumeric = false;
        break;
      }
      parsed.push(num);
    }
    if (allNumeric) {
      data[col] = parsed;
      numericColumns.push(col);
    } else {
      skippedColumns.push(col);
    }
  }

  if (numericColumns.length === 0) {
    return { message: "CSV has no numeric columns to pass to Stan." };
  }
  return { data, columns: header, numericColumns, skippedColumns };
}

function splitCsvLine(line: string): string[] {
  // Minimal split: no quoted-field support. Stan data is numeric so commas
  // in values aren't expected.
  return line.split(",").map((c) => c.trim());
}
