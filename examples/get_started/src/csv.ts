import type { Preset } from "./models";

export interface CsvParseError {
  message: string;
}

/**
 * Parse a CSV string into rows of numbers and map onto the preset's
 * expected schema. Returns the new `data` object (same shape as
 * `preset.data`) or an error.
 *
 * The CSV must have a header row whose column names exactly match
 * `preset.csvColumns`. All values must parse as finite numbers.
 * `preset.rowCountScalar` is auto-set to the number of data rows.
 */
export function csvToData(
  text: string,
  preset: Preset,
): { data: Record<string, number | number[]> } | CsvParseError {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !l.startsWith("#"));
  if (lines.length < 2) {
    return { message: "CSV needs a header row and at least one data row." };
  }
  const header = splitCsvLine(lines[0]);
  const expected = new Set(preset.csvColumns);
  const got = new Set(header);
  const missing = preset.csvColumns.filter((c) => !got.has(c));
  if (missing.length > 0) {
    return {
      message: `CSV is missing required column(s): ${missing.join(", ")}. Expected columns: ${preset.csvColumns.join(", ")}.`,
    };
  }

  const cols: Record<string, number[]> = {};
  for (const col of preset.csvColumns) cols[col] = [];

  for (let i = 1; i < lines.length; i++) {
    const row = splitCsvLine(lines[i]);
    if (row.length !== header.length) {
      return {
        message: `Row ${i} has ${row.length} fields but the header has ${header.length}.`,
      };
    }
    for (let j = 0; j < header.length; j++) {
      if (!expected.has(header[j])) continue;
      const v = Number(row[j]);
      if (!Number.isFinite(v)) {
        return {
          message: `Row ${i} column "${header[j]}" is not a finite number: ${row[j]}`,
        };
      }
      cols[header[j]].push(v);
    }
  }

  const n = cols[preset.csvColumns[0]].length;
  for (const col of preset.csvColumns) {
    if (cols[col].length !== n) {
      return {
        message: `Column "${col}" has ${cols[col].length} values but expected ${n}.`,
      };
    }
  }

  const data: Record<string, number | number[]> = {
    [preset.rowCountScalar]: n,
    ...cols,
  };
  return { data };
}

function splitCsvLine(line: string): string[] {
  // Minimal split: doesn't handle quoted commas. Stan data is numeric so
  // commas in values aren't expected.
  return line.split(",").map((c) => c.trim());
}
