interface Props {
  data: Record<string, number | number[]>;
}

/**
 * Renders Stan data in a spreadsheet-style layout:
 * - scalar fields appear as `name = value` badges at the top
 * - vector fields of equal length become columns of a single table
 */
export function DataTable({ data }: Props) {
  const scalars = Object.entries(data).filter(
    ([, v]) => typeof v === "number",
  ) as [string, number][];
  const vectors = Object.entries(data).filter(([, v]) =>
    Array.isArray(v),
  ) as [string, number[]][];
  const nRows = vectors.length === 0 ? 0 : Math.max(...vectors.map(([, v]) => v.length));

  // A CSV column of 0.1-steps arrives as -0.899999999999999 once it has been
  // through binary floating point, and printing that verbatim makes a column
  // several times wider than the data it carries. Six significant digits is
  // more than the display precision anywhere else in the app, and integers
  // and already-short values are left exactly as they were.
  const fmt = (v: number) =>
    Number.isInteger(v) || String(v).length <= 8 ? String(v) : Number(v.toPrecision(6)).toString();

  return (
    <div className="data-table-wrap">
      {scalars.length > 0 && (
        <div className="scalar-row">
          {scalars.map(([k, v]) => (
            <span key={k} className="scalar-badge">
              {k} = <strong>{v}</strong>
            </span>
          ))}
        </div>
      )}
      {nRows > 0 && (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th className="num">i</th>
                {vectors.map(([k]) => (
                  <th key={k}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: nRows }, (_, i) => (
                <tr key={i}>
                  <td className="num idx">{i + 1}</td>
                  {vectors.map(([k, vals]) => (
                    <td key={k} className="num">
                      {vals[i] === undefined ? "" : fmt(vals[i])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
