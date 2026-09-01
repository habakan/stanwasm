interface Props {
  data: Record<string, number | number[]>;
}

/** Scalar fields become `name = value` badges; equal-length vector fields become
 *  columns of one table. */
export function DataTable({ data }: Props) {
  const scalars = Object.entries(data).filter(
    ([, v]) => typeof v === "number",
  ) as [string, number][];
  const vectors = Object.entries(data).filter(([, v]) =>
    Array.isArray(v),
  ) as [string, number[]][];
  const nRows = vectors.length === 0 ? 0 : Math.max(...vectors.map(([, v]) => v.length));

  // 0.1-steps arrive as -0.899999999999999 from binary floating point, which
  // sets the column width to several times the data it carries.
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
