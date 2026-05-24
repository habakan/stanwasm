interface Props {
  values: number[];
  bins?: number;
  width?: number;
  height?: number;
}

/** Minimal SVG histogram, no chart library. */
export function Histogram({ values, bins = 30, width = 240, height = 60 }: Props) {
  if (values.length === 0) return <span>—</span>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max === min) {
    return <svg width={width} height={height}><line x1={0} y1={height/2} x2={width} y2={height/2} stroke="#c2410c" strokeWidth={2} /></svg>;
  }
  const range = max - min;
  const binW = range / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const i = Math.min(bins - 1, Math.floor((v - min) / binW));
    counts[i] += 1;
  }
  const maxCount = Math.max(...counts);
  const barW = width / bins;
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {counts.map((c, i) => {
        const h = (c / maxCount) * (height - 4);
        return (
          <rect
            key={i}
            x={i * barW}
            y={height - h}
            width={Math.max(1, barW - 1)}
            height={h}
            fill="#c2410c"
            opacity={0.85}
          />
        );
      })}
    </svg>
  );
}
