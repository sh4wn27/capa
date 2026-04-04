"use client";

interface AttentionHeatmapProps {
  weights:         number[][];
  donorLabels:     string[];
  recipientLabels: string[];
}

/** Interpolate white → navy based on normalised [0,1] weight */
function cellStyle(value: number, max: number): React.CSSProperties {
  const t = max > 0 ? value / max : 0;
  // White → deep navy (#0F1C35)
  const r = Math.round(255 - t * (255 - 15));
  const g = Math.round(255 - t * (255 - 28));
  const b = Math.round(255 - t * (255 - 53));
  return {
    backgroundColor: `rgb(${r},${g},${b})`,
    color:           t > 0.5 ? "#FFFFFF" : "#0F1C35",
  };
}

export default function AttentionHeatmap({
  weights,
  donorLabels,
  recipientLabels,
}: AttentionHeatmapProps) {
  const max = Math.max(...weights.flat(), 0.0001);

  return (
    <div className="overflow-auto scrollbar-thin">
      <table className="border-collapse w-full text-xs">
        <thead>
          <tr>
            <th className="w-20" />
            {recipientLabels.map((l) => (
              <th
                key={l}
                className="pb-2 px-1 font-medium text-muted-foreground text-center text-[10px] uppercase tracking-wider"
              >
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {weights.map((row, i) => (
            <tr key={donorLabels[i]}>
              <td className="pr-3 py-1 font-medium text-muted-foreground text-[10px] uppercase tracking-wider whitespace-nowrap text-right">
                {donorLabels[i]}
              </td>
              {row.map((val, j) => (
                <td
                  key={j}
                  title={`${donorLabels[i]} × ${recipientLabels[j]}: ${val.toFixed(4)}`}
                  style={cellStyle(val, max)}
                  className="w-12 h-12 text-center font-mono font-medium rounded-sm transition-all hover:ring-1 hover:ring-blush/50 cursor-default"
                >
                  {val.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Legend */}
      <div className="mt-4 flex items-center gap-2">
        <span className="text-[10px] text-muted-foreground">Low</span>
        <div
          className="h-2 flex-1 rounded-full"
          style={{
            background: "linear-gradient(to right, rgb(255,255,255), rgb(15,28,53))",
          }}
        />
        <span className="text-[10px] text-muted-foreground">High</span>
      </div>
    </div>
  );
}
