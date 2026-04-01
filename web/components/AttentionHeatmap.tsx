"use client";

interface AttentionHeatmapProps {
  weights: number[][];
  donorLabels: string[];
  recipientLabels: string[];
}

function cellColor(value: number, max: number): string {
  const intensity = Math.round((value / (max || 1)) * 220);
  return `rgb(${255 - intensity}, ${255 - intensity}, 255)`;
}

export default function AttentionHeatmap({
  weights,
  donorLabels,
  recipientLabels,
}: AttentionHeatmapProps) {
  const max = Math.max(...weights.flat());

  return (
    <div className="overflow-auto">
      <table className="border-collapse text-xs">
        <thead>
          <tr>
            <th />
            {recipientLabels.map((l) => (
              <th key={l} className="p-1 font-medium text-gray-600">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {weights.map((row, i) => (
            <tr key={donorLabels[i]}>
              <td className="p-1 font-medium text-gray-600 pr-2">{donorLabels[i]}</td>
              {row.map((val, j) => (
                <td
                  key={j}
                  title={val.toFixed(4)}
                  style={{ backgroundColor: cellColor(val, max) }}
                  className="w-10 h-10 text-center"
                >
                  {val.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
