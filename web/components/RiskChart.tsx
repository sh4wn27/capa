"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface RiskChartProps {
  gvhd: number[];
  relapse: number[];
  trm: number[];
  timeBins?: number;
}

export default function RiskChart({
  gvhd,
  relapse,
  trm,
  timeBins = 100,
}: RiskChartProps) {
  const data = Array.from({ length: timeBins }, (_, i) => ({
    day: i,
    GvHD: gvhd[i] ?? 0,
    Relapse: relapse[i] ?? 0,
    TRM: trm[i] ?? 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="day" label={{ value: "Days", position: "insideBottom", offset: -4 }} />
        <YAxis
          domain={[0, 1]}
          label={{ value: "Cumulative Incidence", angle: -90, position: "insideLeft" }}
        />
        <Tooltip formatter={(v: number) => v.toFixed(3)} />
        <Legend />
        <Line type="monotone" dataKey="GvHD" stroke="#ef4444" dot={false} strokeWidth={2} />
        <Line type="monotone" dataKey="Relapse" stroke="#3b82f6" dot={false} strokeWidth={2} />
        <Line type="monotone" dataKey="TRM" stroke="#8b5cf6" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}
