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
  ReferenceLine,
} from "recharts";

// Wong (2011) colorblind-safe palette
const COLORS = {
  GvHD:    "#E69F00", // orange
  Relapse: "#0072B2", // blue
  TRM:     "#D55E00", // vermilion
};

interface RiskChartProps {
  gvhd:     number[];
  relapse:  number[];
  trm:      number[];
  timeBins?: number;
}

const CustomTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name: string; value: number; color: string }[];
  label?: number;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-white p-3 text-xs shadow-md">
      <p className="font-semibold text-navy mb-1.5">Day {label}</p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2 mb-0.5">
          <span
            className="h-2 w-2 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-muted-foreground">{entry.name}:</span>
          <span className="font-mono font-medium">{entry.value.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
};

export default function RiskChart({
  gvhd,
  relapse,
  trm,
  timeBins,
}: RiskChartProps) {
  const n = timeBins ?? Math.max(gvhd.length, relapse.length, trm.length, 1);
  const data = Array.from({ length: n }, (_, i) => ({
    day:     i,
    GvHD:    gvhd[i]    ?? 0,
    Relapse: relapse[i] ?? 0,
    TRM:     trm[i]     ?? 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 20 }}>
        <CartesianGrid strokeDasharray="2 4" stroke="#E5E8EF" />
        <XAxis
          dataKey="day"
          tick={{ fontSize: 11, fill: "#64748B" }}
          label={{ value: "Days post-transplant", position: "insideBottom", offset: -12, fontSize: 11, fill: "#64748B" }}
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          tick={{ fontSize: 11, fill: "#64748B" }}
          width={44}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
          iconType="circle"
          iconSize={8}
        />
        {/* Reference lines at key time points */}
        {[100, 365, 730].map((d) => (
          <ReferenceLine
            key={d}
            x={d}
            stroke="#E5E8EF"
            strokeDasharray="3 3"
            label={{ value: `${d}d`, fontSize: 9, fill: "#9CA3AF", position: "top" }}
          />
        ))}
        <Line
          type="monotone"
          dataKey="GvHD"
          stroke={COLORS.GvHD}
          dot={false}
          strokeWidth={2}
          activeDot={{ r: 4 }}
        />
        <Line
          type="monotone"
          dataKey="Relapse"
          stroke={COLORS.Relapse}
          dot={false}
          strokeWidth={2}
          activeDot={{ r: 4 }}
        />
        <Line
          type="monotone"
          dataKey="TRM"
          stroke={COLORS.TRM}
          dot={false}
          strokeWidth={2}
          activeDot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
