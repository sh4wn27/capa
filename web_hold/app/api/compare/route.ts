import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = (process.env.CAPA_BACKEND_URL ?? "http://localhost:8000").replace(/\/$/, "");

// Declare this function's execution budget to Vercel's runtime.
// This is a synchronous proxy — the entire round-trip must finish
// within this window. For very large donor lists (>10) on CPU-only
// backends, consider splitting into parallel /predict calls client-side
// instead of relying on a single long-lived function invocation.
export const maxDuration = 60; // seconds

// Abort the upstream fetch if the backend hasn't replied in time.
// Must be less than maxDuration to leave room for serialisation overhead.
const UPSTREAM_TIMEOUT_MS = 55_000;

export async function POST(request: NextRequest): Promise<NextResponse> {
  const t0 = Date.now();

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    console.error("[compare] Failed to parse request body");
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  // Basic validation: require at least 2 donors
  const b = body as { donors?: unknown[] };
  const nDonors = Array.isArray(b?.donors) ? b.donors.length : 0;
  if (nDonors < 2) {
    return NextResponse.json(
      { error: "At least 2 donors are required for comparison." },
      { status: 422 },
    );
  }

  try {
    const upstream = await fetch(`${BACKEND_URL}/compare`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
      signal:  AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });

    const text = await upstream.text();
    const elapsed = Date.now() - t0;

    if (!upstream.ok) {
      let detail = `Backend error ${upstream.status}`;
      try {
        const parsed = JSON.parse(text) as { detail?: string };
        if (parsed.detail) detail = parsed.detail;
      } catch {
        if (text) detail = text;
      }
      console.error("[compare] Backend error status=%d donors=%d elapsed=%dms detail=%s",
        upstream.status, nDonors, elapsed, detail);
      return NextResponse.json({ error: detail }, { status: upstream.status });
    }

    let data: unknown;
    try {
      data = JSON.parse(text);
    } catch {
      console.error("[compare] Backend returned non-JSON response elapsed=%dms", elapsed);
      return NextResponse.json(
        { error: "Backend returned non-JSON response" },
        { status: 502 },
      );
    }

    console.log("[compare] OK donors=%d elapsed=%dms", nDonors, elapsed);
    return NextResponse.json(data);
  } catch (err) {
    const elapsed = Date.now() - t0;

    if (err instanceof DOMException && err.name === "TimeoutError") {
      console.error("[compare] Upstream timeout donors=%d elapsed=%dms", nDonors, elapsed);
      return NextResponse.json(
        { error: `Comparison timed out after ${UPSTREAM_TIMEOUT_MS / 1000} s` },
        { status: 504 },
      );
    }

    const message = err instanceof Error ? err.message : String(err);
    console.error("[compare] Network error donors=%d elapsed=%dms message=%s",
      nDonors, elapsed, message);
    return NextResponse.json(
      { error: `Could not reach prediction backend: ${message}` },
      { status: 502 },
    );
  }
}
