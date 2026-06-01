import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = (process.env.CAPA_BACKEND_URL ?? "http://localhost:8000").replace(/\/$/, "");

// 30-second timeout — model inference on CPU can be slow
const TIMEOUT_MS = 30_000;

export async function POST(request: NextRequest): Promise<NextResponse> {
  const t0 = Date.now();

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    console.error("[predict] Failed to parse request body");
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  // Validate that at least one HLA locus is present on each side
  const b = body as Record<string, Record<string, string | null> | undefined>;
  const donorValues = Object.values(b?.donor_hla  ?? {}).filter(Boolean);
  const recipValues = Object.values(b?.recipient_hla ?? {}).filter(Boolean);
  if (donorValues.length === 0 || recipValues.length === 0) {
    return NextResponse.json(
      { error: "At least one HLA locus must be provided for both donor and recipient." },
      { status: 422 },
    );
  }

  try {
    const upstream = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(TIMEOUT_MS),
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
      console.error("[predict] Backend error status=%d elapsed=%dms detail=%s",
        upstream.status, elapsed, detail);
      return NextResponse.json({ error: detail }, { status: upstream.status });
    }

    let data: unknown;
    try {
      data = JSON.parse(text);
    } catch {
      console.error("[predict] Backend returned non-JSON response elapsed=%dms", elapsed);
      return NextResponse.json(
        { error: "Backend returned non-JSON response" },
        { status: 502 },
      );
    }

    console.log("[predict] OK elapsed=%dms", elapsed);
    return NextResponse.json(data);
  } catch (err) {
    const elapsed = Date.now() - t0;

    if (err instanceof DOMException && err.name === "TimeoutError") {
      console.error("[predict] Upstream timeout elapsed=%dms", elapsed);
      return NextResponse.json(
        { error: `Prediction timed out after ${TIMEOUT_MS / 1000} s` },
        { status: 504 },
      );
    }

    const message = err instanceof Error ? err.message : String(err);
    console.error("[predict] Network error elapsed=%dms message=%s", elapsed, message);
    return NextResponse.json(
      { error: `Could not reach prediction backend: ${message}` },
      { status: 502 },
    );
  }
}

// Proxy the health endpoint so the frontend can poll backend readiness
export async function GET(): Promise<NextResponse> {
  try {
    const upstream = await fetch(`${BACKEND_URL}/health`, {
      signal: AbortSignal.timeout(5_000),
    });
    const data = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[predict/health] Backend unreachable: %s", message);
    return NextResponse.json({ status: "unreachable" }, { status: 503 });
  }
}
