import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = (process.env.CAPA_BACKEND_URL ?? "http://localhost:8000").replace(/\/$/, "");

// 30-second timeout — model inference on CPU can be slow
const TIMEOUT_MS = 30_000;

export async function POST(request: NextRequest): Promise<NextResponse> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  // Validate that at least one HLA locus is present on each side
  const b = body as Record<string, Record<string, string | null> | undefined>;
  const donorValues  = Object.values(b?.donor_hla  ?? {}).filter(Boolean);
  const recipValues  = Object.values(b?.recipient_hla ?? {}).filter(Boolean);
  if (donorValues.length === 0 || recipValues.length === 0) {
    return NextResponse.json(
      { error: "At least one HLA locus must be provided for both donor and recipient." },
      { status: 422 },
    );
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const upstream = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timer);

    // Forward the upstream response body verbatim (whether ok or not)
    const text = await upstream.text();

    if (!upstream.ok) {
      // Try to surface a structured error message from the backend
      let detail = `Backend error ${upstream.status}`;
      try {
        const parsed = JSON.parse(text) as { detail?: string };
        if (parsed.detail) detail = parsed.detail;
      } catch {
        // text is not JSON — use raw text
        if (text) detail = text;
      }
      return NextResponse.json({ error: detail }, { status: upstream.status });
    }

    // Parse and return the successful prediction
    let data: unknown;
    try {
      data = JSON.parse(text);
    } catch {
      return NextResponse.json(
        { error: "Backend returned non-JSON response" },
        { status: 502 },
      );
    }

    return NextResponse.json(data);
  } catch (err) {
    clearTimeout(timer);

    if (err instanceof DOMException && err.name === "AbortError") {
      return NextResponse.json(
        { error: `Prediction timed out after ${TIMEOUT_MS / 1000} s` },
        { status: 504 },
      );
    }

    // Network-level failure (backend not reachable)
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: `Could not reach prediction backend: ${message}` },
      { status: 502 },
    );
  }
}

// Optional: proxy the health endpoint so the frontend can poll it
export async function GET(): Promise<NextResponse> {
  try {
    const upstream = await fetch(`${BACKEND_URL}/health`, {
      signal: AbortSignal.timeout(5_000),
    });
    const data = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch {
    return NextResponse.json({ status: "unreachable" }, { status: 503 });
  }
}
