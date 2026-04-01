import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.CAPA_BACKEND_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  const body = await request.json();

  const response = await fetch(`${BACKEND_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    return NextResponse.json(
      { error: "Backend prediction failed" },
      { status: response.status }
    );
  }

  const data = await response.json();
  return NextResponse.json(data);
}
